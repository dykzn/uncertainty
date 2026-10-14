"""Native full-vocabulary multinomial sampling with captured eager decode."""
import torch
from graphed_decode import GreedyDecodeGraphs


class SampledDecodeGraphs(GreedyDecodeGraphs):
    def __init__(self, model):
        super().__init__(model)
        self.sample_graphs = {}
        self.action_end = model.vocab_size
        self.action_begin = self.action_end - model.config.n_action_bins

    def _pick(self, logits):
        # Retain the full vocabulary tensor and native multinomial call: reducing
        # to 256 entries would change the random-number consumption pattern.
        masked = torch.full_like(logits, float("-inf"))
        masked[:, self.action_begin:self.action_end] = logits[:, self.action_begin:self.action_end]
        probabilities = torch.nn.functional.softmax(masked, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(1)

    def _sample_decode(self, token, past):
        picked = []
        for _ in range(6):
            output = self.model.language_model(input_ids=token.reshape(1, 1), attention_mask=None,
                position_ids=None, past_key_values=past, inputs_embeds=None, labels=None,
                use_cache=True, output_attentions=False, output_hidden_states=False, return_dict=True)
            token = self._pick(output.logits[:, -1])
            picked.append(token)
            past = output.past_key_values
        return torch.stack(picked, dim=1)

    def _sample_graph(self, token, past):
        key = tuple((tuple(k.shape), tuple(v.shape)) for k, v in past)
        if key not in self.sample_graphs:
            # Warmup/capture must not consume the caller's live random stream.
            state = torch.cuda.get_rng_state()
            try:
                static_token = token.detach().clone()
                static_past = tuple((k.detach().clone(), v.detach().clone()) for k,v in past)
                warm = torch.cuda.Stream()
                warm.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warm):
                    for _ in range(3):
                        self._sample_decode(static_token, static_past)
                torch.cuda.current_stream().wait_stream(warm)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output_tokens = self._sample_decode(static_token, static_past)
                self.sample_graphs[key] = (graph, static_token, static_past, output_tokens)
            finally:
                torch.cuda.set_rng_state(state)
        graph, static_token, static_past, output_tokens = self.sample_graphs[key]
        static_token.copy_(token)
        for (target_k, target_v), (source_k, source_v) in zip(static_past, past):
            target_k.copy_(source_k)
            target_v.copy_(source_v)
        graph.replay()
        return output_tokens

    def generate(self, input_ids, **kwargs):
        config = kwargs.get("generation_config", self.model.generation_config)
        sampled = kwargs.get("do_sample", config.do_sample)
        if not sampled:
            return super().generate(input_ids, **kwargs)
        processors = kwargs.get("logits_processor", [])
        supported = (kwargs.get("max_new_tokens") == 7 and config.num_beams == 1
                     and not config.output_attentions and not config.return_dict_in_generate
                     and config.repetition_penalty == 1.0
                     and kwargs.get("temperature", config.temperature) == 1.0
                     and kwargs.get("top_k", config.top_k) == 0
                     and kwargs.get("top_p", config.top_p) == 1.0
                     and len(processors) == 1 and type(processors[0]).__name__ == "ActionOnly")
        if not supported:
            return self.original_generate(input_ids, **kwargs)
        output = self.model(input_ids=input_ids, attention_mask=kwargs.get("attention_mask"),
            pixel_values=kwargs.get("pixel_values"), use_cache=True, output_attentions=False,
            output_hidden_states=False, return_dict=True)
        first = self._pick(output.logits[:, -1])
        tail = self._sample_graph(first, output.past_key_values)
        return torch.cat((input_ids, first.reshape(1, 1), tail), dim=1)
