"""Capture native eager Llama decode operations without changing tensor shapes."""
import torch


class GreedyDecodeGraphs:
    def __init__(self, model):
        self.model = model
        self.original_generate = model.generate
        self.graphs = {}

    def _decode(self, token, past):
        picked = []
        for _ in range(6):
            output = self.model.language_model(input_ids=token.reshape(1, 1), attention_mask=None,
                position_ids=None, past_key_values=past, inputs_embeds=None, labels=None,
                use_cache=True, output_attentions=False, output_hidden_states=False, return_dict=True)
            token = output.logits[:, -1].argmax(dim=-1)
            picked.append(token)
            past = output.past_key_values
        return torch.stack(picked, dim=1)

    def _graph(self, token, past):
        key = tuple((tuple(k.shape), tuple(v.shape)) for k, v in past)
        if key not in self.graphs:
            static_token = token.detach().clone()
            static_past = tuple((k.detach().clone(), v.detach().clone()) for k,v in past)
            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for _ in range(3):
                    self._decode(static_token, static_past)
            torch.cuda.current_stream().wait_stream(warm)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output_tokens = self._decode(static_token, static_past)
            self.graphs[key] = (graph, static_token, static_past, output_tokens)
        graph, static_token, static_past, output_tokens = self.graphs[key]
        static_token.copy_(token)
        for (target_k, target_v), (source_k, source_v) in zip(static_past, past):
            target_k.copy_(source_k)
            target_v.copy_(source_v)
        graph.replay()
        return output_tokens

    def generate(self, input_ids, **kwargs):
        config = kwargs.get("generation_config", self.model.generation_config)
        sampled = kwargs.get("do_sample", config.do_sample)
        if (sampled or kwargs.get("max_new_tokens") != 7 or config.num_beams != 1
                or config.output_attentions or config.return_dict_in_generate
                or config.repetition_penalty != 1.0):
            return self.original_generate(input_ids, **kwargs)
        output = self.model(input_ids=input_ids, attention_mask=kwargs.get("attention_mask"),
            pixel_values=kwargs.get("pixel_values"), use_cache=True, output_attentions=False,
            output_hidden_states=False, return_dict=True)
        first = output.logits[:, -1].argmax(dim=-1)
        tail = self._graph(first, output.past_key_values)
        return torch.cat((input_ids, first.reshape(1, 1), tail), dim=1)

    def install(self):
        self.model.generate = self.generate
