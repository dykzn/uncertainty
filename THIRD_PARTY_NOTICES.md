# Third-party source notices

- `author_sources/` contains the pinned scoring source from [wangjy0116/Black-Box-UE-Hub](https://github.com/wangjy0116/Black-Box-UE-Hub), revision `adffd0046e45f1a4c70a03e0de37fcf0a348edd0`. Original files and hashes are preserved in its manifest. Only the graph reductions and weighted aggregation are used in action-space scoring.
- `vendor/openvla/` contains the evaluated Prismatic/OpenVLA inference implementation and selected preprocessing helpers, obtained from the anonymous VLA source listed in `validation/export_provenance.json`. The model implementation and preprocessing function bodies are preserved; training components and attention-scoring dependencies are excluded. The upstream OpenVLA project is [openvla/openvla](https://github.com/openvla/openvla).
- `vendor/libero/` contains simulator Python code from the LIBERO installation used in the evaluation. The original MIT license is retained in `vendor/libero/LICENSE`. Upstream: [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).
- Benchmark meshes, textures, BDDL files and initial states download separately from [aniri15/LIBERO-REFLECT](https://github.com/aniri15/LIBERO-REFLECT), pinned revision `3b8e6eb2501969b441fdaaf9da32c14ba77e4bfd`.
- Checkpoints download separately from the Hugging Face repositories and revisions in `config/checkpoints.json`; this code repository does not contain model weights.

This private research export does not apply a new blanket license to third-party sources. Retain their original notices when redistributing code or assets.
