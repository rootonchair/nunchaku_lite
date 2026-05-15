# Overview

`nunchaku_lite` is organized around a small public API and model-specific
adapters. The usual workflow is:

1. Start from a standard Diffusers pipeline and model id.
2. Load a Nunchaku SVDQ checkpoint with `load_nunchaku_pipeline(...)`.
3. Use the resulting Diffusers pipeline normally for prompting, scheduling, and
   runtime LoRA loading.

The documentation in this section covers:

- [Supported models](models/flux.md): runnable model-specific loading guides.
- [API Reference](api.md): public loading, patching, and adapter registry APIs.
- [Roadmap](roadmap.md): supported model coverage and remaining feature work.
- [Documentation Deployment](deployment.md): docs update and publishing flow.
