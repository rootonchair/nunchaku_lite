# nunchaku_lite

`nunchaku_lite` loads Nunchaku-quantized image generation components into
standard Diffusers pipelines. It keeps the integration surface small: construct
the target Diffusers pipeline, replace only the quantized transformer or UNet,
and keep scheduling, prompting, LoRA loading, and image generation in regular
Diffusers code.

## Start Here

- [API Reference](api.md) covers the public loading and adapter APIs.
- [Development Guide](development.md) covers local validation, adapter authoring,
  and runtime LoRA implementation.
- [Roadmap](roadmap.md) tracks model support and remaining feature work.
- [Documentation Deployment](deployment.md) explains how to update and publish
  this documentation site.

## Supported Families

| Model family | Adapter target | Runtime LoRA |
| --- | --- | --- |
| FLUX.1 | `flux` | Yes |
| FLUX.2 Klein | `flux2` | Yes |
| Qwen-Image and Qwen-Image-Edit | `qwen_image` | Yes |
| SDXL and SDXL-Turbo | `sdxl` | Not yet |
| Z-Image Turbo | `z_image` | Yes |

Runnable model examples are stored in the repository under `examples/`.
