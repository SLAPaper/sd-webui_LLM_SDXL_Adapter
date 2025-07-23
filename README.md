# SD Webui LLM SDXL Adapter

Implement Gemini3-1B Adapter for SDXL, porting from original ComfyUI custom node.

## Examples

More examples are welcome! Please submit PRs or send to issue if you don't know how to

Images|Prompt|Negative Prompt|Checkpoint|Without Adapter|With Adapter (1,1,0.5)|Notes
---|---|---|---|---|---|---
Panda Riding Motorcycle|`no humans, animal focus, clothed animal, solo, anthro humanoid male panda riding yellow offroad motorcycle, wear blue with white stripes interchange jacket and pants, wear offroad motorcycle helmet, wear protective boots and protective gloves, speeding, detailed background, masterpiece, newest, absurdres, safe`|`1girl, 1boy, foot down, standing position, stoped, low quality, worst quality, normal quality, text, signature, jpeg artifacts, bad anatomy, old, early, mini skirt, nsfw, chibi, multiple girls, multiple boys, multiple tails, multiple views, copyright name, watermark, artist name, signature`|`furrytoonmix_xlIllustrious`|![without adapter](img/grid-0184-2025-07-23%2016-53-32.webp)|![with adapter(1,1,0.5)](img/grid-0185-2025-07-23%2016-53-49.webp)|Notice the color of the motorcycle and the clothes
Panda Cafe|`no humans, animal focus, clothed animal, solo, anthro humanoid male panda in coffee shop at counter, have mechanical arms, (mechanical legs and mechanical boots:0.5), wear short sleeves red hoodie saying "404" and black shorts, dim lighting, detailed background, masterpiece, newest, absurdres, safe`|`1girl, 1boy, low quality, worst quality, normal quality, text, signature, jpeg artifacts, bad anatomy, old, early, mini skirt, nsfw, chibi, multiple girls, multiple boys, multiple tails, multiple views, copyright name, watermark, artist name, signature`|`furrytoonmix_xlIllustrious`|![without adapter](img/grid-0211-2025-07-23%2017-22-21.webp)|![with adapter(1,1,0.5)](img/grid-0212-2025-07-23%2017-22-35.webp)|Notice the pose variety and mechanical parts
Panda Rooftop|`no humans, animal focus, clothed animal, solo, anthro humanoid male panda sitting on stairs on rooftop, have mechanical arms, mechanical legs and mechanical boots, wear short sleeves red hoodie saying "404" and black shorts, looking afar, city below, backlighting, night, moonlight, starry sky, shooting star, detailed background, masterpiece, newest, absurdres, safe`|`1girl, 1boy, looking back, low quality, worst quality, normal quality, text, signature, jpeg artifacts, bad anatomy, old, early, mini skirt, nsfw, chibi, multiple girls, multiple boys, multiple tails, multiple views, copyright name, watermark, artist name, signature`|`furrytoonmix_xlIllustrious`|![without adapter](img/grid-0213-2025-07-23%2017-23-50.webp)|![with adapter(1,1,0.5)](img/grid-0214-2025-07-23%2017-24-04.webp)|Notice the text writing
Marine Panda|`no humans, animal focus, clothed animal, solo, anthro humanoid male panda, blue camouflage, long sleeves, tactical helmet, red armband, gloves, tactical pants, bulletproof vest, ragged battle boots, knee pads, full body, looking at viewer, smile, ocean, metal railing, metal floor, warship, marine, detailed background, masterpiece, newest, absurdres, safe`|`1girl, 1boy, bald crotch, low quality, worst quality, normal quality, text, signature, jpeg artifacts, bad anatomy, old, early, mini skirt, nsfw, chibi, multiple girls, multiple boys, multiple tails, multiple views, copyright name, watermark, artist name, signature`|`furrytoonmix_xlIllustrious`|![without adapter](img/grid-0191-2025-07-23%2017-13-47.webp)|![with adapter(1,1,0.5)](img/grid-0192-2025-07-23%2017-14-02.webp)|Notice the warship in the background
Fire Dragon|`no humans, animal focus, colossal fierce dragon, dark gray scales, fiery red markings, glowing red eyes, sharp horns, fangs, claws, flames emanating from face, partially spread wings with red accents, partially submerged in turbulent ocean waves, damaged burning ship with sails in background, stormy night sky, dramatic cinematic lighting, 3d pixel art, voxel style, ray tracing, intense supernatural horror atmosphere, masterpiece, newest, absurdres, safe`|`1girl, 1boy, low quality, worst quality, normal quality, text, signature, jpeg artifacts, bad anatomy, old, early, mini skirt, nsfw, chibi, multiple girls, multiple boys, multiple tails, multiple views, copyright name, watermark, artist name, signature`|`furrytoonmix_xlIllustrious`|![without adapter](img/grid-0203-2025-07-23%2017-18-18.webp)|![with adapter(1,1,0.5)](img/grid-0204-2025-07-23%2017-18-38.webp)|Notice the ship in the background
Cyber Rabbit|`no humans, animal focus, clothed animal, intricate details, side view shot of a cyberpunk harsh skinny anthro humanoid male white rabbit, dressed in a very silky tactical clothes, one hand in pocket, bokeh, realistic, rembrandt lighting, L USM, polychromatic, detailed background, masterpiece, newest, absurdres, safe`|`1girl, 1boy, low quality, worst quality, normal quality, text, signature, jpeg artifacts, bad anatomy, old, early, mini skirt, nsfw, chibi, multiple girls, multiple boys, multiple tails, multiple views, copyright name, watermark, artist name, signature`|`furrytoonmix_xlIllustrious`|![without adapter](img/grid-0217-2025-07-23%2017-26-41.webp)|![with adapter(1,1,0.5)](img/grid-0218-2025-07-23%2017-26-55.webp)|Notice "one hand in pocket"

---

Below is the ComfyUI readme:

# ComfyUI LLM SDXL Adapter

![Version](https://img.shields.io/badge/version-1.2.2-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-orange.svg)

A comprehensive set of ComfyUI nodes for using Large Language Models (LLM) as text encoders for SDXL image generation through a trained adapter.

<img width="1803" height="904" alt="image" src="https://github.com/user-attachments/assets/e8e5f047-37e7-4f8b-9bbd-78d70e2a7d80" />

[Image with workflow](https://files.catbox.moe/om6tc4.png)


## 🎯 Available Adapters

### RouWei-Gemma Adapter 
Trained adapter for using Gemma-3-1b as text encoder for [Rouwei v0.8](https://civitai.com/models/950531) (vpred or epsilon or [base](https://huggingface.co/Minthy/RouWei-0.8/blob/main/rouwei_080_base_fp16.safetensors)).

**Download Links:**
- [CivitAI Model](https://civitai.com/models/1782437)
- [HuggingFace Repository](https://huggingface.co/Minthy/RouWei-Gemma)

## 📦 Installation

### Requirements
- Python 3.8+
- ComfyUI
- Latest transformers library (tested on 4.53.1)

### Install Dependencies
```bash
pip install transformers>=4.53.1 safetensors einops torch
```

### Install Nodes
1. Clone the repository to `ComfyUI/custom_nodes/`:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/NeuroSenko/ComfyUI_LLM_SDXL_Adapter.git
```

2. Restart ComfyUI

### Setup RouWei-Gemma Adapter

1. **Download the adapter:**
   - Download from [CivitAI](https://civitai.com/models/1782437) or [HuggingFace](https://huggingface.co/Minthy/RouWei-Gemma)
   - Place the adapter file in `ComfyUI/models/llm_adapters/`

2. **Download Gemma-3-1b-it model:**
   - Download [gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) ([non-gated mirror](https://huggingface.co/unsloth/gemma-3-1b-it))
   - Place in `ComfyUI/models/llm/gemma-3-1b-it/`
   - **Note:** You need ALL files from the original model for proper functionality (not just .safetensors)

3. **Download Rouwei checkpoint:**
   - Get [Rouwei v0.8](https://civitai.com/models/950531) (vpred, epsilon, or [base](https://huggingface.co/Minthy/RouWei-0.8/blob/main/rouwei_080_base_fp16.safetensors)) if you don't have it
   - Place in your regular ComfyUI checkpoints folder

## 📁 File Structure Example

```
ComfyUI/models/
├── llm/gemma-3-1b-it/
│   ├── added_tokens.json
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer.model
│   └── tokenizer_config.json
├── llm_adapters/
│   └── rouweiGemma_g31b27k.safetensors
└── checkpoints/
    └── rouwei_v0.8_vpred.safetensors
```

## 🔍 Debugging

To enable detailed logging, edit `__init__.py`:
```python
# Change from:
logger.setLevel(logging.WARN)
# To:
logger.setLevel(logging.INFO)
```