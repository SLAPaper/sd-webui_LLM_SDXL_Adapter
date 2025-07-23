# Copyright 2025 SLAPaper
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import pathlib
import typing as tg

import gradio as gr
import torch
from adapter_libs.llm_adapter_loader import LLMAdapterLoader
from adapter_libs.llm_model_loader import LLMModelLoader
from adapter_libs.llm_text_encoder import LLMTextEncoder
from adapter_libs.llm_to_sdxl_adapter import LLMToSDXLAdapter

from modules import devices, errors, script_callbacks, scripts
from modules.processing import Processed, StableDiffusionProcessing
from modules.shared import cmd_opts


logger = logging.getLogger("LLM-SDXL-Adapter")
logger.setLevel(logging.INFO)

curr_path: pathlib.Path = pathlib.Path(__file__).parent.parent.absolute()
model_path: pathlib.Path = curr_path / "models"
gemma_path = model_path / "gemma-3-1b-it"
adapter_path = model_path / "adapter" / "rw_gemma_3_1_39k.safetensors"

if (
    hasattr(cmd_opts, "llm_sdxl_adapter_gemma_path")
    and cmd_opts.llm_sdxl_adapter_gemma_path
):
    gemma_path = cmd_opts.llm_sdxl_adapter_gemma_path

if (
    hasattr(cmd_opts, "llm_sdxl_adapter_model_path")
    and cmd_opts.llm_sdxl_adapter_model_path
):
    adapter_path = cmd_opts.llm_sdxl_adapter_model_path

logger.info(f"LLM SDXL Adapter Gemma Path: {gemma_path}, Adapter Path: {adapter_path}")


def set_nan_to_zero(x: torch.Tensor, tag: str = "") -> torch.Tensor:
    """
    Set NaN values in the tensor to zero
    Args:
        x: Input tensor
        tag: Optional tag for logging
    Returns:
        Tensor with NaN values replaced by zero
    """
    if torch.isnan(x).any():
        logger.warning("NaN values found in tensor [{tag}], replacing with zero")
        x = torch.nan_to_num(x, nan=0.0)
    return x


def apply_llm_adaptation(
    sdxl_embeds: tuple[torch.Tensor, torch.Tensor],
    llm_embeds: tuple[torch.Tensor, torch.Tensor],
    original_strength: float,
    llm_strength: float,
    pooled_llm_strength: float,
    tag: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply LLM adaptation to SDXL embeddings with adapter and strength control
    Args:
        sdxl_embeds: SDXL text encoder outputs, ([batch_size, seq_len, embed_dim], [batch_size, embed_dim])
        llm_embeds: LLM embeddings ([1, seq_len, embed_dim], [1, embed_dim])
        original_strength: Original strength of SDXL embeddings (0-1)
        llm_strength: Adaptation strength (0-10)
        pooled_llm_strength: Strength for pooled embeddings (0-1)
        tag: Optional tag for logging
    Returns:
        Adapted embeddings { "cross_attn": [batch, seq_len, embed_dim], "vector": [batch, embed_dim]}
    """
    # cross_attn for normal conditioning
    if llm_strength > 0:
        sdxl_cross_attn = sdxl_embeds[0]
        llm_cross_attn = set_nan_to_zero(llm_embeds[0], tag=f"{tag} cross_attn")
        batch_size = sdxl_cross_attn.shape[0]
        new_cross_attn = torch.cat(
            [
                original_strength * sdxl_cross_attn,
                llm_strength * llm_cross_attn.repeat(batch_size, 1, 1),
            ],
            dim=1,
        )
    else:
        new_cross_attn = sdxl_embeds[0]
        logger.info(f"LLM strength is zero, using original SDXL cross_attn {tag}")

    # vector for pooled conditioning
    if pooled_llm_strength > 0:
        sdxl_vector = sdxl_embeds[1]
        llm_vector = set_nan_to_zero(llm_embeds[1], tag=f"{tag} vector")
        batch_size = sdxl_vector.shape[0]

        # for some reason the llm_adapter output 1x1280 while SDXL expects 1x2816, so split first
        # It seems that the remaining 1536 dimensions are the extra embeddings for image size and other features
        # original_size, crop_topleft, target_size (256*3*2, sgm.modules.encoders.modules.ConcatTimestepEmbedderND)
        sdxl_front = sdxl_vector[:, : llm_vector.shape[1]]
        sdxl_back = sdxl_vector[:, llm_vector.shape[1] :]

        new_front = (
            1 - pooled_llm_strength
        ) * sdxl_front + pooled_llm_strength * llm_vector.repeat(batch_size, 1)

        new_vector = torch.cat([new_front, sdxl_back], dim=1)
    else:
        new_vector = sdxl_embeds[1]
        logger.info(f"Pooled LLM strength is zero, using original SDXL vector {tag}")

    return new_cross_attn, new_vector


class LLM_SDXL_Adapter:
    def __init__(self) -> None:
        self._is_initialized = False
        self.model: torch.nn.Module | None = None
        self.tokenizer: tg.Any = None
        self.text_encoder: LLMTextEncoder | None = None
        self.adapter: LLMToSDXLAdapter | None = None
        self.device: str = devices.get_optimal_device_name()
        self.last_prompt_hash: int | None = None
        self.last_negative_hash: int | None = None
        self.original_strength: float = 1.0
        self.llm_strength: float = 3.0
        self.pooled_llm_strength: float = 0.5
        self.prompt_embeds: tuple[torch.Tensor, torch.Tensor] | None = None
        self.negative_embeds: tuple[torch.Tensor, torch.Tensor] | None = None

    def _initialize_model(self) -> None:
        """Initialize model once and keep it in memory"""
        if self._is_initialized:
            return

        try:
            # Initialize loaders
            self.model_loader = LLMModelLoader()
            self.text_encoder = LLMTextEncoder()
            self.adapter_loader = LLMAdapterLoader()

            # Load base Gemma3 1B model
            model, tokenizer, _ = self.model_loader.load_model(
                gemma_path, device="cpu", force_reload=True
            )
            if model is None or tokenizer is None:
                raise RuntimeError("Failed to load model")

            # Load adapter
            adapter, info = self.adapter_loader.load_adapter(
                adapter_path, device=self.device
            )

            self.model = model
            self.tokenizer = tokenizer
            self.adapter = adapter
            self._is_initialized = True

            logger.info(f"Gemma3 1B model loaded successfully: {info}")

        except Exception as e:
            logger.error(f"Failed to initialize Gemma3 1B model: {e}")
            errors.display(e, "LLM SDXL Adapter Initialization")
            raise

    def _move_to_device(self) -> None:
        """Move model to current device (CPU or CUDA)"""
        if not self._is_initialized:
            return

        if self.model is not None:
            self.model = self.model.to(self.device)

        if self.adapter is not None:
            self.adapter = self.adapter.to(self.device)

    def _move_to_cpu(self) -> None:
        """Move model to offloading device (CPU)"""
        if not self._is_initialized:
            return

        if self.model is not None:
            self.model = self.model.to("cpu")

        if self.adapter is not None:
            self.adapter = self.adapter.to("cpu")

    def generate_embeddings(
        self, text: str
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Generate embeddings using LLMTextEncoder"""
        if not text or self.text_encoder is None or self.adapter is None:
            return None

        try:
            hidden_states, info = self.text_encoder.encode_text(
                self.model,
                self.tokenizer,
                text,
                system_prompt="Describe this image prompt in detail",
                device=self.device,
                skip_first=0,
            )
            logger.info(f"Generated embeddings: {info}")
            with torch.no_grad():
                compressed_sequence, pooled_output = self.adapter(hidden_states)
            return compressed_sequence, pooled_output
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            errors.display(e, "LLM SDXL Adapter Embedding Generation")
            return None

    def on_cfg_denoiser(self, params: script_callbacks.CFGDenoiserParams) -> None:
        """Callback for modifying text conditioning in CFG denoiser"""
        if not isinstance(params.text_cond, dict) or not isinstance(
            params.text_uncond, dict
        ):
            logger.warning("Text conditioning is not a dictionary, skipping adaptation")
            return

        # SDXL case
        if self.prompt_embeds is not None:
            cond_crossattn, cond_vector = apply_llm_adaptation(
                (params.text_cond["crossattn"], params.text_cond["vector"]),
                self.prompt_embeds,
                self.original_strength,
                self.llm_strength,
                self.pooled_llm_strength,
                tag="text_cond",
            )
            params.text_cond["crossattn"] = cond_crossattn
            params.text_cond["vector"] = cond_vector

        if self.negative_embeds is not None:
            uncond_crossattn, uncond_vector = apply_llm_adaptation(
                (params.text_uncond["crossattn"], params.text_uncond["vector"]),
                self.negative_embeds,
                self.original_strength,
                self.llm_strength,
                self.pooled_llm_strength,
                tag="text_uncond",
            )
            params.text_uncond["crossattn"] = uncond_crossattn
            params.text_uncond["vector"] = uncond_vector


G_LLM_SDXL_ADAPTER = LLM_SDXL_Adapter()
script_callbacks.on_cfg_denoiser(lambda p: G_LLM_SDXL_ADAPTER.on_cfg_denoiser(p))


class LLM_SDXL_Adapter_Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self) -> str:
        return "LLM SDXL Adapter"

    def show(self, is_img2img: bool) -> bool | object:
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool) -> tg.List[gr.components.Component]:
        with gr.Accordion("LLM SDXL Adapter", open=False):
            enabled = gr.Checkbox(
                value=False,
                label="Enable LLM SDXL Adapter",
                elem_id="llm_sdxl_adapter_enabled",
            )

            original_strength = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=1.0,
                label="Orignal Strength",
                elem_id="llm_sdxl_original_strength",
            )

            llm_strength = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=1.0,
                label="Adapter Strength",
                elem_id="llm_sdxl_adapter_strength",
            )

            pooled_llm_strength = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.5,
                label="Pooled LLM Strength",
                elem_id="llm_sdxl_pooled_llm_strength",
            )

            return [enabled, original_strength, llm_strength, pooled_llm_strength]

    def process(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        orignal_strength: float,
        llm_strength: float,
        pooled_llm_strength: float,
    ) -> None:
        if not enabled:
            # Clear prompt references
            G_LLM_SDXL_ADAPTER.prompt_embeds = None
            G_LLM_SDXL_ADAPTER.negative_embeds = None

            G_LLM_SDXL_ADAPTER.last_prompt_hash = None
            G_LLM_SDXL_ADAPTER.last_negative_hash = None
            return

        # Check if SDXL model
        if not hasattr(p, "sd_model") or not hasattr(p.sd_model, "conditioner"):
            logger.error("LLM SDXL Adapter only works with SDXL models")
            return

        # Ensure model is initialized
        if not G_LLM_SDXL_ADAPTER._is_initialized:
            try:
                G_LLM_SDXL_ADAPTER._initialize_model()
            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
                return

        # Check if prompts changed
        current_prompt_hash = hash(p.prompt)
        current_negative_hash = hash(p.negative_prompt)

        # Generate embeddings only if prompts changed
        if (
            current_prompt_hash != G_LLM_SDXL_ADAPTER.last_prompt_hash
            or current_negative_hash != G_LLM_SDXL_ADAPTER.last_negative_hash
        ):
            # Move to optimal device
            G_LLM_SDXL_ADAPTER._move_to_device()

            with torch.inference_mode(), devices.autocast():
                prompt_embeds = G_LLM_SDXL_ADAPTER.generate_embeddings(p.prompt)
                negative_embeds = G_LLM_SDXL_ADAPTER.generate_embeddings(
                    p.negative_prompt
                )

            G_LLM_SDXL_ADAPTER._move_to_cpu()

            # Store embeddings and hashes
            if prompt_embeds is not None:
                G_LLM_SDXL_ADAPTER.prompt_embeds = prompt_embeds
                G_LLM_SDXL_ADAPTER.last_prompt_hash = current_prompt_hash
            else:
                logger.error(
                    "Failed to generate prompt embeddings, adapter won't affect positive prompt"
                )

            if negative_embeds is not None:
                G_LLM_SDXL_ADAPTER.negative_embeds = negative_embeds
                G_LLM_SDXL_ADAPTER.last_negative_hash = current_negative_hash
            else:
                logger.error(
                    "Failed to generate negative embeddings, adapter won't affect negative prompt"
                )

            G_LLM_SDXL_ADAPTER.original_strength = orignal_strength
            G_LLM_SDXL_ADAPTER.llm_strength = llm_strength
            G_LLM_SDXL_ADAPTER.pooled_llm_strength = pooled_llm_strength

        if any(
            x is not None
            for x in (
                G_LLM_SDXL_ADAPTER.prompt_embeds,
                G_LLM_SDXL_ADAPTER.negative_embeds,
            )
        ):
            p.extra_generation_params.update(
                {
                    "LLM SDXL Adapter Enabled": True,
                    "LLM SDXL Orignal Strength": orignal_strength,
                    "LLM SDXL Adapter Strength": llm_strength,
                    "LLM SDXL Pooled LLM Strength": pooled_llm_strength,
                }
            )

            logger.info(
                f"LLM SDXL Adapter active with {orignal_strength=}, {llm_strength=}, {pooled_llm_strength=}"
            )

    def postprocess(
        self,
        p: StableDiffusionProcessing,
        processed: Processed,
        enabled: bool,
        original_strength: float,
        llm_strength: float,
        pooled_llm_strength: float,
    ) -> None:
        # Clear prompt references
        G_LLM_SDXL_ADAPTER.prompt_embeds = None
        G_LLM_SDXL_ADAPTER.negative_embeds = None

        G_LLM_SDXL_ADAPTER.last_prompt_hash = None
        G_LLM_SDXL_ADAPTER.last_negative_hash = None
