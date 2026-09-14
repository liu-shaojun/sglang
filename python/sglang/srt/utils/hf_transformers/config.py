# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Config loading utilities."""

import os
from pathlib import Path
from typing import Optional

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from sglang.srt.configs.model_config_parser_registry import (
    ModelConfigParserBase,
    get_model_config_parser,
    register_model_config_parser,
)
from sglang.srt.connector import create_remote_connector
from sglang.srt.utils import is_remote_url, lru_cache_frozenset

from ..hf_transformers_patches import _ensure_gguf_version
from .common import (
    _CONFIG_REGISTRY,
    AutoConfig,
    DeepseekVLV2Config,
    _is_deepseek_ocr2_model,
    _is_deepseek_ocr_model,
    _override_v_head_dim_if_zero,
    check_gguf_file,
    get_hf_text_config,
    resolve_runai_obj_uri,
)
from .mistral_utils import is_mistral_model, load_mistral_config


def _set_architectures(config, arch_name):
    config.update({"architectures": [arch_name]})


def _apply_deepseek_ocr_overrides(config, model):
    _override_v_head_dim_if_zero(config)
    _set_architectures(config, "DeepseekOCRForCausalLM")
    config._name_or_path = model


@register_model_config_parser("hf")
class HfModelConfigParser(ModelConfigParserBase):
    def parse(
        self,
        model,
        trust_remote_code: bool,
        revision: Optional[str] = None,
        **kwargs,
    ):
        # XPU/qwen35 GGUF bypass: transformers' GGUF config parser
        # (load_gguf_checkpoint) rejects arch "qwen35"/"qwen35moe" with
        # "not supported yet". When SGLANG_GGUF_HF_CONFIG_DIR points at a sibling
        # HF checkpoint dir (config.json + tokenizer), read the config straight
        # from there and never hand transformers the .gguf path. The GGUF weights
        # are still loaded by the GGUF model loader; only the *config* is sourced
        # from HF here. Matches the _get_gguf_weights_map_xpu_qwen35 name-map
        # bypass in loader.py.
        import os as _os

        _hf_cfg_dir = _os.environ.get("SGLANG_GGUF_HF_CONFIG_DIR")
        if _hf_cfg_dir and str(model).endswith(".gguf"):
            model = _hf_cfg_dir
            kwargs.pop("gguf_file", None)

        config = AutoConfig.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )

        if (
            config.architectures is not None
            and config.architectures[0] == "GlmMoeDsaForCausalLM"
        ):
            # GlmMoeDsaConfig drops/clobbers raw checkpoint fields the DSA path
            # needs, so re-read them from config.json and restore. Fixed upstream
            # by https://github.com/huggingface/transformers/pull/46338; remove
            # this block once SGLang requires transformers >= 5.10.
            from transformers import PretrainedConfig

            raw_config, _ = PretrainedConfig.get_config_dict(model, revision=revision)
            for key in (
                "qk_rope_head_dim",
                "index_topk_freq",
            ):
                if key in raw_config:
                    setattr(config, key, raw_config[key])
            if hasattr(config, "qk_head_dim") and hasattr(config, "qk_nope_head_dim"):
                config.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        if (
            config.architectures is not None
            and config.architectures[0] == "Phi4MMForCausalLM"
        ):
            from transformers import SiglipVisionConfig

            config.vision_config = SiglipVisionConfig(
                hidden_size=1152,
                image_size=448,
                intermediate_size=4304,
                model_type="siglip_vision_model",
                num_attention_heads=16,
                num_hidden_layers=26,
                patch_size=14,
            )

        if config.architectures in [
            ["LongcatCausalLM"],
            ["LongcatFlashForCausalLM"],
            ["LongcatFlashNgramForCausalLM"],
        ]:
            config.model_type = "longcat_flash"

        text_config = get_hf_text_config(config=config)

        if isinstance(model, str) and text_config is not None:
            items = (
                text_config.items()
                if hasattr(text_config, "items")
                else vars(text_config).items()
            )
            for key, val in items:
                if not hasattr(config, key) and val is not None:
                    setattr(config, key, val)

        is_ocr = _is_deepseek_ocr_model(config)
        is_ocr2 = _is_deepseek_ocr2_model(config)

        if is_ocr2:
            _override_v_head_dim_if_zero(config)
            config.model_type = "deepseek-ocr"
            _set_architectures(config, "DeepseekOCRForCausalLM")
            config = DeepseekVLV2Config.from_pretrained(model, revision=revision)
            _apply_deepseek_ocr_overrides(config, model)
        elif config.model_type in _CONFIG_REGISTRY:
            model_type = config.model_type
            if model_type == "deepseek_vl_v2" and is_ocr:
                model_type = "deepseek-ocr"
            config = _CONFIG_REGISTRY[model_type].from_pretrained(
                model, revision=revision
            )

            # Re-check after reloading config from registry
            if _is_deepseek_ocr_model(config) or _is_deepseek_ocr2_model(config):
                _apply_deepseek_ocr_overrides(config, model)
            else:
                config._name_or_path = model

        if isinstance(model, str) and config.model_type == "internvl_chat":
            for key, val in config.llm_config.__dict__.items():
                if not hasattr(config, key):
                    setattr(config, key, val)

        if config.model_type == "multi_modality":
            _set_architectures(config, "MultiModalityCausalLM")

        if config.model_type in (
            "gemma4",
            "gemma4_assistant",
            "gemma4_unified",
            "gemma4_unified_assistant",
        ):
            # Gemma4 configs use base attributes for SWA layers and `global_*`
            # variants for full-attention layers.  SGLang expects the opposite:
            # base = full-attention, `swa_*` = sliding-window overrides.
            text_config = config.text_config
            global_head_dim = getattr(text_config, "global_head_dim", None)
            global_kv_heads = getattr(text_config, "num_global_key_value_heads", None)

            swa_head_dim = text_config.head_dim
            swa_kv_heads = text_config.num_key_value_heads

            text_config.swa_head_dim = swa_head_dim
            text_config.swa_v_head_dim = swa_head_dim
            text_config.swa_num_key_value_heads = swa_kv_heads

            if global_head_dim is not None:
                text_config.head_dim = global_head_dim
            if global_kv_heads is not None:
                text_config.num_key_value_heads = global_kv_heads

            if not hasattr(text_config, "v_head_dim"):
                text_config.v_head_dim = text_config.head_dim
            if not hasattr(text_config, "swa_v_head_dim"):
                text_config.swa_v_head_dim = text_config.swa_head_dim

            # Unified Gemma4 names the end-of-audio token `eoa_token_index`,
            # but the multimodal processor expects `eoa_token_id`.
            if not hasattr(config, "eoa_token_id") and hasattr(
                config, "eoa_token_index"
            ):
                config.eoa_token_id = config.eoa_token_index

        if config.model_type == "longcat_flash":
            _set_architectures(config, "LongcatFlashForCausalLM")

        return config


@register_model_config_parser("mistral")
class MistralModelConfigParser(ModelConfigParserBase):
    def parse(
        self,
        model,
        trust_remote_code: bool,
        revision: Optional[str] = None,
        **kwargs,
    ):
        del kwargs
        return load_mistral_config(
            model, trust_remote_code=trust_remote_code, revision=revision
        )


@lru_cache_frozenset(maxsize=32)
def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    model_config_parser: str = "auto",
    **kwargs,
):
    is_gguf = check_gguf_file(model)
    if is_gguf:
        if model_config_parser not in ("auto", "hf"):
            raise ValueError(
                f"model_config_parser={model_config_parser!r} is incompatible "
                "with GGUF inputs; only 'hf' (or 'auto') is supported."
            )
        # XPU/qwen35 GGUF bypass: transformers' GGUF config parser
        # (load_gguf_checkpoint) rejects arch "qwen35"/"qwen35moe" with
        # "not supported yet". When SGLANG_GGUF_HF_CONFIG_DIR points at a sibling
        # HF checkpoint dir (config.json + tokenizer), read the config straight
        # from there and NEVER pass gguf_file= to transformers. The GGUF weights
        # are still loaded by the GGUF model loader; only the *config* is sourced
        # from HF here. Matches the _get_gguf_weights_map_xpu_qwen35 name-map
        # bypass in loader.py.
        _hf_cfg_dir = os.environ.get("SGLANG_GGUF_HF_CONFIG_DIR")
        if _hf_cfg_dir:
            model = _hf_cfg_dir
        else:
            _ensure_gguf_version()
            kwargs["gguf_file"] = model
            model = Path(model).parent
        # Skip auto-resolution for GGUF: the name-based Mistral heuristic
        # would misfire on the rewritten parent dir.
        model_config_parser = "hf"

    model = resolve_runai_obj_uri(model)

    if is_remote_url(model):
        client = create_remote_connector(model)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        model = client.get_local_dir()

    if model_config_parser == "auto":
        # `model` is post-rewrite (gguf parent / runai uri / remote pull).
        model_config_parser = "mistral" if is_mistral_model(model) else "hf"

    parser = get_model_config_parser(model_config_parser)
    config = parser.parse(
        model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
    )

    if model_override_args:
        config.update(model_override_args)

    if is_gguf and not os.environ.get("SGLANG_GGUF_HF_CONFIG_DIR"):
        # Normal GGUF path: transformers' CausalLM name map gives the runtime
        # arch. Skipped under SGLANG_GGUF_HF_CONFIG_DIR — there we already read
        # the correct architectures (e.g. Qwen3_5MoeForConditionalGeneration)
        # straight from the sibling HF config.json and must NOT clobber it with
        # transformers' text-only CausalLM name (qwen3_5_moe →
        # Qwen3_5MoeForCausalLM, which has no SGLang registry entry).
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        _set_architectures(config, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type])

    if is_gguf and os.environ.get("SGLANG_GGUF_HF_CONFIG_DIR"):
        config = _maybe_gguf_gemma4_text_only(config)

    return config


def _maybe_gguf_gemma4_text_only(config):
    """Collapse a multimodal gemma4 config to its text tower for GGUF runs.

    The published gemma-4 GGUFs (26B-A4B and 31B) contain the TEXT tower only —
    there is not a single vision / audio tensor in the file — while the sibling
    HF ``config.json`` declares ``Gemma4ForConditionalGeneration``.  Running the
    multimodal entry would build vision (and, for the 26B, audio) towers that no
    weight ever fills.  ``Gemma4ForCausalLM`` is the text-only entry and its
    ``config_class`` is ``Gemma4TextConfig``, so hand it the nested text config
    with the generation-relevant top-level ids carried over.

    Set ``SGLANG_GGUF_GEMMA4_KEEP_MM=1`` to opt out and keep the multimodal arch.
    """
    if getattr(config, "model_type", None) != "gemma4":
        return config
    if os.environ.get("SGLANG_GGUF_GEMMA4_KEEP_MM", "0") == "1":
        return config
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return config

    # bos/eos live on the multimodal config; the text config carries only the
    # single-id defaults, which would break stop handling for the IT models.
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(config, attr, None)
        if value is not None:
            setattr(text_config, attr, value)
    _set_architectures(text_config, "Gemma4ForCausalLM")
    return text_config
