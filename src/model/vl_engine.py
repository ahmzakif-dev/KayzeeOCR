"""Shared vision-language inference engine.

The layout detector (Stage I) and the content recognizer (Stage II) run the
exact same generation procedure: build a chat prompt, attach an image, generate,
and decode. ``VisionLanguageEngine`` owns the model and processor and exposes
that single procedure, so the weights are loaded once and reused by both stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for loading and running the vision-language model.

    Attributes:
        model_name: Hugging Face model id. Defaults to Qwen2.5-VL-2B; swap for
            Qwen3-VL-2B once it is available in your transformers build.
        device: Device map passed to ``from_pretrained`` ("auto", "cuda",
            "cpu", ...).
        torch_dtype: Torch dtype name ("auto", "float16", "bfloat16", ...).
        min_pixels: Lower bound on the visual-token budget (Qwen processor).
        max_pixels: Upper bound on the visual-token budget (Qwen processor).
    """

    model_name: str = "Qwen/Qwen2.5-VL-2B-Instruct"
    device: str = "auto"
    torch_dtype: str = "auto"
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28


class VisionLanguageEngine:
    """Load a Qwen-VL model once and run image+text generation on demand."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        """Initialize the engine; the model is loaded lazily on first use.

        Args:
            config: Engine configuration. Defaults to :class:`EngineConfig`.
        """
        self.config = config or EngineConfig()
        self._model: Any | None = None
        self._processor: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model and processor are currently in memory."""
        return self._model is not None and self._processor is not None

    def load(self) -> None:
        """Load the model and processor into memory.

        Idempotent: a no-op once the model is already loaded. Selects the
        Qwen2.5-VL or Qwen3-VL model class based on ``config.model_name``.
        """
        if self.is_loaded:
            return

        import torch
        from transformers import AutoProcessor

        model_cls = _resolve_model_class(self.config.model_name)
        logger.info("Loading vision-language model '%s' ...", self.config.model_name)
        self._model = model_cls.from_pretrained(
            self.config.model_name,
            torch_dtype=_resolve_dtype(self.config.torch_dtype),
            device_map=self.config.device,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
        )
        logger.info("Vision-language model loaded.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self) -> None:
        """Release the model and processor and free GPU memory."""
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int,
        do_sample: bool = False,
    ) -> str:
        """Run a single generation pass and return the decoded text.

        Loads the model first if necessary.

        Args:
            messages: Chat-format messages, e.g. as built by
                :meth:`image_text_message`.
            max_new_tokens: Maximum number of tokens to generate.
            do_sample: Whether to sample. ``False`` gives deterministic greedy
                decoding.

        Returns:
            The decoded model output with the prompt tokens removed.
        """
        self.load()
        from qwen_vl_utils import process_vision_info

        processor = self._processor
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        generated_ids = self._model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
        )
        new_tokens = [
            output[len(prompt):]
            for prompt, output in zip(inputs.input_ids, generated_ids)
        ]
        decoded = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""

    @staticmethod
    def image_text_message(
        image: Image.Image, text: str, role: str = "user"
    ) -> dict:
        """Build one chat message pairing an image with a text instruction.

        Args:
            image: The image to attach.
            text: The accompanying instruction text.
            role: The chat role for the message. Defaults to ``"user"``.

        Returns:
            A single chat-format message dict.
        """
        return {
            "role": role,
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }


def _resolve_model_class(model_name: str) -> Any:
    """Return the conditional-generation class matching ``model_name``.

    Args:
        model_name: The Hugging Face model id.

    Returns:
        The Qwen3-VL class when the name indicates Qwen3 and it is importable,
        otherwise the Qwen2.5-VL class.
    """
    if "qwen3" in model_name.lower():
        try:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        except ImportError:
            logger.warning(
                "Qwen3-VL class unavailable; falling back to the Qwen2.5-VL class."
            )

    from transformers import Qwen2_5_VLForConditionalGeneration

    return Qwen2_5_VLForConditionalGeneration


def _resolve_dtype(torch_dtype: str) -> Any:
    """Map a dtype name to a torch dtype, or pass through ``"auto"``.

    Args:
        torch_dtype: A dtype name such as ``"float16"`` or the string ``"auto"``.

    Returns:
        The resolved ``torch.dtype``, or the string ``"auto"``.
    """
    if torch_dtype == "auto":
        return "auto"

    import torch

    return getattr(torch, torch_dtype, "auto")


def create_engine(
    model_name: str | None = None, device: str = "auto"
) -> VisionLanguageEngine:
    """Convenience factory untuk penggunaan standalone atau testing.

    Pipeline utama (pipeline.py) membangun objek ini secara langsung.

    Contoh:
        engine = create_engine("Qwen/Qwen2.5-VL-2B-Instruct")

    Args:
        model_name: Optional model id override.
        device: Device map for the model.

    Returns:
        A configured (but not yet loaded) engine.
    """
    config = EngineConfig(device=device)
    if model_name:
        config.model_name = model_name
    return VisionLanguageEngine(config)
