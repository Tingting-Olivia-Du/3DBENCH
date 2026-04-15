"""
Qwen3-VL inference interface.

Supports both local inference and API-based inference.
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
import base64

from .base_vlm import BaseVLM, VLMResponse


class QwenVL(BaseVLM):
    """
    Qwen-VL / Qwen2.5-VL / Qwen3-VL inference.

    Supports:
    - Local inference with transformers
    - OpenAI-compatible API (e.g., vLLM, Ollama)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        use_api: bool = False,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        device: str = "cuda",
        max_new_tokens: int = 512,
    ):
        """
        Initialize Qwen-VL model.

        Args:
            model_name: HuggingFace model name or API model name
            use_api: Whether to use API instead of local inference
            api_base: API base URL (for OpenAI-compatible APIs)
            api_key: API key
            device: Device for local inference
            max_new_tokens: Maximum tokens to generate
        """
        super().__init__(model_name)
        self.use_api = use_api
        self.api_base = api_base
        self.api_key = api_key
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.model = None
        self.processor = None
        self.client = None

        if use_api:
            self._init_api_client()
        else:
            self._init_local_model()

    def _init_api_client(self):
        """Initialize OpenAI-compatible API client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key or "dummy",
        )

    def _init_local_model(self):
        """Initialize local transformers model."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch required: pip install transformers torch"
            )

        print(f"Loading model: {self.model_name}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        print(f"Model loaded on {self.device}")

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def inference(
        self,
        images: List[Union[str, Path]],
        prompt: str,
    ) -> VLMResponse:
        """
        Run inference on images with prompt.

        Args:
            images: List of image paths [agentview, eye_in_hand]
            prompt: Text prompt

        Returns:
            VLMResponse with raw text and parsed JSON
        """
        if self.use_api:
            return self._inference_api(images, prompt)
        else:
            return self._inference_local(images, prompt)

    def _inference_api(
        self,
        images: List[Union[str, Path]],
        prompt: str,
    ) -> VLMResponse:
        """Run inference using API."""
        # Build message with images
        content = []

        # Add images
        for i, img_path in enumerate(images):
            view_name = "agentview" if i == 0 else "eye_in_hand"
            img_b64 = self._encode_image(img_path)

            content.append({
                "type": "text",
                "text": f"Image {i+1} ({view_name}):"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}"
                }
            })

        # Add prompt
        content.append({
            "type": "text",
            "text": prompt
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.max_new_tokens,
                temperature=0.0,
            )

            raw_text = response.choices[0].message.content
            parsed = self.parse_json_response(raw_text)

            return VLMResponse(
                raw_text=raw_text,
                parsed_json=parsed,
                success=parsed is not None,
            )

        except Exception as e:
            return VLMResponse(
                raw_text="",
                parsed_json=None,
                success=False,
                error_message=str(e),
            )

    def _inference_local(
        self,
        images: List[Union[str, Path]],
        prompt: str,
    ) -> VLMResponse:
        """Run inference using local model."""
        try:
            from PIL import Image
            import torch
        except ImportError:
            raise ImportError("PIL and torch required")

        # Load images
        pil_images = [Image.open(str(p)).convert("RGB") for p in images]

        # Build conversation format for Qwen2-VL
        # Format: <image>text<image>text...
        conversation = []
        content = []

        for i, img in enumerate(pil_images):
            view_name = "agentview" if i == 0 else "eye_in_hand"
            content.append({
                "type": "image",
                "image": img,
            })
            content.append({
                "type": "text",
                "text": f"[Image {i+1}: {view_name}]\n"
            })

        content.append({
            "type": "text",
            "text": prompt
        })

        conversation.append({
            "role": "user",
            "content": content
        })

        # Process inputs
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=pil_images,
            padding=True,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Decode
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        # Parse JSON
        parsed = self.parse_json_response(raw_text)

        return VLMResponse(
            raw_text=raw_text,
            parsed_json=parsed,
            success=parsed is not None,
        )


class MockVLM(BaseVLM):
    """Mock VLM for testing without actual model."""

    def __init__(self, model_name: str = "mock"):
        super().__init__(model_name)

    def inference(
        self,
        images: List[Union[str, Path]],
        prompt: str,
    ) -> VLMResponse:
        """Return mock response based on prompt content."""
        import random

        if "3D coordinates" in prompt or "estimate" in prompt.lower():
            # Object localization
            parsed = {
                "x": round(random.uniform(0.3, 0.7), 3),
                "y": round(random.uniform(-0.3, 0.3), 3),
                "z": round(random.uniform(0.8, 1.0), 3),
            }
        elif "close and grasp" in prompt.lower():
            # Grasp timing
            parsed = {
                "can_close": random.choice([True, False]),
                "reason": "Mock response"
            }
        elif "direction" in prompt.lower():
            # Move direction
            dx, dy, dz = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
            norm = (dx**2 + dy**2 + dz**2) ** 0.5
            parsed = {
                "dx": round(dx / norm, 3),
                "dy": round(dy / norm, 3),
                "dz": round(dz / norm, 3),
            }
        else:
            parsed = {"response": "unknown question type"}

        raw_text = str(parsed)

        return VLMResponse(
            raw_text=raw_text,
            parsed_json=parsed,
            success=True,
        )
