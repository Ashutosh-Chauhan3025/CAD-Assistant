"""
vllm_planner.py
===============

Planner that talks to a remote vLLM server (Colab A100) over the
OpenAI-compatible /v1/chat/completions endpoint.

Runs on the FreeCAD machine; holds no GPU state, so it is safe to
instantiate one per worker process.

Difference from the original OpenAIPlannerChain: /v1/chat/completions is
STATELESS. There is no `previous_response_id`, so the full conversation is
kept locally in `self.messages` and resent each step.

config.json:
{
  "planner_backend": "vllm",
  "vllm_base_url": "https://<something>.trycloudflare.com/v1",
  "vllm_api_key": "sgpbench-<the key you passed to vllm serve>",
  "qwen_model_id": "Qwen/Qwen3-VL-8B-Instruct",
  "qwen_max_new_tokens": 1536,
  "qwen_sampling": false,
  "qwen_max_history_turns": 16,
  "qwen_max_history_images": 1,
  "request_timeout": 600
}
"""

import base64
import json
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

from openai import OpenAI
from PIL import Image

from .prompts import (
    create_followup_prompt,
    create_init_prompt,
    developer_prompt,
    system_prompt,
)


class PlannerError(RuntimeError):
    """Raised instead of silently swallowing a failed request."""


class VLLMPlannerChain:
    def __init__(self, config_path: str = "config.json", prompt_name: str = "default"):
        self.config = self._load_config(config_path)

        self.client = OpenAI(
            base_url=self.config["vllm_base_url"],
            api_key=self.config.get("vllm_api_key", "EMPTY"),
            timeout=float(self.config.get("request_timeout", 600)),
            max_retries=5,          # tunnel hiccups are routine, not fatal
        )
        self.model = self.config.get("qwen_model_id", "Qwen/Qwen3-VL-8B-Instruct")
        self.max_tokens = int(self.config.get("qwen_max_new_tokens", 1536))
        self.sampling = bool(self.config.get("qwen_sampling", False))
        self.max_history_turns = int(self.config.get("qwen_max_history_turns", 16))
        self.max_history_images = int(self.config.get("qwen_max_history_images", 1))

        self.prompt_name = prompt_name
        self.step_count = 0
        self.logdir = None                       # set by CADAssistantCore
        self.messages: List[Dict[str, Any]] = []  # replaces previous_response_id

    # ------------------------------------------------------------------
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    def _load_previous_step_image(self, previous_step: int) -> Optional[Dict[str, Any]]:
        if not self.logdir:
            print("Warning: planner.logdir not set — image feedback disabled")
            return None

        image_path = os.path.join(self.logdir, f"step_image_{previous_step}.png")
        if not os.path.exists(image_path):
            return None

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                return {
                    "type": "sketch_render",
                    "data": base64.b64encode(buffer.getvalue()).decode("utf-8"),
                    "file_path": image_path,
                    "metadata": {"step": previous_step, "format": "PNG", "size": img.size},
                }
        except Exception as e:
            print(f"Warning: Could not load previous step image: {e}")
            return None

    # ------------------------------------------------------------------
    def __call__(self, context: Dict[str, Any], iteration_history) -> Dict[str, Any]:
        if self.step_count == 0:
            result = self._parse_response(self._get_init_response(context))
        else:
            structured = self._get_followup_response(iteration_history)
            result = self._parse_response(structured["text"])
            if structured["image"]:
                result["image"] = structured["image"]
                result["metadata"] = structured["metadata"]

        self.step_count += 1
        return result

    # ------------------------------------------------------------------
    def _get_init_response(self, context: Dict[str, Any]) -> str:
        system_message = system_prompt()
        developer_message = developer_prompt(
            initialcode=context.get("env_init_response", ""),
            **self.config.get("freecad_documentation", {}),
        )
        user_message = create_init_prompt(context.get("user_input", ""), self.step_count)

        # Qwen's chat template expects a single system turn; the OpenAI version sent two.
        merged_system = "\n\n".join(m for m in (system_message, developer_message) if m)

        self.messages = [
            {"role": "system", "content": merged_system},
            {"role": "user", "content": [{"type": "text", "text": user_message}]},
        ]
        return self._generate()

    # ------------------------------------------------------------------
    def _get_followup_response(self, iteration_history) -> Dict[str, Any]:
        previous_step = len(iteration_history) - 1
        previous_image = self._load_previous_step_image(previous_step)

        content: List[Dict[str, Any]] = [
            {"type": "text", "text": create_followup_prompt(iteration_history)}
        ]
        if previous_image:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{previous_image['data']}"},
            })

        self.messages.append({"role": "user", "content": content})
        self._prune_history()

        return {
            "text": self._generate(),
            "image": previous_image,
            "metadata": {
                "step": previous_step + 1,
                "has_previous_image": previous_image is not None,
            },
        }

    # ------------------------------------------------------------------
    def _generate(self) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
        }
        if self.sampling:
            # Only meaningful if you actually want majority voting to vary.
            kwargs.update(temperature=0.7, top_p=0.8, extra_body={"top_k": 20})
        else:
            kwargs.update(temperature=0.0)

        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # Do NOT swallow this — a dead tunnel should fail the sample loudly,
            # not produce an empty plan that scores as a wrong answer.
            raise PlannerError(f"vLLM request failed: {e}") from e

        text = (resp.choices[0].message.content or "").strip()
        if resp.choices[0].finish_reason == "length":
            print(f"⚠️  hit max_tokens at step {self.step_count}; code block may be truncated")

        self.messages.append({"role": "assistant", "content": text})
        return text

    # ------------------------------------------------------------------
    def _prune_history(self) -> None:
        """Keep the conversation inside max-model-len across a 10-step loop."""
        system_turns = [m for m in self.messages if m["role"] == "system"]
        rest = [m for m in self.messages if m["role"] != "system"]

        if len(rest) > self.max_history_turns:
            rest = rest[-self.max_history_turns:]

        seen = 0
        for msg in reversed(rest):
            if not isinstance(msg.get("content"), list):
                continue
            new_content = []
            for part in msg["content"]:
                if part.get("type") == "image_url":
                    seen += 1
                    if seen > self.max_history_images:
                        new_content.append(
                            {"type": "text", "text": "[render from an earlier step, omitted]"}
                        )
                        continue
                new_content.append(part)
            msg["content"] = new_content

        self.messages = system_turns + rest

    # ------------------------------------------------------------------
    def _parse_response(self, content: str) -> Dict[str, str]:
        try:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            action_idx = content.find("ACTION")
            plan = content[:action_idx].strip() if action_idx != -1 else content.strip()

            blocks = re.findall(r"```python\s*(.*?)\s*```", content, re.DOTALL)
            if not blocks:  # Qwen sometimes drops the language tag
                blocks = re.findall(r"```\s*(.*?)\s*```", content, re.DOTALL)

            return {"plan": plan, "action": "\n".join(blocks).strip() if blocks else ""}
        except Exception as e:
            return {"plan": f"Failed to parse response: {str(e)}", "action": ""}

    # ------------------------------------------------------------------
    def reset_steps(self):
        self.step_count = 0
        self.messages = []   # must clear, or state leaks into the next sample


# ----------------------------------------------------------------------
def build_planner(config_path: str = "config.json", prompt_name: str = "default"):
    with open(config_path) as f:
        backend = json.load(f).get("planner_backend", "openai")

    if backend == "vllm":
        return VLLMPlannerChain(config_path, prompt_name)
    if backend == "qwen":
        from .qwen_planner import QwenPlannerChain   # local-GPU variant
        return QwenPlannerChain(config_path, prompt_name)

    from .openai_planner import OpenAIPlannerChain
    return OpenAIPlannerChain(config_path, prompt_name)