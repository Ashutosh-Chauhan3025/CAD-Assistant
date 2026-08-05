#!/usr/bin/env python3
"""Verify the Colab vLLM endpoint before starting a benchmark run."""

import base64, json, sys, time
from io import BytesIO

from PIL import Image, ImageDraw
from openai import OpenAI
import openai as oai

cfg = json.load(open("config.json"))
BASE = cfg["vllm_base_url"]
KEY = cfg.get("vllm_api_key", "EMPTY")
MODEL = cfg.get("qwen_model_id", "Qwen/Qwen3-VL-8B-Instruct")

print(f"endpoint : {BASE}\nmodel    : {MODEL}\n" + "-" * 60)
client = OpenAI(base_url=BASE, api_key=KEY, timeout=120.0, max_retries=1)
results = {}


def chat(messages, max_tokens=256, label=""):
    t0 = time.time()
    r = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=max_tokens, temperature=0.0
    )
    dt = time.time() - t0
    out = (r.choices[0].message.content or "").strip()
    tok = r.usage.completion_tokens
    print(f"  {dt:5.1f}s  {tok:4d} tok  ({tok/dt:.1f} tok/s)  finish={r.choices[0].finish_reason}")
    return out, dt, r.usage


# ---------------------------------------------------------------- 1. reachable
print("\n[1] server reachable + auth")
try:
    served = [m.id for m in client.models.list().data]
    print(f"  models served: {served}")
    if MODEL not in served:
        print(f"  ⚠️  '{MODEL}' not in served list — set qwen_model_id to one of the above")
    results["reachable"] = True
except oai.AuthenticationError:
    sys.exit("  ✗ 401 — vllm_api_key does not match the --api-key on the Colab side")
except oai.APIConnectionError as e:
    sys.exit(f"  ✗ cannot connect — tunnel down or URL stale?\n     {e}")
except Exception as e:
    sys.exit(f"  ✗ {type(e).__name__}: {e}")

# ---------------------------------------------------------------- 2. text
print("\n[2] plain text generation")
out, t_text, _ = chat([{"role": "user", "content": "Reply with exactly: OK"}], 16)
print(f"  -> {out!r}")
results["text"] = "OK" in out.upper()

# ---------------------------------------------------------------- 3. vision
print("\n[3] vision path (the one most likely to break over a tunnel)")
if len(sys.argv) > 1:
    img = Image.open(sys.argv[1]).convert("RGB")
    question = "Describe this CAD render in one sentence."
    print(f"  using real render: {sys.argv[1]}  size={img.size}")
else:
    img = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(img).polygon([(256, 60), (60, 440), (452, 440)], outline="black", width=6)
    question = "How many sides does the shape have? Reply with just the number."
    print("  using generated triangle (pass a step_image_N.png path to use a real render)")

buf = BytesIO(); img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
print(f"  payload: {len(b64)/1024:.0f} KB base64")

try:
    out, t_img, _ = chat([{"role": "user", "content": [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}], 64)
    print(f"  -> {out!r}")
    results["vision"] = (len(sys.argv) > 1) or ("3" in out or "three" in out.lower())
    if not results["vision"]:
        print("  ⚠️  model responded but got the shape wrong — check --limit-mm-per-prompt.image")
except oai.BadRequestError as e:
    print(f"  ✗ 400 — server rejected the image. Is it a VL model? {e}")
    results["vision"] = False
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {e}")
    results["vision"] = False

# ---------------------------------------------------------------- 4. history
print("\n[4] stateless multi-turn (resent history — what the planner relies on)")
hist = [
    {"role": "user", "content": "The extrusion depth is 17mm. Acknowledge."},
    {"role": "assistant", "content": "Acknowledged: 17mm."},
    {"role": "user", "content": "What was the depth? Reply with just the number."},
]
out, _, _ = chat(hist, 16)
print(f"  -> {out!r}")
results["history"] = "17" in out

# ---------------------------------------------------------------- 5. format
print("\n[5] plan / ```python format adherence")
out, t_code, usage = chat([{"role": "user", "content":
    "Briefly state a plan, then give the code in a ```python block.\n"
    "Task: print the number of objects in a FreeCAD document named `doc`."}], 400)
has_block = "```python" in out
print(f"  ```python block: {has_block}")
if not has_block:
    print(f"  ---\n{out[:400]}\n  ---")
results["format"] = has_block

# ---------------------------------------------------------------- summary
print("\n" + "=" * 60)
for k, v in results.items():
    print(f"  {'✓' if v else '✗'}  {k}")

if all(results.values()):
    per_step = t_code
    print(f"\n  ~{per_step:.1f}s per planner step (1 request, no concurrency)")
    print(f"  ~{per_step*10/60:.1f} min per sample at max_steps=10, sequential")
    print(f"  ~{per_step*10*100/3600:.1f} GPU-hours per 100 samples before batching gains")
    print("\n  Endpoint is good. Run the 3-sample eval next.")
else:
    print("\n  Fix the ✗ items before starting a run.")



# [1] server reachable + auth
#   ✗ InternalServerError: Error code: 502 - {'type': 'https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-502/', 'title': 'Error 502: Bad gateway', 'status': 502, 'detail': 'The origin web server returned an invalid or incomplete response to Cloudflare. This typically indicates the origin is overloaded or misconfigured.', 'instance': 'a2638edfe82d596f', 'error_code': 502, 'error_name': 'origin_bad_gateway', 'error_category': 'origin', 'ray_id': 'a2638edfe82d596f', 'timestamp': '2026-08-05T05:56:21Z', 'zone': 'taylor-whom-mar-looking.trycloudflare.com', 'cloudflare_error': True, 'retryable': True, 'retry_after': 60, 'owner_action_required': True, 'what_you_should_do': '**Wait and retry.** Back off for at least 60 seconds. If the error persists, the website operator should check their origin server health and configuration.', 'footer': 'This error was generated by Cloudflare on behalf of the website owner.'}
