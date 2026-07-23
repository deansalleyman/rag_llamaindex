import os
import re
import requests
from dotenv import load_dotenv

# Load GROK API key from local .env
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")
if not GROK_API_KEY:
    raise RuntimeError("Missing GROK_API_KEY API key. Add GROK=xai-... to your .env file.")

# xAI (Grok) is OpenAI-compatible
XAI_BASE_URL = "https://api.x.ai/v1"
CHAT_MODEL = "grok-4.5"               # used to expand the prompt
IMAGE_MODEL = "grok-imagine-image"    # xAI image generation model

HEADERS = {
    "Authorization": f"Bearer {GROK_API_KEY}",
    "Content-Type": "application/json",
}

# Save generated images under <app_root>/generated_images regardless of cwd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "generated_images")


def _raise_grok_error(resp, stage):
    """Raise a readable error that includes Grok's own message body, so the
    real reason (rate limit, content policy, bad model, etc.) surfaces in the UI
    instead of a bare HTTP status code."""
    detail = None
    try:
        body = resp.json()
        detail = body.get("error", body)
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("error") or str(detail)
    except Exception:
        detail = (resp.text or "").strip()[:500]
    raise RuntimeError(f"Grok API error during {stage} (HTTP {resp.status_code}): {detail}")


def slugify_subject(prompt, max_words=4):
    """Turn a short prompt into an underscore filename subject, e.g. 'the cat' -> 'the_cat'."""
    words = re.findall(r"[A-Za-z0-9]+", prompt.lower())
    if not words:
        return "generated_image"
    slug = "_".join(words[:max_words])[:60]
    return slug or "generated_image"


def generate_the_image(user_prompt):
    # 1. Ask Grok to expand the short idea into a detailed prompt
    print(f"\nOriginal Prompt: '{user_prompt}'")
    print("Asking Grok to make it better...")

    chat_resp = requests.post(
        f"{XAI_BASE_URL}/chat/completions",
        headers=HEADERS,
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Expand this short description into a highly detailed image "
                        f"prompt for an AI image generator: {user_prompt}"
                    ),
                }
            ],
            "temperature": 0.7,
        },
        timeout=60,
    )
    if not chat_resp.ok:
        _raise_grok_error(chat_resp, "prompt expansion")
    try:
        expanded_prompt = chat_resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(f"Unexpected Grok chat response: {chat_resp.text[:500]}") from exc
    print(f"Expanded Prompt:\n{expanded_prompt}\n")

    # 2. Generate the image with Grok's image model
    print("Generating image via Grok...")
    image_resp = requests.post(
        f"{XAI_BASE_URL}/images/generations",
        headers=HEADERS,
        json={
            "model": IMAGE_MODEL,
            "prompt": expanded_prompt,
            "n": 1,
            "response_format": "url",
        },
        timeout=120,
    )
    if not image_resp.ok:
        _raise_grok_error(image_resp, "image generation")
    try:
        image_url = image_resp.json()["data"][0]["url"]
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(f"Unexpected Grok image response: {image_resp.text[:500]}") from exc

    # 3. Download and save with a subject-based filename (e.g. the_cat.png)
    dl_resp = requests.get(image_url, timeout=60)
    if not dl_resp.ok:
        raise RuntimeError(f"Failed to download generated image (HTTP {dl_resp.status_code}).")
    img_bytes = dl_resp.content
    os.makedirs(IMAGES_DIR, exist_ok=True)
    filename = os.path.join(IMAGES_DIR, f"{slugify_subject(user_prompt)}.png")
    with open(filename, "wb") as f:
        f.write(img_bytes)
    print(f"Success! Image saved as {filename}")
    return filename


# Run the tool!
if __name__ == "__main__":
    short_idea = "A cyberpunk raccoon eating a hot dog in the rain"
    generate_the_image(short_idea)
