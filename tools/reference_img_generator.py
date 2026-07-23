import os
import re
import base64
import mimetypes

import requests
from dotenv import load_dotenv

# Load the Api.Airforce API key from local .env
load_dotenv()
API_AIRFORCE_KEY = os.getenv("API_AIRFORCE_KEY")
if not API_AIRFORCE_KEY:
    raise RuntimeError("Missing API_AIRFORCE_KEY. Add API_AIRFORCE_KEY=... to your .env file.")

# Api.Airforce is OpenAI-compatible
AIRFORCE_BASE_URL = "https://api.airforce/v1"
MULTI_IMAGE_MODEL = "grok-imagine-image"  # used for 2-4 references (input_images array)
SINGLE_EDIT_MODEL = "image-1-edit"        # used for exactly 1 reference (init_image)
MAX_REFERENCE_IMAGES = 4                   # cap on reference images we send

# NOTE: Per the Api.Airforce media docs, reference images go in `input_images`
# as an array of {"url": ...} objects (base64 must be a data: URI inside url).
# This is the correct, documented format. However, on the current account the
# service accepts the field (HTTP 200) but does NOT apply the references — every
# model tested (grok-imagine-image, image-1-edit, nano-banana-pro/2) ignored
# both base64 and public-URL references. Treat img2img as unverified here until
# the account/plan is confirmed to support it.

HEADERS = {
    "Authorization": f"Bearer {API_AIRFORCE_KEY}",
    "Content-Type": "application/json",
}

# Save generated images under <app_root>/generated_images regardless of cwd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "generated_images")


def _raise_airforce_error(resp, stage):
    """Raise a readable error that includes Api.Airforce's own message body, so
    the real reason (bad key, plan limits, unsupported model, etc.) surfaces in
    the UI instead of a bare HTTP status code."""
    detail = None
    try:
        body = resp.json()
        detail = body.get("error", body)
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("error") or str(detail)
    except Exception:
        detail = (resp.text or "").strip()[:500]

    detail = detail or ""
    message = f"Api.Airforce error during {stage} (HTTP {resp.status_code}): {detail}"

    # Add a clear, actionable note when the account is out of credits so the
    # agent tells the user to top up instead of just relaying a raw quota string.
    low = detail.lower()
    if resp.status_code in (402, 403) or "quota" in low or "credit" in low or "allowance" in low:
        message += (
            " — Your Api.Airforce credit allowance is exhausted. Top up or upgrade "
            "your plan at https://api.airforce to keep generating images."
        )
    raise RuntimeError(message)


def slugify_subject(prompt, max_words=4):
    """Turn a short prompt into an underscore filename subject, e.g. 'the cat' -> 'the_cat'."""
    words = re.findall(r"[A-Za-z0-9]+", prompt.lower())
    if not words:
        return "reference_image"
    slug = "_".join(words[:max_words])[:60]
    return slug or "reference_image"


def _to_data_uri(reference):
    """Normalise a reference image to a string the API accepts: a public URL
    passed through unchanged, or a local file read and base64-encoded into a
    data: URI."""
    if not reference:
        return None
    reference = reference.strip()
    if reference.startswith(("http://", "https://", "data:")):
        return reference
    if not os.path.isfile(reference):
        raise RuntimeError(f"Reference image not found: {reference}")
    mime, _ = mimetypes.guess_type(reference)
    mime = mime or "image/png"
    with open(reference, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def generate_from_references(
    user_prompt,
    reference_image_1=None,
    reference_image_2=None,
    reference_image_3=None,
    reference_image_4=None,
):
    """A tool that generates an image from a text prompt guided by up to 4
    reference images, using Api.Airforce's image-to-image model.

    Args:
        user_prompt: A description of the image to generate.
        reference_image_1: First reference image (local file path or public URL).
        reference_image_2: Second reference image (local file path or public URL).
        reference_image_3: Third reference image (local file path or public URL).
        reference_image_4: Fourth reference image (local file path or public URL).
    """
    # Collect and encode any provided reference images (up to 4)
    references = [
        reference_image_1,
        reference_image_2,
        reference_image_3,
        reference_image_4,
    ]
    uris = [_to_data_uri(ref) for ref in references if ref]
    uris = uris[:MAX_REFERENCE_IMAGES]

    payload = {
        "prompt": user_prompt,
        "n": 1,
        "response_format": "url",
    }
    if len(uris) == 1:
        # Single reference: use the dedicated edit model with `init_image`
        payload["model"] = SINGLE_EDIT_MODEL
        payload["init_image"] = uris[0]
    elif len(uris) > 1:
        # Multiple references: multi-image model with the documented
        # `input_images` array of {"url": ...} objects
        payload["model"] = MULTI_IMAGE_MODEL
        payload["input_images"] = [{"url": u} for u in uris]
    else:
        # No references: plain text-to-image
        payload["model"] = MULTI_IMAGE_MODEL

    print(f"\nPrompt: '{user_prompt}' with {len(uris)} reference image(s)")
    print(f"Generating image via Api.Airforce (model={payload['model']})...")

    resp = requests.post(
        f"{AIRFORCE_BASE_URL}/images/generations",
        headers=HEADERS,
        json=payload,
        timeout=180,
    )
    if not resp.ok:
        _raise_airforce_error(resp, "image generation")

    try:
        item = resp.json()["data"][0]
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(f"Unexpected Api.Airforce response: {resp.text[:500]}") from exc

    # Response may carry a URL or an inline base64 image
    os.makedirs(IMAGES_DIR, exist_ok=True)
    filename = os.path.join(IMAGES_DIR, f"{slugify_subject(user_prompt)}_ref.png")

    if item.get("url"):
        dl_resp = requests.get(item["url"], timeout=60)
        if not dl_resp.ok:
            raise RuntimeError(f"Failed to download generated image (HTTP {dl_resp.status_code}).")
        img_bytes = dl_resp.content
    elif item.get("b64_json"):
        img_bytes = base64.b64decode(item["b64_json"])
    else:
        raise RuntimeError(f"No image URL or data in Api.Airforce response: {str(item)[:500]}")

    with open(filename, "wb") as f:
        f.write(img_bytes)
    print(f"Success! Image saved as {filename}")
    return filename


# Run the tool!
if __name__ == "__main__":
    generate_from_references(
        "A watercolor portrait in the style of the reference",
        reference_image_1="https://picsum.photos/512",
    )
