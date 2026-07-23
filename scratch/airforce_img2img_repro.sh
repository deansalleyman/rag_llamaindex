#!/usr/bin/env bash
# Minimal reproduction: Api.Airforce ignores input_images (image-to-image).
# Expected if working: the output image is the Mona Lisa with a green sky + snow.
# Actual on this account: a snowy forest with NO trace of the Mona Lisa.
#
# Usage:  API_AIRFORCE_KEY=xxxx ./airforce_img2img_repro.sh
set -euo pipefail
: "${API_AIRFORCE_KEY:?set API_AIRFORCE_KEY}"

curl -s https://api.airforce/v1/images/generations \
  -H "Authorization: Bearer $API_AIRFORCE_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "image-1-edit",
    "prompt": "make the sky bright green and add falling snow",
    "input_images": [
      {"url": "https://upload.wikimedia.org/wikipedia/commons/6/6a/Mona_Lisa.jpg"}
    ]
  }'
echo
# Open the returned data[0].url to see that the reference was ignored.
