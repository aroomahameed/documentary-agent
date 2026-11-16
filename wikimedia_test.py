"""
Can we get Wikimedia images through Openverse's own image server?

    python wikimedia_test.py

Searches Openverse for Wikimedia artwork, then downloads each result:
  1) from Wikimedia directly (expected to fail on networks that block it)
  2) through Openverse's thumbnail endpoint
  3) through the same endpoint with ?full_size=true (undocumented, may be ignored)
and prints the image size each way returned.
"""

import io

import requests
from PIL import Image

HEADERS = {"User-Agent": "documentary-agent/0.1 (learning project)"}


def size_of(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        w, h = Image.open(io.BytesIO(resp.content)).size
        return f"{w}x{h}"
    except Exception as e:  # noqa: BLE001
        return f"FAILED ({type(e).__name__})"


resp = requests.get(
    "https://api.openverse.org/v1/images/",
    headers=HEADERS, timeout=20,
    params={"q": "Mughal miniature", "source": "wikimedia", "page_size": 3},
)
resp.raise_for_status()
results = resp.json().get("results", [])
print(f"Found {len(results)} Wikimedia results\n")

for r in results:
    print(r.get("title", "")[:70])
    print("  direct from Wikimedia: ", size_of(r["url"]))
    print("  Openverse thumbnail:   ", size_of(r["thumbnail"]))
    print("  thumbnail full_size:   ", size_of(r["thumbnail"] + "?full_size=true"))
    print()