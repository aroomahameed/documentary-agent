"""
Dev tool: render a manifest you already have into an MP4. No LLM, no API calls.

    python render_video.py output/<folder>/media-<time>/manifest.json
    python render_video.py output/<folder>/media/manifest.json --height 480   # quick preview

The video is saved next to the manifest as documentary.mp4.
"""

import argparse
import time
from pathlib import Path

from doc_agent.video import render_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="path to manifest.json")
    parser.add_argument("--height", type=int, default=1080,
                        help="1080 for the final video, 480 for a fast preview")
    args = parser.parse_args()

    start = time.time()
    out = render_video(args.manifest, height=args.height)
    print(f"\n✅ Rendered {out} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()