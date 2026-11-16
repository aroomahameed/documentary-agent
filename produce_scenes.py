"""
Dev tool: run ONLY the scene subgraph on a script you already have.
No research, no fact-checking, so it's cheap and fast to iterate on.

    python produce_scenes.py output/<folder>/script.json

Media goes to output/<folder>/media/, plus a manifest.json describing every scene.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from doc_agent.production import build_scene_graph, choose_voice  # noqa: E402
from doc_agent.state import Script  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python produce_scenes.py output/<folder>/script.json")

    script_path = Path(sys.argv[1])
    script = Script.model_validate_json(script_path.read_text(encoding="utf-8"))
    media_dir = script_path.parent / "media"
    print(f"🎬 {script.title}: producing {len(script.scenes)} scenes into {media_dir}")

    # Choose the narrator ONCE, from the title and the opening narration
    voice, reason = choose_voice(f"{script.title}. {script.scenes[0].narration}")
    print(f"🎙  Narrator: {voice} ({reason})")

    scene_graph = build_scene_graph()

    # .batch() runs the subgraph once per scene, in parallel threads.
    # return_exceptions=True: one failing scene must not crash the others.
    results = scene_graph.batch(
        [{"scene": s, "media_dir": str(media_dir), "voice": voice} for s in script.scenes],
        config={"max_concurrency": 4},
        return_exceptions=True,
    )

    produced = []
    for scene, result in zip(script.scenes, results):
        if isinstance(result, Exception):
            print(f"  scene {scene.scene_number}: ❌ FAILED ({type(result).__name__}: {str(result)[:150]})")
        else:
            produced.append(result["produced_scenes"][0])

    for p in produced:
        kinds = ", ".join(s.kind for s in p.shots)
        print(f"  scene {p.scene_number}: {p.duration_sec:.1f}s, {len(p.shots)} shot(s): {kinds}")
        for problem in p.problems:
            print(f"     ⚠ {problem}")

    total = sum(p.duration_sec for p in produced)
    manifest = media_dir / "manifest.json"
    manifest.write_text(
        json.dumps([p.model_dump() for p in produced], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    failed = len(script.scenes) - len(produced)
    print(f"\n✅ {len(produced)} scene(s) produced, {failed} failed. "
          f"Total narration: {total:.0f}s. Manifest: {manifest}")


if __name__ == "__main__":
    main()