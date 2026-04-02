"""CLI entry point: python -m VCD.agent <fusion_json> [options]

Examples:
    # Print assembled scene text only (no LLM call)
    python -m VCD.agent fusion_results/video_0001.json --scene-only

    # Run full agent with GPT-3.5
    python -m VCD.agent fusion_results/video_0001.json

    # Finer temporal resolution (every 15 frames = every slow keyframe)
    python -m VCD.agent fusion_results/video_0001.json --stride 15

    # Run on all three test videos
    for f in fusion_results/video_000{1,2,3}.json; do
        python -m VCD.agent "$f" --print-scene
    done
"""

import argparse
import sys

from VCD.agent.llm import assemble_scene, run_agent


def main():
    parser = argparse.ArgumentParser(
        prog="python -m VCD.agent",
        description="VCD co-driver agent: GOFAI scene assembly + GPT-3.5 analysis",
    )
    parser.add_argument("fusion_json", help="Path to fusion results JSON")
    parser.add_argument("--video-name", default=None, help="Override video label")
    parser.add_argument("--model",  default="gpt-3.5-turbo", help="OpenAI model ID")
    parser.add_argument("--stride", type=int, default=30,
                        help="Sample every N frames (default 30 = 1 keyframe/sec)")
    parser.add_argument("--fps",    type=float, default=30.0, help="Source video FPS")
    parser.add_argument("--scene-only", action="store_true",
                        help="Print assembled scene text and exit (no LLM call)")
    parser.add_argument("--print-scene", action="store_true",
                        help="Print assembled scene before LLM call")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    args = parser.parse_args()

    if args.scene_only:
        text = assemble_scene(
            args.fusion_json,
            video_name=args.video_name,
            stride=args.stride,
            fps=args.fps,
        )
        print(text)
        return

    response = run_agent(
        args.fusion_json,
        video_name=args.video_name,
        model=args.model,
        stride=args.stride,
        fps=args.fps,
        api_key=args.api_key,
        print_scene=args.print_scene,
    )
    print("\n" + "=" * 60)
    print("CO-DRIVER RESPONSE:")
    print("=" * 60)
    print(response)


if __name__ == "__main__":
    main()
