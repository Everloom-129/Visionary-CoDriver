"""VCD Co-driver Agent — scene assembler + LLM client.

Pipeline:
  Raw fusion JSON records
       │
       ▼
  analyze_scene()   — GOFAI trajectory analysis → BehaviorDescriptor per track
  (risk_analyzer)     crossing / approaching / parallel / stationary / receding
       │
       ▼
  SceneAssembler    — converts behavior descriptors to natural-language scene text
       │
       ▼
  LLMClient         — sends assembled text to GPT-3.5 for co-driver reasoning
       │
       ▼
  Co-driver narrative + per-person risk

Usage:
    from VCD.agent.llm import run_agent
    response = run_agent("fusion_results/video_0001.json")
    print(response)

CLI:
    python -m VCD.agent fusion_results/video_0001.json --stride 30
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from VCD.agent.risk_analyzer import (
    analyze_scene,
    BehaviorDescriptor,
    most_critical,
    _MOTION_SEVERITY,
)

_REGION_LABEL = {
    0: "road", 1: "sidewalk", 2: "crosswalk",
    3: "parking area", 4: "grass", 5: "unknown surface",
}


# ---------------------------------------------------------------------------
# Scene Assembler — deterministic text generation from behavior descriptors
# ---------------------------------------------------------------------------

class SceneAssembler:
    """Converts fused detection records + trajectory behaviors into scene text for the LLM.

    The output is structured natural language that describes *what each pedestrian
    is doing*, not raw sensor numbers. The LLM then reasons about risk from behavior.
    """

    def frame_to_text(
        self,
        frame_id: int,
        recs: List[dict],
        behaviors: Dict[int, BehaviorDescriptor],
        fps: float = 30.0,
    ) -> str:
        ts = frame_id / fps
        lines = [f"Frame {frame_id} (t={ts:.1f}s):"]

        # Unique surface types visible in this frame
        surfaces = sorted({r.get("region_category") for r in recs
                           if r.get("region_category") is not None})
        if surfaces:
            lines.append(
                "  Surfaces: " + ", ".join(_REGION_LABEL.get(s, "unknown") for s in surfaces)
            )

        tracked   = sorted([r for r in recs if r.get("track_id") is not None],
                           key=lambda r: r["track_id"])
        untracked = [r for r in recs if r.get("track_id") is None]

        if not recs:
            lines.append("  No persons detected.")
            return "\n".join(lines)

        lines.append(f"  Persons ({len(recs)}, {len(untracked)} untracked):")

        for rec in tracked:
            tid = rec["track_id"]
            b   = behaviors.get(tid)
            if b:
                lines.append(f"    Person track#{tid}: {b.situation}")
            else:
                # Track just appeared — no history window yet
                depth  = rec.get("avg_depth")
                region = _REGION_LABEL.get(rec.get("region_category"), "unknown surface")
                d_str  = f"{depth:.1f}m away" if depth is not None else "unknown distance"
                lines.append(f"    Person track#{tid}: Just appeared, {d_str}, on {region}.")

        for rec in untracked:
            depth  = rec.get("avg_depth")
            region = _REGION_LABEL.get(rec.get("region_category"), "unknown surface")
            angle  = rec.get("angle")
            d_str  = f"{depth:.1f}m away" if depth is not None else "unknown distance"
            a_str  = f" at {angle:+.0f}°" if angle is not None else ""
            lines.append(f"    Person (untracked): {d_str}, on {region}{a_str}.")

        # Frame-level alert for the most critical behavior
        critical = most_critical({k: v for k, v in behaviors.items()
                                  if v.motion_type in ("crossing", "approaching")})
        if critical:
            lines.append(
                f"  !! Alert: Person track#{critical.track_id} is "
                f"{critical.motion_type.upper()} — requires immediate attention."
            )

        return "\n".join(lines)

    def video_to_text(
        self,
        video_name: str,
        fused_by_frame: Dict[int, List[dict]],
        stride: int = 30,
        fps: float = 30.0,
    ) -> str:
        """Build a full scene report, sampling one keyframe every `stride` frames.

        stride=30 → 1 keyframe/second  (recommended for GPT-3.5 context budget)
        stride=15 → every slow-vision keyframe (more detail, longer prompt)
        """
        header = (
            f"Scene report: {video_name}\n"
            "Dashcam co-driver system. Track IDs are globally unique across frames.\n"
            "Pedestrian behavior is inferred from trajectory over a ~1-second window:\n"
            "  'crossing'    — significant lateral motion across the road\n"
            "  'approaching' — bbox growing, closing distance to vehicle\n"
            "  'parallel'    — moving along the road, not heading toward vehicle\n"
            "  'stationary'  — minimal motion detected\n"
            "  'receding'    — moving away from the vehicle\n"
        )

        blocks = [header]
        person_worst: Dict[int, str] = {}  # track_id → worst motion type seen

        for frame_id in sorted(fused_by_frame):
            if frame_id % stride != 0:
                continue
            recs      = fused_by_frame[frame_id]
            behaviors = analyze_scene(fused_by_frame, frame_id)
            blocks.append(self.frame_to_text(frame_id, recs, behaviors, fps=fps))

            for tid, b in behaviors.items():
                prev_sev = _MOTION_SEVERITY.get(person_worst.get(tid, "receding"), 0)
                if _MOTION_SEVERITY.get(b.motion_type, 0) > prev_sev:
                    person_worst[tid] = b.motion_type

        # Cross-frame behavior summary
        if person_worst:
            summary = ["Overall behavior summary (worst across sampled frames):"]
            for tid, mtype in sorted(person_worst.items()):
                summary.append(f"  Person track#{tid}: {mtype}")
            blocks.append("\n".join(summary))

        return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a co-driver assistant for a dashcam-equipped vehicle.
You receive structured scene descriptions where each pedestrian's behavior has been
inferred from trajectory analysis: crossing / approaching / parallel / stationary / receding.

Risk interpretation guidelines:
  - 'crossing' on road → HIGH risk (may enter vehicle path)
  - 'approaching' toward vehicle → MEDIUM-HIGH risk
  - 'parallel' on road → MEDIUM risk (could step into path)
  - 'stationary' on road → MEDIUM risk (obstruction)
  - 'stationary' on sidewalk → LOW risk
  - 'receding' → LOW risk

Think step by step:
1. Summarise the overall traffic situation in 2-3 sentences.
2. Assess risk for each tracked person and explain the reasoning.
3. Recommend a specific driver action.

Required output format — one line per person, then action:
  Person track#<id>: <low|medium|high> — <reason based on their behavior>
  ...
  Recommended action: <specific action>
"""

_USER_TEMPLATE = """\
Video: {video_name}

{scene_text}

Provide your co-driver analysis now.
"""


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin wrapper around the OpenAI chat completions API."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
    ):
        self.model       = model
        self.temperature = temperature
        self.api_key     = api_key or os.environ.get("OPENAI_API_KEY", "")

    def query(self, scene_text: str, video_name: str = "video") -> str:
        try:
            import openai
        except ImportError as e:
            raise ImportError("Install the openai package: pip install openai") from e

        if not self.api_key:
            raise ValueError(
                "No OpenAI API key. Set the OPENAI_API_KEY environment variable."
            )

        client   = openai.OpenAI(api_key=self.api_key)
        user_msg = _USER_TEMPLATE.format(video_name=video_name, scene_text=scene_text)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def load_fusion_json(path: str) -> Dict[int, List[dict]]:
    """Load a fusion JSON and group raw records by frame_id."""
    with open(path) as f:
        records = json.load(f)
    fused_by_frame: Dict[int, List[dict]] = {}
    for rec in records:
        fused_by_frame.setdefault(rec["frame_id"], []).append(rec)
    return fused_by_frame


def assemble_scene(
    fusion_json_path: str,
    video_name: Optional[str] = None,
    stride: int = 30,
    fps: float = 30.0,
) -> str:
    """Assemble the behavior-based scene text without calling the LLM."""
    if video_name is None:
        video_name = os.path.splitext(os.path.basename(fusion_json_path))[0]
    fused_by_frame = load_fusion_json(fusion_json_path)
    return SceneAssembler().video_to_text(video_name, fused_by_frame, stride=stride, fps=fps)


def run_agent(
    fusion_json_path: str,
    video_name: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    stride: int = 30,
    fps: float = 30.0,
    api_key: Optional[str] = None,
    print_scene: bool = False,
) -> str:
    """Full pipeline: load fusion JSON → analyze behaviors → assemble scene → query LLM.

    Args:
        fusion_json_path: Path to fusion JSON produced by run_fusion().
        video_name:       Optional label override for prompts.
        model:            OpenAI model ID (default: gpt-3.5-turbo).
        stride:           Sample one keyframe per `stride` frames (default 30 = 1 Hz).
        fps:              Source video frame rate (default 30).
        api_key:          OpenAI API key (falls back to OPENAI_API_KEY env var).
        print_scene:      If True, print the assembled scene text before the LLM call.

    Returns:
        LLM co-driver response string.
    """
    if video_name is None:
        video_name = os.path.splitext(os.path.basename(fusion_json_path))[0]

    scene_text = assemble_scene(fusion_json_path, video_name, stride=stride, fps=fps)

    if print_scene:
        print("=" * 60)
        print("ASSEMBLED SCENE TEXT (sent to LLM):")
        print("=" * 60)
        print(scene_text)
        print("=" * 60)

    return LLMClient(model=model, api_key=api_key).query(scene_text, video_name=video_name)
