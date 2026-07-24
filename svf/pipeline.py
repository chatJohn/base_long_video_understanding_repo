"""Command-line pipeline for the SVF-Panel keyframe selection method."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from .benchmarks import (
    BenchmarkRecord,
    benchmark_defaults,
    create_adapter,
    feature_filename,
)
from .core import SVFConfig, SVFPanelSelector, compute_min_boundary_distance


def parse_index_list(index_list: Optional[str]) -> Optional[List[int]]:
    if not index_list:
        return None
    return [int(token.strip()) for token in index_list.split(",") if token.strip()]


def select_by_indices(
    raw_items: List[Dict[str, Any]],
    start: int,
    end: int,
    index_list: Optional[List[int]],
) -> List[tuple[int, Dict[str, Any]]]:
    if index_list is not None:
        return [(idx, raw_items[idx]) for idx in index_list if 0 <= idx < len(raw_items)]
    final = len(raw_items) if end < 0 else min(end, len(raw_items))
    return [(idx, raw_items[idx]) for idx in range(max(0, start), final)]


def uniform_sample_from_video(video_path: Path, budget: int) -> List[int]:
    if budget <= 0:
        return []
    if video_path.exists():
        try:
            from decord import VideoReader, cpu

            reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
            if len(reader) > 0:
                return np.linspace(0, len(reader) - 1, budget, dtype=int).tolist()
        except Exception:
            pass
    return list(range(budget))


class FeatureStore:
    """Lazy cache for frame scores and visual features."""

    def __init__(self, features_dir: Path) -> None:
        self.features_dir = features_dir
        self._scores: Dict[str, Optional[Dict[str, Any]]] = {}
        self._features: Dict[str, Optional[np.ndarray]] = {}

    def load_scores(self, feature_id: str) -> Optional[Dict[str, Any]]:
        if feature_id not in self._scores:
            path = self.features_dir / feature_id / "similarity_scores.json"
            if not path.exists():
                self._scores[feature_id] = None
            else:
                with path.open("r", encoding="utf-8") as handle:
                    self._scores[feature_id] = json.load(handle)
        return self._scores[feature_id]

    def load_visual_features(
        self,
        feature_id: str,
        feature_model: str,
    ) -> Optional[np.ndarray]:
        key = f"{feature_id}:{feature_model}"
        if key not in self._features:
            path = self.features_dir / feature_id / feature_filename(feature_model)
            if not path.exists():
                self._features[key] = None
            else:
                with path.open("rb") as handle:
                    value = pickle.load(handle)
                self._features[key] = None if value is None else np.asarray(value)
        return self._features[key]


def _map_selected_frames(selected: Sequence[int], frame_indices: Sequence[int]) -> List[int]:
    """Map sampled positions back to source-video frame indices."""

    mapping = list(frame_indices)
    if not mapping:
        return [int(idx) for idx in selected]
    return [int(mapping[idx]) if 0 <= idx < len(mapping) else int(idx) for idx in selected]


def _fallback_item(
    adapter: Any,
    record: BenchmarkRecord,
    budget: int,
    reason: str,
    include_narrative: bool,
) -> Dict[str, Any]:
    keyframes = uniform_sample_from_video(record.video_path, budget)
    item = adapter.to_output_item(record, keyframes)
    if include_narrative:
        item["narrative"] = {
            "method": "uniform_fallback",
            "reason": reason,
            "panels": [
                {"position": pos, "frame_index": int(frame)}
                for pos, frame in enumerate(keyframes)
            ],
            "pages": [list(range(len(keyframes)))] if keyframes else [],
            "transitions": [],
            "emphasis": {},
        }
    return item


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Run SVF-Panel over one benchmark slice and write benchmark JSON."""

    adapter = create_adapter(args.benchmark)
    raw_items = adapter.load_raw(Path(args.questions_file))
    selected_raw = select_by_indices(
        raw_items,
        args.start_index,
        args.end_index,
        parse_index_list(args.index_list),
    )
    dataset_root = Path(args.dataset_root)
    records = [
        adapter.build_record(index=index, item=item, dataset_root=dataset_root)
        for index, item in selected_raw
    ]

    config = SVFConfig(
        smooth_window=args.smooth_window,
        speed_weight=args.speed_weight,
        curvature_weight=args.curvature_weight,
        relevance_gradient_weight=args.relevance_gradient_weight,
        query_gate_floor=args.query_gate_floor,
        peak_height_factor=args.peak_height_factor,
        peak_prominence_factor=args.peak_prominence_factor,
        min_distance_ratio=args.min_distance_ratio,
        min_distance_absolute=args.min_distance_absolute,
        segment_duration_weight=args.segment_duration_weight,
        segment_mean_relevance_weight=args.segment_mean_relevance_weight,
        segment_max_relevance_weight=args.segment_max_relevance_weight,
        segment_motion_weight=args.segment_motion_weight,
        frame_relevance_weight=args.frame_relevance_weight,
        frame_boundary_weight=args.frame_boundary_weight,
        frame_speed_weight=args.frame_speed_weight,
        frame_curvature_weight=args.frame_curvature_weight,
        mmr_lambda=args.mmr_lambda,
        gutter_low_factor=args.gutter_low_factor,
        gutter_high_factor=args.gutter_high_factor,
        page_gutter_factor=args.page_gutter_factor,
        refinement_steps=args.refinement_steps,
        emphasis_fraction=args.emphasis_fraction,
    )
    selector = SVFPanelSelector(config)
    store = FeatureStore(Path(args.features_dir))

    results: List[Dict[str, Any]] = []
    success_count = 0
    query_proxy_count = 0
    fallback_count = 0
    missing_scores = 0

    for record in tqdm(records, desc=f"SVF-Panel-{adapter.name}"):
        score_payload = store.load_scores(record.feature_id)
        if score_payload is None:
            missing_scores += 1
            fallback_count += 1
            results.append(
                _fallback_item(
                    adapter,
                    record,
                    args.max_frames,
                    "missing_scores",
                    not args.no_narrative,
                )
            )
            continue

        relevance_scores = adapter.get_scores(score_payload, record, args.feature_model)
        if relevance_scores is None or len(relevance_scores) < args.max_frames:
            fallback_count += 1
            results.append(
                _fallback_item(
                    adapter,
                    record,
                    args.max_frames,
                    "insufficient_scores",
                    not args.no_narrative,
                )
            )
            continue

        frame_indices = score_payload.get("frame_indices", [])
        if not isinstance(frame_indices, list) or len(frame_indices) < len(relevance_scores):
            frame_indices = list(range(len(relevance_scores)))

        features = None
        if not args.no_visual_features:
            features = store.load_visual_features(record.feature_id, args.feature_model)

        min_distance = compute_min_boundary_distance(
            len(relevance_scores),
            ratio=args.min_distance_ratio,
            absolute_min=args.min_distance_absolute,
        )
        selection = selector.select_with_narrative(
            relevance_scores=np.asarray(relevance_scores, dtype=float),
            num_frames=args.max_frames,
            features=features,
            min_boundary_distance=min_distance,
        )
        keyframes = _map_selected_frames(selection.selected_indices, frame_indices)
        item = adapter.to_output_item(record, keyframes)
        if not args.no_narrative:
            item["narrative"] = selection.narrative.to_dict(frame_indices)
        results.append(item)
        success_count += 1
        if selection.fields.source == "query_proxy":
            query_proxy_count += 1

    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path("outputs")
        / adapter.name
        / f"SVF_Panel_{adapter.name}_{args.feature_model}_{args.max_frames}f.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    return {
        "benchmark": adapter.name,
        "num_records": len(records),
        "success": success_count,
        "query_proxy": query_proxy_count,
        "fallback": fallback_count,
        "missing_scores": missing_scores,
        "output_path": str(output_path),
        "svf_config": asdict(config),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run training-free Semantic Velocity Field + Panel selection"
    )
    parser.add_argument("--benchmark", required=True, choices=["videomme", "lvb", "mlvu"])
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--questions_file", default=None)
    parser.add_argument("--features_dir", default=None)
    parser.add_argument(
        "--feature_model",
        default="blip2",
        choices=["blip2", "blip1", "clip", "siglip"],
    )
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--index_list", default=None)
    parser.add_argument("--no_visual_features", action="store_true")
    parser.add_argument("--no_narrative", action="store_true")

    parser.add_argument("--smooth_window", type=int, default=3)
    parser.add_argument("--speed_weight", type=float, default=0.45)
    parser.add_argument("--curvature_weight", type=float, default=0.30)
    parser.add_argument("--relevance_gradient_weight", type=float, default=0.25)
    parser.add_argument("--query_gate_floor", type=float, default=0.20)
    parser.add_argument("--peak_height_factor", type=float, default=0.50)
    parser.add_argument("--peak_prominence_factor", type=float, default=0.08)
    parser.add_argument("--min_distance_ratio", type=float, default=0.02)
    parser.add_argument("--min_distance_absolute", type=int, default=3)

    parser.add_argument("--segment_duration_weight", type=float, default=0.15)
    parser.add_argument("--segment_mean_relevance_weight", type=float, default=0.35)
    parser.add_argument("--segment_max_relevance_weight", type=float, default=0.25)
    parser.add_argument("--segment_motion_weight", type=float, default=0.25)
    parser.add_argument("--frame_relevance_weight", type=float, default=0.55)
    parser.add_argument("--frame_boundary_weight", type=float, default=0.25)
    parser.add_argument("--frame_speed_weight", type=float, default=0.10)
    parser.add_argument("--frame_curvature_weight", type=float, default=0.10)
    parser.add_argument("--mmr_lambda", type=float, default=0.65)

    parser.add_argument("--gutter_low_factor", type=float, default=0.35)
    parser.add_argument("--gutter_high_factor", type=float, default=1.75)
    parser.add_argument("--page_gutter_factor", type=float, default=1.50)
    parser.add_argument("--refinement_steps", type=int, default=4)
    parser.add_argument("--emphasis_fraction", type=float, default=0.20)
    return parser


def resolve_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    defaults = benchmark_defaults(args.benchmark)
    if args.dataset_root is None:
        args.dataset_root = str(defaults["dataset_root"])
    if args.questions_file is None:
        args.questions_file = str(defaults["questions_file"])
    if args.features_dir is None:
        args.features_dir = str(defaults["features_dir"])
    if args.max_frames is None:
        args.max_frames = int(defaults["max_frames"])
    return args


def main() -> None:
    args = resolve_default_paths(build_parser().parse_args())
    summary = run_pipeline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
