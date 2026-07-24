"""Training-free semantic-velocity keyframe selection.

The method treats normalized frame embeddings as a trajectory in semantic
space. First-order displacement gives semantic speed, changes in displacement
direction give curvature, and query relevance gates both signals. The selected
frames are then refined with a comic-inspired closure objective: overly wide
semantic gaps receive bridge frames while redundant close panels give up their
budget.

In this module, a *panel* is one selected video frame, a *page* is a temporal
group of panels, and a *collage* is an optional downstream raster composition.
This module builds panel/page metadata but does not modify image pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks


EPS = 1e-8


def _robust_unit(values: np.ndarray, flat_value: float = 0.0) -> np.ndarray:
    """Robustly scale a one-dimensional signal to ``[0, 1]``."""

    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(arr, [5.0, 95.0])
    if high - low <= EPS:
        return np.full_like(arr, flat_value, dtype=float)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    """Apply an edge-preserving centered moving average."""

    arr = np.asarray(values, dtype=float)
    window = max(1, int(window))
    if window == 1 or len(arr) < 2:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(arr, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _smooth_features(features: np.ndarray, window: int) -> np.ndarray:
    """Smooth a frame-by-feature matrix along its temporal axis."""

    if window <= 1 or len(features) < 2:
        return features.copy()
    columns = [_smooth_1d(features[:, idx], window) for idx in range(features.shape[1])]
    return np.stack(columns, axis=1)


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, EPS)


def _prepare_features(features: np.ndarray, num_frames: int) -> np.ndarray:
    """Convert frame-aligned extractor output into an ``[N, D]`` matrix."""

    arr = np.asarray(features, dtype=float)
    if arr.ndim == 0 or arr.shape[0] != num_frames:
        raise ValueError("visual features must be frame-aligned with relevance scores")
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        # Pool token/spatial dimensions while preserving the embedding axis.
        arr = arr.reshape(num_frames, -1, arr.shape[-1]).mean(axis=1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


@dataclass
class SemanticFieldResult:
    """Frame-aligned signals derived from the semantic trajectory."""

    speed: np.ndarray
    curvature: np.ndarray
    relevance: np.ndarray
    relevance_gradient: np.ndarray
    query_gate: np.ndarray
    boundary_energy: np.ndarray
    normalized_features: np.ndarray
    source: str


@dataclass
class SVFConfig:
    """Hyperparameters for semantic fields, selection, and narrative closure."""

    smooth_window: int = 3
    speed_weight: float = 0.45
    curvature_weight: float = 0.30
    relevance_gradient_weight: float = 0.25
    query_gate_floor: float = 0.20
    peak_height_factor: float = 0.50
    peak_prominence_factor: float = 0.08
    min_distance_ratio: float = 0.02
    min_distance_absolute: int = 3

    segment_duration_weight: float = 0.15
    segment_mean_relevance_weight: float = 0.35
    segment_max_relevance_weight: float = 0.25
    segment_motion_weight: float = 0.25

    frame_relevance_weight: float = 0.55
    frame_boundary_weight: float = 0.25
    frame_speed_weight: float = 0.10
    frame_curvature_weight: float = 0.10
    mmr_lambda: float = 0.65

    gutter_low_factor: float = 0.35
    gutter_high_factor: float = 1.75
    page_gutter_factor: float = 1.50
    refinement_steps: int = 4
    emphasis_fraction: float = 0.20


class SemanticVelocityField:
    """Build speed, curvature, and query-conditioned boundary fields."""

    def __init__(self, config: Optional[SVFConfig] = None) -> None:
        self.config = config or SVFConfig()

    def compute(
        self,
        relevance_scores: np.ndarray,
        features: Optional[np.ndarray],
    ) -> SemanticFieldResult:
        """Compute all frame-aligned fields.

        Missing or malformed visual features trigger a documented query-only
        proxy. This preserves pipeline availability but should be reported
        separately in experiments because it is not semantic velocity.
        """

        raw_relevance = np.asarray(relevance_scores, dtype=float).reshape(-1)
        raw_relevance = np.nan_to_num(raw_relevance, nan=0.0, posinf=0.0, neginf=0.0)
        num_frames = len(raw_relevance)
        relevance = _robust_unit(raw_relevance, flat_value=1.0)

        source = "visual_features"
        try:
            if features is None:
                raise ValueError("missing visual features")
            trajectory = _prepare_features(features, num_frames)
        except (TypeError, ValueError):
            source = "query_proxy"
            # Time prevents identical points from producing undefined directions.
            time_axis = np.linspace(0.0, 0.05, num_frames, dtype=float)
            trajectory = np.stack([relevance, time_axis], axis=1)

        trajectory = _normalize_rows(trajectory)
        trajectory = _normalize_rows(_smooth_features(trajectory, self.config.smooth_window))

        velocity = np.zeros_like(trajectory)
        if num_frames > 1:
            velocity[1:] = trajectory[1:] - trajectory[:-1]
        speed_raw = np.linalg.norm(velocity, axis=1)

        curvature_raw = np.zeros(num_frames, dtype=float)
        for idx in range(2, num_frames):
            left_norm = speed_raw[idx - 1]
            right_norm = speed_raw[idx]
            if left_norm <= EPS or right_norm <= EPS:
                continue
            cosine = float(np.dot(velocity[idx - 1], velocity[idx]) / (left_norm * right_norm))
            curvature_raw[idx] = 0.5 * (1.0 - np.clip(cosine, -1.0, 1.0))

        relevance_gradient_raw = np.zeros(num_frames, dtype=float)
        if num_frames > 1:
            relevance_gradient_raw[1:] = np.abs(np.diff(relevance))

        speed = _robust_unit(_smooth_1d(speed_raw, self.config.smooth_window))
        curvature = _robust_unit(_smooth_1d(curvature_raw, self.config.smooth_window))
        relevance_gradient = _robust_unit(
            _smooth_1d(relevance_gradient_raw, self.config.smooth_window)
        )
        gate = self.config.query_gate_floor + (1.0 - self.config.query_gate_floor) * relevance

        transition = (
            self.config.speed_weight * speed
            + self.config.curvature_weight * curvature
        ) * gate
        energy_raw = transition + self.config.relevance_gradient_weight * relevance_gradient
        boundary_energy = _robust_unit(_smooth_1d(energy_raw, self.config.smooth_window))

        return SemanticFieldResult(
            speed=speed,
            curvature=curvature,
            relevance=relevance,
            relevance_gradient=relevance_gradient,
            query_gate=gate,
            boundary_energy=boundary_energy,
            normalized_features=trajectory,
            source=source,
        )


@dataclass
class NarrativeStructure:
    """Serializable panel sequence and its inferred temporal relationships."""

    panels: List[Dict[str, Any]]
    pages: List[List[int]]
    transitions: List[Dict[str, Any]]
    emphasis: Dict[str, float]
    field_source: str
    boundary_indices: List[int]
    gutter_target: float
    gutter_low: float
    gutter_high: float

    def to_dict(self, frame_indices: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        payload = asdict(self)
        payload["method"] = "svf_panel_v1"
        if frame_indices is not None:
            mapping = list(frame_indices)
            for panel in payload["panels"]:
                sampled = panel["sampled_index"]
                panel["frame_index"] = int(mapping[sampled]) if 0 <= sampled < len(mapping) else int(sampled)
            for transition in payload["transitions"]:
                for key in ("from_frame", "to_frame"):
                    sampled = transition[key]
                    transition[key] = int(mapping[sampled]) if 0 <= sampled < len(mapping) else int(sampled)
            payload["boundary_indices"] = [
                int(mapping[idx]) if 0 <= idx < len(mapping) else int(idx)
                for idx in payload["boundary_indices"]
            ]
        return payload


@dataclass
class SelectionResult:
    """Selected sampled-frame indices plus narrative metadata."""

    selected_indices: List[int]
    narrative: NarrativeStructure
    fields: SemanticFieldResult


class NarrativeBuilder:
    """Refine a fixed-budget frame sequence and infer comic-style metadata."""

    def __init__(self, config: SVFConfig) -> None:
        self.config = config

    @staticmethod
    def _arc_prefix(speed: np.ndarray) -> np.ndarray:
        num_frames = len(speed)
        prefix = np.zeros(num_frames, dtype=float)
        if num_frames <= 1:
            return prefix
        steps = np.asarray(speed[1:], dtype=float)
        total = float(np.sum(steps))
        if total <= EPS:
            steps = np.ones(num_frames - 1, dtype=float)
            total = float(num_frames - 1)
        prefix[1:] = np.cumsum(steps / total)
        return prefix

    @staticmethod
    def _gutter(prefix: np.ndarray, left: int, right: int) -> float:
        return float(max(0.0, prefix[right] - prefix[left]))

    def _bridge_candidate(
        self,
        left: int,
        right: int,
        prefix: np.ndarray,
        utility: np.ndarray,
        excluded: set[int],
    ) -> Optional[int]:
        candidates = [idx for idx in range(left + 1, right) if idx not in excluded]
        if not candidates:
            return None
        midpoint = 0.5 * (prefix[left] + prefix[right])
        gap = max(prefix[right] - prefix[left], EPS)
        return max(
            candidates,
            key=lambda idx: -abs(prefix[idx] - midpoint) / gap + 0.20 * float(utility[idx]),
        )

    def refine(self, selected: Sequence[int], utility: np.ndarray, speed: np.ndarray) -> List[int]:
        """Exchange redundant panels for bridge panels without changing budget."""

        panels = sorted(set(int(idx) for idx in selected))
        if len(panels) < 3:
            return panels
        prefix = self._arc_prefix(speed)
        target = 1.0 / max(len(panels) - 1, 1)
        low = self.config.gutter_low_factor * target
        high = self.config.gutter_high_factor * target
        seen = {tuple(panels)}

        for _ in range(max(0, self.config.refinement_steps)):
            gaps = [self._gutter(prefix, panels[i], panels[i + 1]) for i in range(len(panels) - 1)]
            widest_idx = int(np.argmax(gaps))
            closest_idx = int(np.argmin(gaps))

            if gaps[widest_idx] > high:
                wide_endpoints = {panels[widest_idx], panels[widest_idx + 1]}
                bridge = self._bridge_candidate(
                    panels[widest_idx], panels[widest_idx + 1], prefix, utility, set(panels)
                )
                if bridge is None:
                    break
                augmented = sorted(panels + [bridge])
                removable = [idx for idx in panels if idx not in wide_endpoints]
                if not removable:
                    removable = list(panels)

                def removal_cost(idx: int) -> float:
                    pos = augmented.index(idx)
                    neighbor_gaps: List[float] = []
                    if pos > 0:
                        neighbor_gaps.append(self._gutter(prefix, augmented[pos - 1], idx))
                    if pos + 1 < len(augmented):
                        neighbor_gaps.append(self._gutter(prefix, idx, augmented[pos + 1]))
                    redundancy = min(neighbor_gaps) if neighbor_gaps else 0.0
                    return float(utility[idx]) + 0.5 * redundancy

                remove = min(
                    removable,
                    key=removal_cost,
                )
                candidate_panels = sorted(idx for idx in augmented if idx != remove)
            elif gaps[closest_idx] < low:
                pair = [panels[closest_idx], panels[closest_idx + 1]]
                remove = min(pair, key=lambda idx: float(utility[idx]))
                reduced = [idx for idx in panels if idx != remove]
                reduced_gaps = [
                    self._gutter(prefix, reduced[i], reduced[i + 1])
                    for i in range(len(reduced) - 1)
                ]
                if not reduced_gaps:
                    break
                refill_gap = int(np.argmax(reduced_gaps))
                bridge = self._bridge_candidate(
                    reduced[refill_gap], reduced[refill_gap + 1], prefix, utility, set(reduced)
                )
                if bridge is None:
                    break
                candidate_panels = sorted(reduced + [bridge])
            else:
                break

            state = tuple(candidate_panels)
            if state in seen:
                break
            seen.add(state)
            panels = candidate_panels

        return panels

    def _transition_type(
        self,
        left: int,
        right: int,
        gutter: float,
        target: float,
        fields: SemanticFieldResult,
    ) -> str:
        interval = slice(left + 1, right + 1)
        speed_peak = float(np.max(fields.speed[interval])) if right > left else 0.0
        curvature_peak = float(np.max(fields.curvature[interval])) if right > left else 0.0
        direct_similarity = float(np.dot(fields.normalized_features[left], fields.normalized_features[right]))
        direct_change = 0.5 * (1.0 - np.clip(direct_similarity, -1.0, 1.0))
        relevance_continuity = 1.0 - abs(float(fields.relevance[left] - fields.relevance[right]))

        low = self.config.gutter_low_factor * target
        high = self.config.gutter_high_factor * target
        if direct_change > 0.80 and relevance_continuity < 0.35:
            return "non_sequitur"
        if gutter > high or (speed_peak > 0.75 and curvature_peak > 0.65):
            return "scene_to_scene"
        if curvature_peak > 0.65 and direct_change > 0.45:
            return "subject_to_subject"
        if gutter < low and curvature_peak < 0.35:
            return "moment_to_moment"
        if curvature_peak < 0.40:
            return "action_to_action"
        return "aspect_to_aspect"

    def build(
        self,
        selected: Sequence[int],
        fields: SemanticFieldResult,
        utility: np.ndarray,
        boundaries: Sequence[int],
    ) -> NarrativeStructure:
        panels = sorted(int(idx) for idx in selected)
        prefix = self._arc_prefix(fields.speed)
        target = 1.0 / max(len(panels) - 1, 1) if len(panels) > 1 else 0.0
        low = self.config.gutter_low_factor * target
        high = self.config.gutter_high_factor * target

        transition_rows: List[Dict[str, Any]] = []
        for pos in range(len(panels) - 1):
            left, right = panels[pos], panels[pos + 1]
            gutter = self._gutter(prefix, left, right)
            transition_rows.append(
                {
                    "from": pos,
                    "to": pos + 1,
                    "from_frame": left,
                    "to_frame": right,
                    "type": self._transition_type(left, right, gutter, target, fields),
                    "gutter": round(gutter, 6),
                }
            )

        pages: List[List[int]] = []
        if panels:
            current = [0]
            page_threshold = self.config.page_gutter_factor * target
            for pos, transition in enumerate(transition_rows):
                starts_new_page = (
                    transition["gutter"] > page_threshold
                    or transition["type"] in {"scene_to_scene", "non_sequitur"}
                )
                if starts_new_page:
                    pages.append(current)
                    current = [pos + 1]
                else:
                    current.append(pos + 1)
            pages.append(current)

        salience = np.asarray(
            [fields.relevance[idx] * (0.5 * fields.speed[idx] + 0.5 * fields.boundary_energy[idx]) for idx in panels],
            dtype=float,
        )
        if len(salience) and np.max(salience) <= EPS:
            salience = np.asarray([fields.relevance[idx] for idx in panels], dtype=float)
        salience = _robust_unit(salience)
        emphasis_count = min(len(panels), max(1, int(np.ceil(len(panels) * self.config.emphasis_fraction))))
        emphasized = set(np.argsort(salience)[-emphasis_count:].tolist()) if panels else set()
        emphasis = {str(pos): round(1.0 + float(salience[pos]), 4) for pos in sorted(emphasized)}

        panel_rows = [
            {
                "position": pos,
                "sampled_index": idx,
                "relevance": round(float(fields.relevance[idx]), 6),
                "boundary_energy": round(float(fields.boundary_energy[idx]), 6),
            }
            for pos, idx in enumerate(panels)
        ]
        return NarrativeStructure(
            panels=panel_rows,
            pages=pages,
            transitions=transition_rows,
            emphasis=emphasis,
            field_source=fields.source,
            boundary_indices=[int(idx) for idx in boundaries],
            gutter_target=round(target, 6),
            gutter_low=round(low, 6),
            gutter_high=round(high, 6),
        )


class SVFPanelSelector:
    """End-to-end Semantic Velocity Field and panelized frame selector."""

    def __init__(self, config: Optional[SVFConfig] = None) -> None:
        self.config = config or SVFConfig()
        self.field_builder = SemanticVelocityField(self.config)
        self.narrative_builder = NarrativeBuilder(self.config)

    def detect_boundaries(self, energy: np.ndarray, min_distance: int) -> np.ndarray:
        """Find robust local maxima in query-conditioned boundary energy."""

        signal = np.asarray(energy, dtype=float)
        if len(signal) < 3 or np.max(signal) <= EPS:
            return np.array([], dtype=int)
        median = float(np.median(signal))
        mad = float(np.median(np.abs(signal - median)))
        height = median + self.config.peak_height_factor * 1.4826 * mad
        prominence = self.config.peak_prominence_factor * float(np.ptp(signal))
        peaks, _ = find_peaks(
            signal,
            height=height,
            prominence=prominence,
            distance=max(1, int(min_distance)),
        )
        return peaks.astype(int)

    @staticmethod
    def _segments(boundaries: Sequence[int], total_frames: int) -> List[Tuple[int, int]]:
        points = [0] + sorted({int(idx) for idx in boundaries if 0 < idx < total_frames}) + [total_frames]
        return [(points[idx], points[idx + 1]) for idx in range(len(points) - 1)]

    def _segment_importance(self, segment: Tuple[int, int], fields: SemanticFieldResult) -> float:
        start, end = segment
        relevance = fields.relevance[start:end]
        motion = fields.boundary_energy[start:end]
        duration = (end - start) / max(len(fields.relevance), 1)
        return float(
            self.config.segment_duration_weight * duration
            + self.config.segment_mean_relevance_weight * np.mean(relevance)
            + self.config.segment_max_relevance_weight * np.max(relevance)
            + self.config.segment_motion_weight * np.mean(motion)
        )

    @staticmethod
    def _allocate_budget(
        importance: np.ndarray,
        capacities: np.ndarray,
        budget: int,
    ) -> Dict[int, int]:
        allocation = np.zeros(len(importance), dtype=int)
        active = [idx for idx, cap in enumerate(capacities) if cap > 0]
        if budget <= 0 or not active:
            return {}

        if budget >= len(active):
            allocation[active] = 1
            remaining = budget - len(active)
        else:
            chosen = sorted(active, key=lambda idx: float(importance[idx]), reverse=True)[:budget]
            allocation[chosen] = 1
            remaining = 0

        while remaining > 0:
            candidates = [idx for idx in active if allocation[idx] < capacities[idx]]
            if not candidates:
                break
            best = max(candidates, key=lambda idx: float(importance[idx]) / (allocation[idx] + 1.0))
            allocation[best] += 1
            remaining -= 1
        return {idx: int(count) for idx, count in enumerate(allocation) if count > 0}

    def _frame_utility(self, fields: SemanticFieldResult) -> np.ndarray:
        return (
            self.config.frame_relevance_weight * fields.relevance
            + self.config.frame_boundary_weight * fields.boundary_energy
            + self.config.frame_speed_weight * fields.speed
            + self.config.frame_curvature_weight * fields.curvature
        )

    def _select_mmr(
        self,
        candidates: Sequence[int],
        count: int,
        utility: np.ndarray,
        features: np.ndarray,
        seed: Optional[Sequence[int]] = None,
    ) -> List[int]:
        selected = list(dict.fromkeys(int(idx) for idx in (seed or [])))
        pool = [int(idx) for idx in candidates if int(idx) not in selected]
        while len(selected) < count and pool:
            if not selected:
                best = max(pool, key=lambda idx: float(utility[idx]))
            else:
                best = max(
                    pool,
                    key=lambda idx: self.config.mmr_lambda * float(utility[idx])
                    - (1.0 - self.config.mmr_lambda)
                    * float(np.max(features[selected] @ features[idx])),
                )
            selected.append(best)
            pool.remove(best)
        return selected[:count]

    def select_with_narrative(
        self,
        relevance_scores: np.ndarray,
        num_frames: int,
        features: Optional[np.ndarray] = None,
        min_boundary_distance: Optional[int] = None,
    ) -> SelectionResult:
        """Select a fixed-size frame set and construct narrative metadata."""

        scores = np.asarray(relevance_scores, dtype=float).reshape(-1)
        total_frames = len(scores)
        if total_frames == 0:
            raise ValueError("relevance_scores must not be empty")
        budget = min(max(0, int(num_frames)), total_frames)
        fields = self.field_builder.compute(scores, features)
        distance = min_boundary_distance or compute_min_boundary_distance(
            total_frames,
            self.config.min_distance_ratio,
            self.config.min_distance_absolute,
        )
        boundaries = self.detect_boundaries(fields.boundary_energy, distance)
        segments = self._segments(boundaries, total_frames)
        utility = self._frame_utility(fields)

        importance = np.asarray([self._segment_importance(segment, fields) for segment in segments])
        capacities = np.asarray([end - start for start, end in segments], dtype=int)
        allocation = self._allocate_budget(importance, capacities, budget)

        selected: List[int] = []
        for segment_idx, count in allocation.items():
            start, end = segments[segment_idx]
            chosen = self._select_mmr(
                candidates=range(start, end),
                count=count,
                utility=utility,
                features=fields.normalized_features,
            )
            selected.extend(chosen)

        if len(selected) < budget:
            selected = self._select_mmr(
                candidates=range(total_frames),
                count=budget,
                utility=utility,
                features=fields.normalized_features,
                seed=selected,
            )
        elif len(selected) > budget:
            selected = sorted(selected, key=lambda idx: float(utility[idx]), reverse=True)[:budget]

        selected = self.narrative_builder.refine(selected, utility, fields.speed)
        selected = sorted(selected)
        narrative = self.narrative_builder.build(selected, fields, utility, boundaries)
        return SelectionResult(selected_indices=selected, narrative=narrative, fields=fields)

    def select_keyframes(
        self,
        relevance_scores: np.ndarray,
        num_frames: int,
        features: Optional[np.ndarray] = None,
        min_boundary_distance: Optional[int] = None,
    ) -> List[int]:
        """Compatibility wrapper returning only sampled-frame indices."""

        return self.select_with_narrative(
            relevance_scores=relevance_scores,
            num_frames=num_frames,
            features=features,
            min_boundary_distance=min_boundary_distance,
        ).selected_indices


def compute_min_boundary_distance(
    num_frames: int,
    ratio: float = 0.02,
    absolute_min: int = 3,
) -> int:
    """Compute the minimum spacing between semantic-boundary peaks."""

    return max(int(absolute_min), int(num_frames * ratio), 1)
