"""Benchmark adapters used by the standalone SVF-Panel pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def similarity_key(feature_model: str) -> str:
    return f"{feature_model}_similarities"


def feature_filename(feature_model: str) -> str:
    return f"{feature_model}_vision_features.pkl"


@dataclass
class BenchmarkRecord:
    index: int
    raw: Dict[str, Any]
    feature_id: str
    video_path: Path
    question_index: Optional[int] = None


class BenchmarkAdapter:
    name = "base"

    def load_raw(self, questions_file: Path) -> List[Dict[str, Any]]:
        with questions_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"{self.name}: annotation file must contain a JSON list")
        return data

    def build_record(
        self,
        index: int,
        item: Dict[str, Any],
        dataset_root: Path,
    ) -> BenchmarkRecord:
        raise NotImplementedError

    def get_scores(
        self,
        scores_payload: Dict[str, Any],
        record: BenchmarkRecord,
        feature_model: str,
    ) -> Optional[np.ndarray]:
        raise NotImplementedError

    def to_output_item(
        self,
        record: BenchmarkRecord,
        keyframe_indices: List[int],
    ) -> Dict[str, Any]:
        item = dict(record.raw)
        item["keyframe_indices"] = [int(idx) for idx in keyframe_indices]
        return item


class VideoMMEAdapter(BenchmarkAdapter):
    name = "videomme"

    def build_record(
        self,
        index: int,
        item: Dict[str, Any],
        dataset_root: Path,
    ) -> BenchmarkRecord:
        question_index: Optional[int] = None
        question_id = item.get("question_id", "")
        if isinstance(question_id, str) and "-" in question_id:
            try:
                question_index = int(question_id.split("-")[-1]) - 1
            except ValueError:
                pass
        return BenchmarkRecord(
            index=index,
            raw=item,
            feature_id=str(item["video_id"]),
            video_path=dataset_root / "data" / f"{item['videoID']}.mp4",
            question_index=question_index,
        )

    def get_scores(
        self,
        scores_payload: Dict[str, Any],
        record: BenchmarkRecord,
        feature_model: str,
    ) -> Optional[np.ndarray]:
        key = similarity_key(feature_model)
        questions = scores_payload.get("questions", [])
        if not isinstance(questions, list):
            return None
        if record.question_index is not None:
            for question in questions:
                if question.get("question_index") == record.question_index and key in question:
                    return np.asarray(question[key], dtype=float)
        # Some preprocessing outputs preserve global rather than per-video order.
        if 0 <= record.index < len(questions) and key in questions[record.index]:
            return np.asarray(questions[record.index][key], dtype=float)
        return None


class LongVideoBenchAdapter(BenchmarkAdapter):
    name = "lvb"

    def build_record(
        self,
        index: int,
        item: Dict[str, Any],
        dataset_root: Path,
    ) -> BenchmarkRecord:
        return BenchmarkRecord(
            index=index,
            raw=item,
            feature_id=str(index),
            video_path=dataset_root / "videos" / item["video_path"],
        )

    def get_scores(
        self,
        scores_payload: Dict[str, Any],
        record: BenchmarkRecord,
        feature_model: str,
    ) -> Optional[np.ndarray]:
        del record
        key = similarity_key(feature_model)
        return np.asarray(scores_payload[key], dtype=float) if key in scores_payload else None


class MLVUAdapter(BenchmarkAdapter):
    name = "mlvu"

    def build_record(
        self,
        index: int,
        item: Dict[str, Any],
        dataset_root: Path,
    ) -> BenchmarkRecord:
        return BenchmarkRecord(
            index=index,
            raw=item,
            feature_id=str(item["question_id"]),
            video_path=dataset_root / "video" / item["video_name"],
        )

    def get_scores(
        self,
        scores_payload: Dict[str, Any],
        record: BenchmarkRecord,
        feature_model: str,
    ) -> Optional[np.ndarray]:
        del record
        key = similarity_key(feature_model)
        return np.asarray(scores_payload[key], dtype=float) if key in scores_payload else None


def create_adapter(benchmark: str) -> BenchmarkAdapter:
    normalized = benchmark.lower().strip()
    if normalized == "videomme":
        return VideoMMEAdapter()
    if normalized in {"lvb", "longvideobench"}:
        return LongVideoBenchAdapter()
    if normalized == "mlvu":
        return MLVUAdapter()
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def benchmark_defaults(benchmark: str) -> Dict[str, Any]:
    normalized = benchmark.lower().strip()
    if normalized == "videomme":
        return {
            "dataset_root": Path("datasets/videomme"),
            "questions_file": Path("datasets/videomme/videomme_json_file.json"),
            "features_dir": Path("datasets/videomme/blip2_features_and_scores"),
            "max_frames": 16,
        }
    if normalized in {"lvb", "longvideobench"}:
        return {
            "dataset_root": Path("datasets/longvideobench"),
            "questions_file": Path("datasets/longvideobench/lvb_val.json"),
            "features_dir": Path("datasets/longvideobench/blip2_features_and_scores"),
            "max_frames": 16,
        }
    if normalized == "mlvu":
        return {
            "dataset_root": Path("datasets/mlvu"),
            "questions_file": Path("datasets/mlvu/mlvu_dev.json"),
            "features_dir": Path("datasets/mlvu/blip2_features_and_scores"),
            "max_frames": 16,
        }
    raise ValueError(f"Unsupported benchmark: {benchmark}")
