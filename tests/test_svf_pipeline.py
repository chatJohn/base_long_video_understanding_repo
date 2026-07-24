"""Small end-to-end test for the standalone SVF-Panel pipeline."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from svf.pipeline import build_parser, run_pipeline


class SVFPipelineTests(unittest.TestCase):
    def test_lvb_artifacts_are_mapped_to_source_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            features_dir = root / "features"
            artifact_dir = features_dir / "0"
            artifact_dir.mkdir(parents=True)
            questions_file = root / "questions.json"
            output_file = root / "result.json"

            item = {
                "id": "q0",
                "video_id": "v0",
                "video_path": "missing.mp4",
                "question": "What changes?",
                "candidates": ["A", "B"],
                "correct_choice": 0,
            }
            questions_file.write_text(json.dumps([item]), encoding="utf-8")

            frame_indices = [idx * 25 for idx in range(12)]
            score_payload = {
                "frame_indices": frame_indices,
                "blip2_similarities": np.linspace(0.1, 0.9, 12).tolist(),
            }
            (artifact_dir / "similarity_scores.json").write_text(
                json.dumps(score_payload),
                encoding="utf-8",
            )
            features = np.vstack([np.eye(3)[idx % 3] for idx in range(12)])
            with (artifact_dir / "blip2_vision_features.pkl").open("wb") as handle:
                pickle.dump(features, handle)

            args = build_parser().parse_args(
                [
                    "--benchmark",
                    "lvb",
                    "--dataset_root",
                    str(dataset_root),
                    "--questions_file",
                    str(questions_file),
                    "--features_dir",
                    str(features_dir),
                    "--output_path",
                    str(output_file),
                    "--max_frames",
                    "4",
                ]
            )
            summary = run_pipeline(args)
            output = json.loads(output_file.read_text(encoding="utf-8"))[0]

            panel_frames = [panel["frame_index"] for panel in output["narrative"]["panels"]]
            self.assertEqual(summary["success"], 1)
            self.assertEqual(summary["fallback"], 0)
            self.assertEqual(len(output["keyframe_indices"]), 4)
            self.assertEqual(output["keyframe_indices"], panel_frames)
            self.assertTrue(set(panel_frames).issubset(set(frame_indices)))


if __name__ == "__main__":
    unittest.main()
