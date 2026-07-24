"""Unit tests for the training-free SVF-Panel selector."""

import unittest

import numpy as np

from svf import SVFConfig, SVFPanelSelector, SemanticVelocityField


class SemanticVelocityFieldTests(unittest.TestCase):
    def test_scene_change_produces_speed_response(self) -> None:
        features = np.vstack(
            [
                np.tile([1.0, 0.0, 0.0], (6, 1)),
                np.tile([0.0, 1.0, 0.0], (6, 1)),
            ]
        )
        scores = np.linspace(0.2, 0.9, 12)
        fields = SemanticVelocityField(SVFConfig(smooth_window=1)).compute(scores, features)

        self.assertEqual(fields.source, "visual_features")
        self.assertEqual(len(fields.speed), len(scores))
        self.assertGreater(fields.speed[6], 0.9)
        self.assertTrue(np.all((fields.boundary_energy >= 0.0) & (fields.boundary_energy <= 1.0)))

    def test_missing_features_uses_query_proxy(self) -> None:
        fields = SemanticVelocityField().compute(np.array([0.1, 0.3, 0.2]), None)
        self.assertEqual(fields.source, "query_proxy")
        self.assertEqual(fields.normalized_features.shape[0], 3)


class SVFPanelSelectorTests(unittest.TestCase):
    def test_selection_is_sorted_unique_and_exact_budget(self) -> None:
        rng = np.random.default_rng(7)
        first = rng.normal(loc=[1.0, 0.0, 0.0], scale=0.03, size=(10, 3))
        second = rng.normal(loc=[0.0, 1.0, 0.0], scale=0.03, size=(10, 3))
        third = rng.normal(loc=[0.0, 0.0, 1.0], scale=0.03, size=(10, 3))
        features = np.vstack([first, second, third])
        scores = np.concatenate(
            [np.linspace(0.2, 0.5, 10), np.linspace(0.8, 0.6, 10), np.linspace(0.3, 0.9, 10)]
        )

        result = SVFPanelSelector().select_with_narrative(scores, 8, features)

        self.assertEqual(len(result.selected_indices), 8)
        self.assertEqual(result.selected_indices, sorted(set(result.selected_indices)))
        self.assertEqual(len(result.narrative.panels), 8)
        self.assertEqual(len(result.narrative.transitions), 7)
        self.assertEqual(
            sorted(position for page in result.narrative.pages for position in page),
            list(range(8)),
        )

    def test_flat_signal_still_respects_budget(self) -> None:
        scores = np.ones(10)
        features = np.ones((10, 4))
        result = SVFPanelSelector().select_with_narrative(scores, 4, features)

        self.assertEqual(len(result.selected_indices), 4)
        self.assertEqual(len(set(result.selected_indices)), 4)
        self.assertTrue(all(0 <= idx < 10 for idx in result.selected_indices))

    def test_narrative_maps_to_original_frame_indices(self) -> None:
        scores = np.linspace(0.0, 1.0, 8)
        features = np.eye(8)
        result = SVFPanelSelector().select_with_narrative(scores, 4, features)
        mapping = [idx * 30 for idx in range(8)]
        narrative = result.narrative.to_dict(mapping)

        mapped = [panel["frame_index"] for panel in narrative["panels"]]
        expected = [mapping[idx] for idx in result.selected_indices]
        self.assertEqual(mapped, expected)
        self.assertEqual(narrative["method"], "svf_panel_v1")


if __name__ == "__main__":
    unittest.main()
