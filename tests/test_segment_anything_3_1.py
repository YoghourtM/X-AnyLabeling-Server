from collections import OrderedDict
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import MagicMock

import numpy as np
import yaml

from app.models import segment_anything_3_1 as sam31
from app.models.segment_anything_3_1 import (
    SegmentAnything31,
    interactive_prompt,
    load_detector_weights,
    mask_to_polygon,
    split_marks,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeTorch:
    """Just enough of torch for predict(): no-op contexts."""

    class _Context:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    bfloat16 = "bf16"

    def autocast(self, *args, **kwargs):
        return self._Context()

    def inference_mode(self):
        return self._Context()


def _model(**params):
    config = yaml.safe_load(
        (ROOT / "configs/auto_labeling/segment_anything_3_1.yaml").read_text()
    )
    config["params"].update(params)
    model = SegmentAnything31(config)
    model.torch = FakeTorch()
    model._lock = threading.RLock()
    model._image_state = None
    model._sessions = OrderedDict()
    model._sessions_with_object = set()
    model.processor = MagicMock()
    model.checkpoint = "/ckpt/sam3.1_multiplex.pt"
    return model


def _square_mask(height=40, width=60):
    mask = np.zeros((height, width), dtype=bool)
    mask[10:30, 20:50] = True
    return mask


class TestHelpers(unittest.TestCase):
    def test_split_marks(self):
        points, point_labels, boxes, box_labels = split_marks(
            [
                {"type": "point", "data": [1, 2], "label": 1},
                {"type": "point", "data": [3, 4], "label": 0},
                {"type": "rectangle", "data": [9, 8, 1, 2], "label": 1},
                {"type": "rectangle", "data": [1, 2]},
            ]
        )
        self.assertEqual(points, [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(point_labels, [1, 0])
        self.assertEqual(boxes, [[1.0, 2.0, 9.0, 8.0]])
        self.assertEqual(box_labels, [1])

    def test_interactive_prompt_adds_box_corners(self):
        points, labels = interactive_prompt(
            [
                {"type": "point", "data": [50, 25], "label": 0},
                {"type": "rectangle", "data": [10, 10, 90, 40], "label": 1},
            ],
            100,
            50,
        )
        self.assertEqual(points, [[0.5, 0.5], [0.1, 0.2], [0.9, 0.8]])
        self.assertEqual(labels, [0, 2, 3])

    def test_mask_to_polygon_is_closed(self):
        polygon = mask_to_polygon(_square_mask(), 0.001)
        self.assertGreaterEqual(len(polygon), 5)
        self.assertEqual(polygon[0], polygon[-1])
        xs = [x for x, _ in polygon]
        self.assertEqual((min(xs), max(xs)), (20.0, 49.0))
        self.assertEqual(mask_to_polygon(np.zeros((5, 5))), [])

    def test_detector_weights_fail_closed(self):
        neck = types.SimpleNamespace(
            convs=[0, 1, 2, 3], scale_factors=[4.0, 2.0, 1.0, 0.5]
        )
        model = MagicMock()
        model.backbone.scalp = 1
        model.backbone.vision_backbone = neck
        unused = sorted(sam31._UNUSED_FPN_KEYS)
        model.load_state_dict.side_effect = [(unused, ["x"]), ([], ["x"])]
        report = load_detector_weights(model, {"w": 1})
        self.assertTrue(report["removed_unused_fpn_level"])
        self.assertEqual(len(neck.convs), 3)
        self.assertEqual(model.backbone.scalp, 0)

        model.load_state_dict.side_effect = [(["transformer.layer.w"], [])]
        model.backbone.scalp = 1
        with self.assertRaisesRegex(RuntimeError, "missing"):
            load_detector_weights(model, {"w": 1})

    def test_vendored_sam3_is_rejected(self):
        vendored = types.ModuleType("sam3")
        vendored.__file__ = str(
            ROOT / "app" / "models" / "sam3" / "__init__.py"
        )
        saved = sys.modules.get("sam3")
        sys.modules["sam3"] = vendored
        try:
            with self.assertRaisesRegex(RuntimeError, "vendored"):
                sam31.import_upstream_sam3()
        finally:
            if saved is None:
                sys.modules.pop("sam3", None)
            else:
                sys.modules["sam3"] = saved


class TestConfigAndMetadata(unittest.TestCase):
    def test_registration_config_and_tasks(self):
        from app.core.registry import ModelRegistry, _MODEL_REGISTRY

        self.assertIs(
            _MODEL_REGISTRY["segment_anything_3_1"], SegmentAnything31
        )
        model = _model()
        ModelRegistry.__new__(ModelRegistry)._validate_configs([model.config])
        tasks = model.get_metadata()["available_tasks"]
        self.assertEqual(
            [t["id"] for t in tasks], ["text", "interactive", "exemplar"]
        )
        declared = {w["name"] for w in model.config["widgets"]}
        for task in tasks:
            self.assertLessEqual(set(task["active_widgets"]), declared)
        hidden = _model(interactive=False).get_metadata()["available_tasks"]
        self.assertNotIn("interactive", [t["id"] for t in hidden])
        enabled = yaml.safe_load(
            (ROOT / "configs/models.sam31.yaml").read_text()
        )["enabled_models"]
        self.assertEqual(enabled, ["segment_anything_3_1"])


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((40, 60, 3), dtype=np.uint8)

    def test_text_prompts_label_each_instance(self):
        model = _model()
        mask = _square_mask()[None]
        model.processor.set_image.return_value = {}
        model.processor.set_text_prompt.side_effect = [
            {
                "masks": np.stack([mask, mask]),
                "boxes": np.array([[20, 10, 49, 29]] * 2, dtype=float),
                "scores": np.array([0.9, 0.6]),
            },
            {"masks": [], "boxes": [], "scores": []},
        ]
        result = model.predict(
            self.image,
            {
                "current_task": "text",
                "text_prompt": "car; tree",
                "conf_threshold": 0.3,
            },
        )
        self.assertEqual([s.label for s in result["shapes"]], ["car", "car"])
        model.processor.set_confidence_threshold.assert_called_with(0.3)
        # The image is encoded once and reused for later prompts.
        model.predict(self.image, {"current_task": "text", "text_prompt": ""})
        model.processor.set_text_prompt.side_effect = None
        model.processor.set_text_prompt.return_value = {"scores": []}
        model.predict(self.image, {"current_task": "text", "text_prompt": "x"})
        self.assertEqual(model.processor.set_image.call_count, 1)

    def test_exemplar_boxes_are_normalized_cxcywh(self):
        model = _model()
        model.processor.set_image.return_value = {}
        model.processor.add_geometric_prompt.return_value = {
            "masks": _square_mask()[None][None],
            "boxes": np.array([[20, 10, 49, 29]], dtype=float),
            "scores": np.array([0.8]),
        }
        result = model.predict(
            self.image,
            {
                "current_task": "exemplar",
                "marks": [
                    {"type": "rectangle", "data": [0, 0, 30, 20], "label": 1}
                ],
            },
        )
        kwargs = model.processor.add_geometric_prompt.call_args.kwargs
        self.assertEqual(kwargs["box"], [0.25, 0.25, 0.5, 0.5])
        self.assertTrue(kwargs["label"])
        self.assertEqual(len(result["shapes"]), 1)

    def _fake_predictor(self, model):
        predictor = MagicMock()
        predictor._all_inference_states = {}
        predictor.model.init_state.return_value = {"frames": 1}
        empty = {"out_obj_ids": [], "out_binary_masks": [], "out_probs": []}
        full = {
            "out_obj_ids": [1],
            "out_binary_masks": [_square_mask()],
            "out_probs": [0.97],
        }
        calls = []

        def handle(request):
            calls.append(request["type"])
            if request["type"] != "add_prompt":
                return {}
            # First add of an object returns nothing (upstream quirk).
            first = calls.count("add_prompt") % 2 == 1
            return {"outputs": empty if first else full}

        predictor.handle_request.side_effect = handle
        model._predictor = predictor
        return predictor, calls

    def test_interactive_retries_first_add_and_resets_object(self):
        model = _model()
        predictor, calls = self._fake_predictor(model)
        marks = [{"type": "point", "data": [30, 20], "label": 1}]
        result = model.predict(
            self.image, {"current_task": "interactive", "marks": marks}
        )
        self.assertEqual(calls, ["add_prompt", "add_prompt"])
        self.assertFalse(result["replace"])
        [shape] = result["shapes"]
        self.assertEqual(shape.label, "AUTOLABEL_OBJECT")
        request = predictor.handle_request.call_args_list[0].args[0]
        self.assertEqual(request["points"], [[0.5, 0.5]])
        self.assertEqual(predictor.model.init_state.call_count, 1)

        model.predict(
            self.image, {"current_task": "interactive", "marks": marks}
        )
        self.assertEqual(calls[2], "remove_object")
        self.assertEqual(predictor.model.init_state.call_count, 1)

    def test_tracker_sessions_are_lru_evicted(self):
        model = _model(cache_size=1)
        predictor, calls = self._fake_predictor(model)
        marks = [{"type": "point", "data": [3, 3], "label": 1}]
        for value in (0, 1):
            image = np.full((40, 60, 3), value, dtype=np.uint8)
            model.predict(
                image, {"current_task": "interactive", "marks": marks}
            )
        self.assertIn("close_session", calls)
        self.assertEqual(len(model._sessions), 1)

    def test_marks_without_task_default_to_interactive(self):
        model = _model()
        predictor, _ = self._fake_predictor(model)
        model.predict(
            self.image,
            {"marks": [{"type": "point", "data": [1, 1], "label": 1}]},
        )
        predictor.handle_request.assert_called()


if __name__ == "__main__":
    unittest.main()
