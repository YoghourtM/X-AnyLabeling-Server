"""Segment Anything 3.1 for images, backed by the upstream ``sam3`` package.

One ``sam3.1_multiplex.pt`` checkpoint serves three tasks:

- ``text``: segment every instance of the prompted concepts (image detector).
- ``exemplar``: positive / negative example boxes find similar objects.
- ``interactive``: clicks or one box segment a single object with the SAM 3.1
  multiplex tracker, built lazily on first use.

The vendored ``app/models/sam3`` copy predates SAM 3.1 (no multiplex code),
so this model imports the installed upstream package instead.
"""

import hashlib
import os
import sys
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from PIL import Image

from . import BaseModel, parse_prompts
from app.core.registry import register_model
from app.schemas.shape import Shape

TASK_TEXT = "text"
TASK_EXEMPLAR = "exemplar"
TASK_INTERACTIVE = "interactive"
INTERACTIVE_OBJ_ID = 1

# SAM 3.1 has three image FPN levels; the SAM 3 image builder creates four
# and crops the last one (scalp=1), so its weights are absent and unused.
_UNUSED_FPN_KEYS = frozenset(
    f"backbone.vision_backbone.convs.3.{conv}.{kind}"
    for conv in ("conv_1x1", "conv_3x3")
    for kind in ("weight", "bias")
)


def import_upstream_sam3():
    """Import the installed ``sam3`` package, never the vendored copy."""
    vendored_parent = os.path.dirname(os.path.abspath(__file__))
    loaded = sys.modules.get("sam3")
    if loaded is not None:
        location = os.path.abspath(getattr(loaded, "__file__", "") or "")
        if location.startswith(vendored_parent + os.sep):
            raise RuntimeError(
                "The vendored SAM 3 package is already imported (is "
                "segment_anything_3 enabled too?). Enable only one of "
                "segment_anything_3 / segment_anything_3_1 per server."
            )
    sys.path[:] = [
        entry
        for entry in sys.path
        if os.path.abspath(entry or os.curdir) != vendored_parent
    ]
    import sam3.model_builder as model_builder

    return model_builder


def load_detector_weights(model, state: Dict[str, Any]) -> Dict[str, Any]:
    """Load SAM 3.1 detector weights into a SAM 3 image model, fail closed.

    The upstream loader uses ``strict=False`` and silently tolerates missing
    weights. Only the unused fourth FPN branch may be absent; it is removed
    together with its crop so every consumed tensor comes from the file.
    """
    missing, unexpected = model.load_state_dict(state, strict=False)
    removed_fpn_level = False
    if missing:
        neck = model.backbone.vision_backbone
        if (
            set(missing) != _UNUSED_FPN_KEYS
            or model.backbone.scalp != 1
            or len(neck.convs) != 4
        ):
            raise RuntimeError(
                "SAM 3.1 checkpoint is missing image detector weights: "
                f"{sorted(missing)[:10]}"
            )
        neck.convs = neck.convs[:3]
        neck.scale_factors = neck.scale_factors[:3]
        model.backbone.scalp = 0
        removed_fpn_level = True
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Missing detector weights: {missing[:10]}")
    return {
        "tensors": len(state),
        "unexpected_keys": len(unexpected),
        "removed_unused_fpn_level": removed_fpn_level,
    }


def split_marks(marks: List[Dict[str, Any]]):
    """Return (points, point_labels, boxes, box_labels) in pixels."""
    points, point_labels, boxes, box_labels = [], [], [], []
    for mark in marks or []:
        data = mark.get("data") or []
        label = 1 if mark.get("label", 1) == 1 else 0
        if mark.get("type") == "point" and len(data) == 2:
            points.append([float(data[0]), float(data[1])])
            point_labels.append(label)
        elif mark.get("type") == "rectangle" and len(data) == 4:
            x1, y1, x2, y2 = (float(v) for v in data)
            boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
            box_labels.append(label)
    return points, point_labels, boxes, box_labels


def interactive_prompt(
    marks: List[Dict[str, Any]], width: int, height: int
) -> Tuple[List[List[float]], List[int]]:
    """Clicks plus at most one positive box, as SAM tracker point prompts.

    A box becomes its two corners labelled 2 / 3 (the SAM convention);
    coordinates are relative to the image size.
    """
    points, labels, boxes, box_labels = split_marks(marks)
    rel_points = [[x / width, y / height] for x, y in points]
    positive_boxes = [b for b, l in zip(boxes, box_labels) if l == 1]
    if positive_boxes:
        x1, y1, x2, y2 = positive_boxes[-1]
        rel_points += [[x1 / width, y1 / height], [x2 / width, y2 / height]]
        labels = labels + [2, 3]
    return rel_points, labels


def mask_to_polygon(
    mask: np.ndarray, epsilon_factor: float = 0.001
) -> List[List[float]]:
    """Largest external contour of a binary mask as a closed polygon."""
    import cv2

    mask = np.asarray(mask).squeeze()
    contours, _ = cv2.findContours(
        (mask > 0.5).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if epsilon_factor > 0:
        contour = cv2.approxPolyDP(
            contour, epsilon_factor * cv2.arcLength(contour, True), True
        )
    points = [[float(x), float(y)] for x, y in contour.reshape(-1, 2)]
    if len(points) < 3:
        return []
    return points + [points[0]]


def image_key(image: np.ndarray) -> str:
    digest = hashlib.sha1(np.ascontiguousarray(image).data)
    digest.update(str(image.shape).encode())
    return digest.hexdigest()


def _task_widgets(*names: str) -> Dict[str, Dict[str, Any]]:
    return {name: {} for name in names}


TASKS = [
    {
        "id": TASK_TEXT,
        "name": "Text prompt",
        "description": "Segment every instance of the given concepts, "
        "e.g. 'person, car'",
        "batch_processing_mode": "text_prompt",
        "active_widgets": {
            **_task_widgets(
                "edit_text",
                "button_send",
                "input_conf",
                "edit_conf",
                "mask_fineness_slider",
                "mask_fineness_value_label",
                "toggle_preserve_existing_annotations",
            ),
            "edit_text": {"placeholder": "person, car, traffic light"},
        },
    },
    {
        "id": TASK_INTERACTIVE,
        "name": "Click / box (single object)",
        "description": "Positive / negative clicks or one box segment one "
        "object",
        "batch_processing_mode": None,
        "active_widgets": _task_widgets(
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
            "mask_fineness_slider",
            "mask_fineness_value_label",
        ),
    },
    {
        "id": TASK_EXEMPLAR,
        "name": "Example boxes (find similar)",
        "description": "Box one or more examples; every similar object is "
        "segmented",
        "batch_processing_mode": None,
        "active_widgets": _task_widgets(
            "add_pos_rect",
            "add_neg_rect",
            "button_run_rect",
            "button_clear",
            "input_conf",
            "edit_conf",
            "mask_fineness_slider",
            "mask_fineness_value_label",
            "toggle_preserve_existing_annotations",
        ),
    },
]


@register_model("segment_anything_3_1")
class SegmentAnything31(BaseModel):
    """SAM 3.1 text, exemplar and interactive image segmentation."""

    def load(self):
        import torch

        self.torch = torch
        builder = import_upstream_sam3()
        from sam3.model.sam3_image_processor import Sam3Processor

        self.checkpoint = self.params["model_path"]
        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(f"SAM 3.1 checkpoint: {self.checkpoint}")
        if not torch.cuda.is_available():
            raise RuntimeError("SAM 3.1 requires a CUDA GPU")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.resolution = int(self.params.get("image_size", 1008))
        # predict() runs on a thread pool; SAM state is not thread-safe.
        self._lock = threading.RLock()
        self._image_state: Optional[Tuple[str, Dict[str, Any]]] = None
        self._predictor = None
        self._sessions: "OrderedDict[str, str]" = OrderedDict()
        self._sessions_with_object = set()

        started = time.monotonic()
        detector_mode = self.params.get("detector", "shared")
        if detector_mode == "shared":
            # One model in GPU memory: text/exemplar run on the multiplex
            # predictor's own detector, clicks on its tracker.
            self.model = self._interactive_predictor().model.detector
            load_report = {"detector": "shared multiplex detector"}
        elif detector_mode == "image_model":
            self.model, load_report = self._build_image_detector(builder)
        else:
            raise ValueError(f"unknown detector mode: {detector_mode!r}")
        self.processor = Sam3Processor(
            self.model,
            resolution=self.resolution,
            device="cuda",
            confidence_threshold=float(self.params.get("conf_threshold", 0.5)),
        )
        logger.info(
            f"SAM 3.1 loaded in {time.monotonic() - started:.1f}s: "
            f"{load_report}"
        )

    def _build_image_detector(self, builder):
        """Separate SAM 3 image model carrying the SAM 3.1 detector weights.

        Matches the batch pipeline's calibration but keeps a second copy of
        the backbone on the GPU (~4 GB more than ``shared``).
        """
        torch = self.torch
        load_report = {}

        def checked_load(model, path):
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(ckpt.get("model"), dict):
                ckpt = ckpt["model"]
            state = {
                key[len("detector.") :]: value
                for key, value in ckpt.items()
                if key.startswith("detector.")
            }
            if not state:
                raise RuntimeError("checkpoint has no detector weights")
            load_report.update(load_detector_weights(model, state))

        original_loader = builder._load_checkpoint
        builder._load_checkpoint = checked_load
        try:
            model = builder.build_sam3_image_model(
                checkpoint_path=self.checkpoint,
                load_from_HF=False,
                enable_segmentation=True,
                enable_inst_interactivity=False,
            )
        finally:
            builder._load_checkpoint = original_loader
        return model, load_report

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        tasks = TASKS
        if not self.params.get("interactive", True):
            tasks = [t for t in TASKS if t["id"] != TASK_INTERACTIVE]
        metadata["available_tasks"] = tasks
        return metadata

    # -- shared helpers -------------------------------------------------

    def _settings(self, params: Dict[str, Any]) -> Tuple[float, float, bool]:
        conf = float(params.get("conf_threshold") or 0)
        if conf <= 0:
            conf = float(self.params.get("conf_threshold", 0.5))
        epsilon = params.get("epsilon_factor")
        if epsilon is None:
            epsilon = self.params.get("epsilon_factor", 0.001)
        show_boxes = bool(
            params.get("show_boxes", self.params.get("show_boxes", False))
        )
        return conf, float(epsilon), show_boxes

    def _autocast(self):
        return self.torch.autocast("cuda", dtype=self.torch.bfloat16)

    def _image_inference_state(self, image: np.ndarray):
        key = image_key(image)
        if self._image_state is None or self._image_state[0] != key:
            pil_image = Image.fromarray(image[:, :, ::-1])
            self._image_state = (key, self.processor.set_image(pil_image))
        return self._image_state[1]

    def _shapes(
        self,
        masks,
        boxes,
        scores,
        labels: List[str],
        epsilon: float,
        show_boxes: bool,
    ) -> List[Shape]:
        shapes = []
        masks = masks.float().cpu().numpy() if hasattr(masks, "cpu") else masks
        boxes = boxes.float().cpu().numpy() if hasattr(boxes, "cpu") else boxes
        scores = (
            scores.float().cpu().numpy() if hasattr(scores, "cpu") else scores
        )
        for index, label in enumerate(labels):
            score = float(min(max(scores[index], 0.0), 1.0))
            polygon = mask_to_polygon(masks[index], epsilon)
            if polygon:
                shapes.append(
                    Shape(
                        label=label,
                        shape_type="polygon",
                        points=polygon,
                        score=score,
                    )
                )
            if show_boxes:
                x1, y1, x2, y2 = (float(v) for v in boxes[index])
                shapes.append(
                    Shape(
                        label=label,
                        shape_type="rectangle",
                        points=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        score=score,
                    )
                )
        return shapes

    # -- tasks ----------------------------------------------------------

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        marks = params.get("marks") or []
        task = params.get("current_task")
        if task not in {TASK_TEXT, TASK_EXEMPLAR, TASK_INTERACTIVE}:
            task = TASK_INTERACTIVE if marks else TASK_TEXT
        with self._lock:
            if task == TASK_INTERACTIVE:
                if not self.params.get("interactive", True):
                    raise RuntimeError("interactive task is disabled")
                # Build outside inference_mode so weights load normally.
                self._interactive_predictor()
            with self.torch.inference_mode():
                if task == TASK_INTERACTIVE:
                    return self._predict_interactive(image, marks, params)
                if task == TASK_EXEMPLAR:
                    return self._predict_exemplar(image, marks, params)
                return self._predict_text(image, params)

    def _predict_text(self, image, params):
        prompts = parse_prompts(
            params.get("text_prompt", ""), separators=[",", ".", ";", "\n"]
        )
        if not prompts:
            return {"shapes": [], "description": ""}
        conf, epsilon, show_boxes = self._settings(params)
        shapes = []
        with self._autocast():
            state = self._image_inference_state(image)
            self.processor.set_confidence_threshold(conf)
            for prompt in prompts:
                self.processor.reset_all_prompts(state)
                out = self.processor.set_text_prompt(
                    prompt=prompt, state=state
                )
                count = len(out["scores"])
                if count:
                    shapes += self._shapes(
                        out["masks"],
                        out["boxes"],
                        out["scores"],
                        [prompt] * count,
                        epsilon,
                        show_boxes,
                    )
            self.processor.reset_all_prompts(state)
        logger.info(f"SAM 3.1 text {prompts}: {len(shapes)} shapes")
        return {"shapes": shapes, "description": ""}

    def _predict_exemplar(self, image, marks, params):
        _, _, boxes, box_labels = split_marks(marks)
        if not any(box_labels):
            return {"shapes": [], "description": ""}
        conf, epsilon, show_boxes = self._settings(params)
        height, width = image.shape[:2]
        with self._autocast():
            state = self._image_inference_state(image)
            self.processor.set_confidence_threshold(conf)
            self.processor.reset_all_prompts(state)
            for (x1, y1, x2, y2), label in zip(boxes, box_labels):
                box = [
                    (x1 + x2) / 2 / width,
                    (y1 + y2) / 2 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ]
                state = self.processor.add_geometric_prompt(
                    box=box, label=bool(label), state=state
                )
            count = len(state.get("scores", []))
            shapes = (
                self._shapes(
                    state["masks"],
                    state["boxes"],
                    state["scores"],
                    ["AUTOLABEL_OBJECT"] * count,
                    epsilon,
                    show_boxes,
                )
                if count
                else []
            )
            self.processor.reset_all_prompts(state)
        logger.info(f"SAM 3.1 exemplar ({len(boxes)} boxes): {count} objects")
        return {"shapes": shapes, "description": ""}

    def _interactive_predictor(self):
        if self._predictor is None:
            from sam3.model_builder import build_sam3_multiplex_video_predictor

            started = time.monotonic()
            self._predictor = build_sam3_multiplex_video_predictor(
                checkpoint_path=self.checkpoint,
                use_fa3=bool(self.params.get("use_fa3", False)),
                compile=False,
                async_loading_frames=False,
            )
            logger.info(
                "SAM 3.1 interactive tracker loaded in "
                f"{time.monotonic() - started:.1f}s"
            )
        return self._predictor

    def _session_for(self, image: np.ndarray) -> str:
        """One single-frame tracker session per image, LRU-cached."""
        predictor = self._interactive_predictor()
        key = image_key(image)
        session_id = self._sessions.get(key)
        if session_id in predictor._all_inference_states:
            self._sessions.move_to_end(key)
            return session_id
        self._sessions_with_object.discard(session_id)
        # Sam3BasePredictor.start_session passes offload_state_to_cpu, which
        # the multiplex model's init_state rejects; register it directly.
        state = predictor.model.init_state(
            resource_path=[Image.fromarray(image[:, :, ::-1])],
            offload_video_to_cpu=False,
            async_loading_frames=False,
        )
        session_id = str(uuid.uuid4())
        predictor._all_inference_states[session_id] = {
            "state": state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        self._sessions[key] = session_id
        while len(self._sessions) > int(self.params.get("cache_size", 4)):
            _, stale = self._sessions.popitem(last=False)
            self._sessions_with_object.discard(stale)
            predictor.handle_request(
                {"type": "close_session", "session_id": stale}
            )
        return session_id

    def _predict_interactive(self, image, marks, params):
        height, width = image.shape[:2]
        points, labels = interactive_prompt(marks, width, height)
        if not points:
            return {"shapes": [], "description": "", "replace": False}
        _, epsilon, _ = self._settings(params)
        predictor = self._interactive_predictor()
        session_id = self._session_for(image)
        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "points": points,
            "point_labels": labels,
            "obj_id": INTERACTIVE_OBJ_ID,
        }
        # Every request carries the full prompt of the current object, so
        # start from a clean object instead of refining the previous one.
        if session_id in self._sessions_with_object:
            predictor.handle_request(
                {
                    "type": "remove_object",
                    "session_id": session_id,
                    "obj_id": INTERACTIVE_OBJ_ID,
                }
            )
            self._sessions_with_object.discard(session_id)
        outputs = predictor.handle_request(request)["outputs"]
        self._sessions_with_object.add(session_id)
        if not len(outputs.get("out_obj_ids", [])):
            # The multiplex tracker returns nothing when an object is first
            # added; refining with the same prompt yields its mask.
            outputs = predictor.handle_request(request)["outputs"]
        ids = [int(i) for i in outputs.get("out_obj_ids", [])]
        if INTERACTIVE_OBJ_ID not in ids:
            return {"shapes": [], "description": "", "replace": False}
        index = ids.index(INTERACTIVE_OBJ_ID)
        mask = np.asarray(outputs["out_binary_masks"][index]).squeeze()
        if mask.shape != (height, width):
            import cv2

            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        polygon = mask_to_polygon(mask, epsilon)
        shapes = []
        if polygon:
            score = float(outputs["out_probs"][index])
            shapes.append(
                Shape(
                    label="AUTOLABEL_OBJECT",
                    shape_type="polygon",
                    points=polygon,
                    score=min(max(score, 0.0), 1.0),
                )
            )
        return {"shapes": shapes, "description": "", "replace": False}

    def unload(self):
        with getattr(self, "_lock", threading.RLock()):
            if self._predictor is not None:
                for session_id in list(self._sessions.values()):
                    try:
                        self._predictor.handle_request(
                            {"type": "close_session", "session_id": session_id}
                        )
                    except Exception:  # noqa: BLE001 - best effort
                        pass
            self._sessions.clear()
            self._sessions_with_object.clear()
            self._predictor = None
            self._image_state = None
            self.model = None
            self.processor = None
        logger.info("SAM 3.1 unloaded")
