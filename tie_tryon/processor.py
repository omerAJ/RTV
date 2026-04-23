from __future__ import annotations

import math
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .catalog import TieCatalog

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError as exc:  # pragma: no cover - exercised at runtime.
    raise RuntimeError(
        "MediaPipe is required for the tie try-on app. Install the slim requirements "
        "with `python -m pip install -r requirements.txt`."
    ) from exc


MODEL_URLS = {
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
}

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
FACE_NOSE_TIP = 1
FACE_CHIN = 152
FACE_JAW_LEFT = 172
FACE_JAW_RIGHT = 397
FACE_FACE_LEFT = 234
FACE_FACE_RIGHT = 454
LOWER_FACE_INDICES = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
    152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
]


@dataclass
class ManualAdjustment:
    offset_x: float = 0.0
    offset_y: float = 0.0
    scale_delta: float = 0.0
    rotation_deg: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class TransformEstimate:
    anchor: np.ndarray
    knot_width: float
    angle_deg: float
    confidence: float
    lower_face_polygon: np.ndarray


class TieTryOnProcessor:
    def __init__(
        self,
        catalog: TieCatalog,
        model_dir: str | Path = "models/mediapipe",
        max_infer_width: int = 640,
        hold_timeout_ms: int = 500,
    ) -> None:
        self.catalog = catalog
        self.model_dir = Path(model_dir).resolve()
        self.max_infer_width = max_infer_width
        self.hold_timeout_ms = hold_timeout_ms
        self.pose_model_path, self.face_model_path = self._ensure_models()
        self.pose_landmarker = self._create_pose_landmarker(self.pose_model_path)
        self.face_landmarker = self._create_face_landmarker(self.face_model_path)
        self.smoothed_transform: TransformEstimate | None = None
        self.last_good_timestamp_ms: int | None = None
        self.last_timestamp_ms = 0

    def close(self) -> None:
        self.pose_landmarker.close()
        self.face_landmarker.close()

    def process(
        self,
        frame_bgr: np.ndarray,
        selected_tie_id: str | None,
        manual_adjustment: ManualAdjustment,
        debug: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        output_frame = frame_bgr.copy()
        state: dict[str, Any] = {
            "selected_tie_id": selected_tie_id,
            "tracking_mode": "hidden",
            "tracking_confidence": 0.0,
            "manual_adjustment": manual_adjustment.to_dict(),
            "anchor": None,
            "knot_width": None,
            "angle_deg": None,
        }

        tie = self.catalog.get(selected_tie_id)
        if tie is None:
            return output_frame, state

        infer_frame, frame_scale = self._resize_for_inference(frame_bgr)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)),
        )
        timestamp_ms = self._next_timestamp_ms()
        pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        transform = self._estimate_transform(
            infer_frame,
            frame_scale,
            pose_result,
            face_result,
            tie,
            manual_adjustment,
        )
        tracking_mode = "hidden"
        if transform is not None:
            self.smoothed_transform = self._smooth_transform(transform, self.smoothed_transform)
            self.last_good_timestamp_ms = timestamp_ms
            tracking_mode = "tracked"
        elif self.smoothed_transform is not None and self.last_good_timestamp_ms is not None:
            if timestamp_ms - self.last_good_timestamp_ms <= self.hold_timeout_ms:
                tracking_mode = "held"
            else:
                self.smoothed_transform = None

        if self.smoothed_transform is None:
            return output_frame, state

        output_frame = self._composite_tie(
            output_frame,
            tie,
            self.smoothed_transform,
        )
        state.update(
            {
                "tracking_mode": tracking_mode,
                "tracking_confidence": float(self.smoothed_transform.confidence),
                "anchor": self.smoothed_transform.anchor.round(2).tolist(),
                "knot_width": round(float(self.smoothed_transform.knot_width), 2),
                "angle_deg": round(float(self.smoothed_transform.angle_deg), 2),
            }
        )
        if debug:
            output_frame = self._draw_debug(output_frame, self.smoothed_transform, tracking_mode)
        return output_frame, state

    def _ensure_models(self) -> tuple[Path, Path]:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_paths: dict[str, Path] = {}
        for filename, url in MODEL_URLS.items():
            target = self.model_dir / filename
            if not target.exists():
                self._download_file(url, target)
            model_paths[filename] = target
        return model_paths["pose_landmarker_lite.task"], model_paths["face_landmarker.task"]

    def _download_file(self, url: str, target: Path) -> None:
        partial = target.with_suffix(target.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=30) as response, partial.open("wb") as handle:
                handle.write(response.read())
            partial.replace(target)
        except Exception as exc:  # pragma: no cover - depends on network state.
            if partial.exists():
                partial.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download MediaPipe model from {url}") from exc

    def _create_pose_landmarker(self, model_path: Path) -> vision.PoseLandmarker:
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.45,
            min_pose_presence_confidence=0.45,
            min_tracking_confidence=0.45,
            output_segmentation_masks=False,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def _create_face_landmarker(self, model_path: Path) -> vision.FaceLandmarker:
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _next_timestamp_ms(self) -> int:
        now_ms = int(time.monotonic() * 1000)
        if now_ms <= self.last_timestamp_ms:
            now_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = now_ms
        return now_ms

    def _resize_for_inference(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = frame_bgr.shape[:2]
        if width <= self.max_infer_width:
            return frame_bgr, 1.0
        scale = width / float(self.max_infer_width)
        infer_height = int(round(height / scale))
        infer_frame = cv2.resize(frame_bgr, (self.max_infer_width, infer_height), interpolation=cv2.INTER_AREA)
        return infer_frame, scale

    def _estimate_transform(
        self,
        infer_frame: np.ndarray,
        frame_scale: float,
        pose_result: vision.PoseLandmarkerResult,
        face_result: vision.FaceLandmarkerResult,
        tie,
        manual_adjustment: ManualAdjustment,
    ) -> TransformEstimate | None:
        if not pose_result.nose_tippose_landmarks or not face_result.face_landmarks:
            return None

        pose_landmarks = pose_result.pose_landmarks[0]
        face_landmarks = face_result.face_landmarks[0]
        left_shoulder = self._pose_point(pose_landmarks, POSE_LEFT_SHOULDER, infer_frame.shape, frame_scale)
        right_shoulder = self._pose_point(pose_landmarks, POSE_RIGHT_SHOULDER, infer_frame.shape, frame_scale)
        shoulder_visibility = min(
            getattr(pose_landmarks[POSE_LEFT_SHOULDER], "visibility", 0.0),
            getattr(pose_landmarks[POSE_RIGHT_SHOULDER], "visibility", 0.0),
        )
        shoulder_width = float(np.linalg.norm(right_shoulder - left_shoulder))
        if shoulder_visibility < 0.45 or shoulder_width < 40:
            return None

        chin = self._face_point(face_landmarks, FACE_CHIN, infer_frame.shape, frame_scale)
        nose_tip = self._face_point(face_landmarks, FACE_NOSE_TIP, infer_frame.shape, frame_scale)
        jaw_left = self._face_point(face_landmarks, FACE_JAW_LEFT, infer_frame.shape, frame_scale)
        jaw_right = self._face_point(face_landmarks, FACE_JAW_RIGHT, infer_frame.shape, frame_scale)
        face_left = self._face_point(face_landmarks, FACE_FACE_LEFT, infer_frame.shape, frame_scale)
        face_right = self._face_point(face_landmarks, FACE_FACE_RIGHT, infer_frame.shape, frame_scale)
        jaw_width = float(np.linalg.norm(face_right - face_left))
        if jaw_width < 20:
            return None

        shoulder_mid = (left_shoulder + right_shoulder) * 0.5
        chin_base = chin * 0.65 + ((jaw_left + jaw_right) * 0.5) * 0.35
        knot_width = max(shoulder_width * 0.18, jaw_width * 0.40)
        knot_width *= tie.default_scale
        knot_width *= max(0.55, 1.0 + manual_adjustment.scale_delta)

        anchor = shoulder_mid * 0.72 + chin_base * 0.28
        anchor += np.array(
            [
                tie.default_offset_x * knot_width + manual_adjustment.offset_x,
                tie.default_offset_y * knot_width + manual_adjustment.offset_y,
            ],
            dtype=np.float32,
        )
        shoulder_angle_deg = math.degrees(
            math.atan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0])
        )
        # Mirror-view webcams can make the semantic left/right shoulder order appear reversed,
        # which turns a level shoulder line into ~180deg and flips the tie upward.
        if shoulder_angle_deg > 90.0:
            shoulder_angle_deg -= 180.0
        elif shoulder_angle_deg < -90.0:
            shoulder_angle_deg += 180.0
        face_center_x = (face_left[0] + face_right[0]) * 0.5
        yaw_bias = np.clip((nose_tip[0] - face_center_x) / max(jaw_width, 1.0), -0.25, 0.25)
        angle_deg = shoulder_angle_deg + yaw_bias * 12.0 + tie.default_rotation_deg + manual_adjustment.rotation_deg
        lower_face_polygon = self._lower_face_polygon(face_landmarks, infer_frame.shape, frame_scale)
        confidence = float(min(1.0, max(0.0, 0.5 * shoulder_visibility + 0.5)))
        return TransformEstimate(
            anchor=anchor.astype(np.float32),
            knot_width=float(knot_width),
            angle_deg=float(angle_deg),
            confidence=confidence,
            lower_face_polygon=lower_face_polygon,
        )

    def _lower_face_polygon(
        self,
        face_landmarks,
        infer_shape: tuple[int, int, int],
        frame_scale: float,
    ) -> np.ndarray:
        points = [
            self._face_point(face_landmarks, index, infer_shape, frame_scale)
            for index in LOWER_FACE_INDICES
        ]
        return np.round(np.asarray(points, dtype=np.float32)).astype(np.int32)

    def _smooth_transform(
        self,
        current: TransformEstimate,
        previous: TransformEstimate | None,
    ) -> TransformEstimate:
        if previous is None:
            return current

        alpha = 0.24 + 0.22 * current.confidence
        anchor = previous.anchor * (1.0 - alpha) + current.anchor * alpha
        knot_width = previous.knot_width * (1.0 - alpha) + current.knot_width * alpha
        angle_delta = ((current.angle_deg - previous.angle_deg + 180.0) % 360.0) - 180.0
        angle_deg = previous.angle_deg + angle_delta * alpha
        polygon = previous.lower_face_polygon.astype(np.float32) * (1.0 - alpha) + current.lower_face_polygon.astype(np.float32) * alpha
        confidence = previous.confidence * (1.0 - alpha) + current.confidence * alpha
        return TransformEstimate(
            anchor=anchor.astype(np.float32),
            knot_width=float(knot_width),
            angle_deg=float(angle_deg),
            confidence=float(confidence),
            lower_face_polygon=np.round(polygon).astype(np.int32),
        )

    def _composite_tie(
        self,
        frame_bgr: np.ndarray,
        tie,
        transform: TransformEstimate,
    ) -> np.ndarray:
        rotation = cv2.getRotationMatrix2D(
            tuple(tie.knot_anchor),
            transform.angle_deg,
            transform.knot_width / tie.knot_width_ref,
        )
        rotation[:, 2] += transform.anchor - np.asarray(tie.knot_anchor, dtype=np.float32)
        frame_h, frame_w = frame_bgr.shape[:2]
        warped_tie = cv2.warpAffine(
            tie.asset_bgra,
            rotation,
            (frame_w, frame_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        warped_top_mask = cv2.warpAffine(
            tie.top_mask,
            rotation,
            (frame_w, frame_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        face_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.fillPoly(face_mask, [transform.lower_face_polygon], 255)
        face_mask = cv2.GaussianBlur(face_mask, (11, 11), 0)
        alpha = warped_tie[:, :, 3]
        occluded_top = cv2.bitwise_and(face_mask, warped_top_mask)
        alpha = cv2.subtract(alpha, occluded_top)
        alpha_f = np.expand_dims(alpha.astype(np.float32) / 255.0, axis=2)
        target_bgr = warped_tie[:, :, :3].astype(np.float32)
        source_bgr = frame_bgr.astype(np.float32)
        composed = target_bgr * alpha_f + source_bgr * (1.0 - alpha_f)
        return composed.astype(np.uint8)

    def _draw_debug(
        self,
        frame_bgr: np.ndarray,
        transform: TransformEstimate,
        tracking_mode: str,
    ) -> np.ndarray:
        debug_frame = frame_bgr.copy()
        cv2.circle(debug_frame, tuple(np.round(transform.anchor).astype(int)), 6, (0, 255, 255), -1)
        cv2.polylines(debug_frame, [transform.lower_face_polygon], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.putText(
            debug_frame,
            f"{tracking_mode} {transform.knot_width:.1f}px {transform.angle_deg:.1f}deg",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return debug_frame

    def _pose_point(
        self,
        pose_landmarks,
        index: int,
        infer_shape: tuple[int, int, int],
        frame_scale: float,
    ) -> np.ndarray:
        landmark = pose_landmarks[index]
        infer_h, infer_w = infer_shape[:2]
        return np.array([landmark.x * infer_w, landmark.y * infer_h], dtype=np.float32) * frame_scale

    def _face_point(
        self,
        face_landmarks,
        index: int,
        infer_shape: tuple[int, int, int],
        frame_scale: float,
    ) -> np.ndarray:
        landmark = face_landmarks[index]
        infer_h, infer_w = infer_shape[:2]
        return np.array([landmark.x * infer_w, landmark.y * infer_h], dtype=np.float32) * frame_scale
