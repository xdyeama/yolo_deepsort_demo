from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.utils.devices import (
	get_device,
	configure_torch_environment,
	should_use_amp,
	should_use_half,
	amp_autocast_kwargs,
)


Detection = List[float]  # [x1, y1, x2, y2, conf, class_id]
Track = Dict[str, Any]  # Track dict with track_id, bbox, confidence, class


class YOLODetector:
	def __init__(self, detector_config: Dict[str, Any], use_tracking: bool = True):
		"""
		YOLOv8/YOLOv11/YOLOv12 Detector with optional built-in tracking.
		
		Args:
			detector_config: configs/detector.yaml mapping
			use_tracking: If True, uses YOLO's built-in tracking (BoT-SORT/ByteTrack)
		"""
		self.cfg = detector_config or {}
		self.use_tracking = use_tracking

		# Resolve and configure device/runtime
		self.device = configure_torch_environment(self.cfg)

		# Load model (supports yolov8, yolov11, yolov12)
		model_path = self.cfg.get("model", "yolov8n.pt")
		self.model = YOLO(model_path)
		self.model.to(self.device)
		self.model.model.eval()

		# Precision controls
		self.use_half = should_use_half(self.cfg, self.cfg, self.device)
		if self.use_half:
			try:
				self.model.model.half()
			except Exception:
				self.use_half = False

		self.use_amp = should_use_amp(self.cfg, self.cfg, self.device)

		# Cache some params
		self.img_size = int(self.cfg.get("img_size", 640))
		self.conf_thr = float(self.cfg.get("conf_threshold", 0.25))
		# detector.yaml uses iou_threshold; align to ultralytics kw 'iou'
		self.iou_thr = float(self.cfg.get("iou_threshold", self.cfg.get("iou_nms_threshold", 0.45)))
		self.max_det = int(self.cfg.get("max_det", 300))
		self.class_allowlist: Optional[Sequence[int]] = self.cfg.get("classes")
		self.agnostic_nms = bool(self.cfg.get("agnostic_nms", False))
		
		# Tracking parameters (for built-in tracker)
		self.tracker_type = self.cfg.get("tracker_type", "bytetrack.yaml")  # or "botsort.yaml"
		self.persist = self.cfg.get("persist_tracks", True)  # Keep tracks between frames

		# # Warmup (optional)
		# warmup_frames = int(self.cfg.get("warmup_frames", 0) or 0)
		# if warmup_frames > 0 and self.device.type != "cpu":
		# 	dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
		# 	for _ in range(warmup_frames):
		# 		_ = self._infer([dummy])

	def _preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
		"""
		Optionally convert BGR->RGB; resizing/letterbox handled by Ultralytics with imgsz.
		We keep raw arrays and delegate to YOLO for robust preprocessing.
		"""
		pre = self.cfg.get("preprocess", {}) or {}
		rgb = bool(pre.get("rgb", True))
		if rgb:
			return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		return frame_bgr

	def _infer(self, frames_rgb: List[np.ndarray], track: bool = False):
		"""
		Run inference via Ultralytics. Accepts list of RGB frames (np arrays).
		Returns Ultralytics Results list.
		
		Args:
			frames_rgb: List of RGB frames
			track: If True and self.use_tracking=True, runs tracking instead of detection
		"""
		kwargs = dict(
			imgsz=self.img_size,
			conf=self.conf_thr,
			iou=self.iou_thr,
			agnostic_nms=self.agnostic_nms,
			device=str(self.device),
			verbose=False,
			max_det=self.max_det,
			classes=self.class_allowlist if self.class_allowlist is not None else None,
		)
		
		# Add tracking parameters if tracking mode
		if track and self.use_tracking:
			kwargs['tracker'] = self.tracker_type
			kwargs['persist'] = self.persist

		if self.use_amp:
			amp_kwargs = amp_autocast_kwargs(self.device)
			with torch.autocast(**amp_kwargs):
				if track and self.use_tracking:
					results = self.model.track(frames_rgb, **kwargs)
				else:
					results = self.model(frames_rgb, **kwargs)
		else:
			if track and self.use_tracking:
				results = self.model.track(frames_rgb, **kwargs)
			else:
				results = self.model(frames_rgb, **kwargs)

		return results

	def _postprocess_results(self, results, min_box_area: Optional[float] = None) -> List[List[Detection]]:
		"""
		Extract detections in [x1,y1,x2,y2,conf,cls] per frame.
		"""
		all_dets: List[List[Detection]] = []
		for res in results:
			frame_dets: List[Detection] = []
			if not hasattr(res, "boxes") or res.boxes is None:
				all_dets.append(frame_dets)
				continue
			# xyxy, conf, cls
			xyxy = res.boxes.xyxy.detach().cpu().numpy() if hasattr(res.boxes, "xyxy") else None
			conf = res.boxes.conf.detach().cpu().numpy() if hasattr(res.boxes, "conf") else None
			cls = res.boxes.cls.detach().cpu().numpy() if hasattr(res.boxes, "cls") else None
			if xyxy is None or conf is None or cls is None:
				all_dets.append(frame_dets)
				continue
			for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
				if min_box_area:
					if max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) < float(min_box_area):
						continue
				frame_dets.append([float(x1), float(y1), float(x2), float(y2), float(c), float(k)])
			all_dets.append(frame_dets)
		return all_dets

	def _postprocess_tracks(self, results, min_box_area: Optional[float] = None) -> List[List[Track]]:
		"""
		Extract tracks from tracking results in DeepSORT-compatible format.
		Returns list of track dicts per frame with format:
		{
			'track_id': int,
			'bbox': [x1, y1, x2, y2],
			'confidence': float,
			'class': int,
			'state': 'Confirmed'
		}
		"""
		all_tracks: List[List[Track]] = []
		for res in results:
			frame_tracks: List[Track] = []
			
			# Check if tracking results exist
			if not hasattr(res, "boxes") or res.boxes is None:
				all_tracks.append(frame_tracks)
				continue
			
			# Extract tracking data
			xyxy = res.boxes.xyxy.detach().cpu().numpy() if hasattr(res.boxes, "xyxy") else None
			conf = res.boxes.conf.detach().cpu().numpy() if hasattr(res.boxes, "conf") else None
			cls = res.boxes.cls.detach().cpu().numpy() if hasattr(res.boxes, "cls") else None
			
			# Extract track IDs (only present in tracking mode)
			track_ids = res.boxes.id.detach().cpu().numpy() if hasattr(res.boxes, "id") and res.boxes.id is not None else None
			
			if xyxy is None or conf is None or cls is None:
				all_tracks.append(frame_tracks)
				continue
			
			# If no track IDs, assign sequential IDs (fallback for detection mode)
			if track_ids is None:
				track_ids = np.arange(len(xyxy))
			
			for idx, ((x1, y1, x2, y2), c, k, tid) in enumerate(zip(xyxy, conf, cls, track_ids)):
				# Filter by box area if specified
				if min_box_area:
					if max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) < float(min_box_area):
						continue
				
				frame_tracks.append({
					'track_id': int(tid),
					'bbox': [float(x1), float(y1), float(x2), float(y2)],
					'confidence': float(c),
					'class': int(k),
					'state': 'Confirmed'  # YOLO tracks are always confirmed
				})
			
			all_tracks.append(frame_tracks)
		return all_tracks

	def predict_frame(self, frame_bgr: np.ndarray, min_box_area: Optional[float] = None) -> List[Detection]:
		"""
		Run detection on a single BGR frame and return detections:
		[x1, y1, x2, y2, conf, class_id]
		"""
		prepared = self._preprocess_frame(frame_bgr)
		results = self._infer([prepared], track=False)
		all_dets = self._postprocess_results(results, min_box_area=min_box_area)
		return all_dets[0] if all_dets else []

	def predict_batch(self, frames_bgr: List[np.ndarray], min_box_area: Optional[float] = None) -> List[List[Detection]]:
		"""
		Run detection on a batch of BGR frames.
		"""
		prepared = [self._preprocess_frame(f) for f in frames_bgr]
		results = self._infer(prepared, track=False)
		return self._postprocess_results(results, min_box_area=min_box_area)
	
	def track_frame(self, frame_bgr: np.ndarray, min_box_area: Optional[float] = None) -> List[Track]:
		"""
		Run tracking on a single BGR frame and return tracks in DeepSORT-compatible format.
		Returns list of track dicts: [{'track_id': int, 'bbox': [x1,y1,x2,y2], 'confidence': float, 'class': int, 'state': str}]
		"""
		if not self.use_tracking:
			raise ValueError("Tracking mode is not enabled. Set use_tracking=True in constructor.")
		
		prepared = self._preprocess_frame(frame_bgr)
		results = self._infer([prepared], track=True)
		all_tracks = self._postprocess_tracks(results, min_box_area=min_box_area)
		return all_tracks[0] if all_tracks else []
	
	def track_batch(self, frames_bgr: List[np.ndarray], min_box_area: Optional[float] = None) -> List[List[Track]]:
		"""
		Run tracking on a batch of BGR frames.
		"""
		if not self.use_tracking:
			raise ValueError("Tracking mode is not enabled. Set use_tracking=True in constructor.")
		
		prepared = [self._preprocess_frame(f) for f in frames_bgr]
		results = self._infer(prepared, track=True)
		return self._postprocess_tracks(results, min_box_area=min_box_area)