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
		model_path = self.cfg.get("model", "yolo12s.pt")
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
	
	# ==================== TRAINING METHODS ====================
	
	def train(self, data_yaml: Optional[str] = None, **overrides) -> Dict[str, Any]:
		"""
		Train YOLO model on custom dataset using parameters from detector.yaml config.
		Args:
			data_yaml: Path to dataset YAML file (overrides config if provided)
			**overrides: Override any training parameter from config

		Returns:
			Dict with training results
		"""
		from pathlib import Path
		
		# Get training config from detector.yaml
		train_cfg = self.cfg.get('training', {})
		
		# Data path: CLI arg > override > config
		data_path = data_yaml or overrides.get('data') or train_cfg.get('data')
		if not data_path:
			raise ValueError(
				"Dataset YAML path must be provided via 'data_yaml' argument or "
				"'training.data' in detector.yaml config"
			)
		
		# Map config keys to YOLO API parameter names
		param_mapping = {
			'batch_size': 'batch',
			'img_size': 'imgsz',
		}
		
		# Start with config parameters, apply mapping
		train_args = {'data': data_path, 'exist_ok': True}
		for key, value in train_cfg.items():
			if key != 'data':
				mapped_key = param_mapping.get(key, key)
				train_args[mapped_key] = value
		
		# Apply overrides with mapping
		for key, value in overrides.items():
			if key != 'data':
				mapped_key = param_mapping.get(key, key)
				train_args[mapped_key] = value
		
		print(f"\n{'='*60}")
		print(f"Starting YOLO Training")
		print(f"{'='*60}")
		print(f"Model: {self.cfg.get('model', 'yolo12s.pt')}")
		print(f"Dataset: {data_path}")
		print(f"Epochs: {train_args.get('epochs', 'N/A')}")
		print(f"Batch size: {train_args.get('batch', 'N/A')}")
		print(f"Image size: {train_args.get('imgsz', 'N/A')}")
		print(f"Device: {train_args.get('device', 'auto')}")
		print(f"Optimizer: {train_args.get('optimizer', 'auto')}")
		print(f"Learning rate: {train_args.get('lr0', 'N/A')}")
		print(f"Save directory: {train_args.get('project', 'yolo_training')}/{train_args.get('name', 'custom_model')}")
		print(f"{'='*60}\n")
		
		# Train the model
		results = self.model.train(**train_args)
		
		# Get paths to saved weights
		save_path = Path(train_args['project']) / train_args['name']
		best_weights = save_path / "weights" / "best.pt"
		last_weights = save_path / "weights" / "last.pt"
		
		# Copy best weights to configured save_dir
		save_dir = Path(train_cfg.get('save_dir', overrides.get('save_dir', 'weights')))
		save_dir.mkdir(parents=True, exist_ok=True)
		
		final_weights_path = save_dir / f"{train_args['name']}_best.pt"
		
		if best_weights.exists():
			import shutil
			shutil.copy2(best_weights, final_weights_path)
			print(f"\n✓ Best weights saved to: {final_weights_path}")
		
		return {
			'best_weights_path': str(final_weights_path) if final_weights_path.exists() else None,
			'last_weights_path': str(last_weights) if last_weights.exists() else None,
			'training_dir': str(save_path),
			'results': results,
			'metrics': {
				'mAP50': getattr(results, 'maps', [0])[0] if hasattr(results, 'maps') else None,
				'mAP50-95': getattr(results, 'map', 0) if hasattr(results, 'map') else None,
			}
		}
	
	def validate(self, data_yaml: Optional[str] = None, **overrides) -> Dict[str, Any]:
		"""
		Validate model on dataset using parameters from detector.yaml config.
		Args:
			data_yaml: Path to dataset YAML file (overrides config if provided)
			**overrides: Override any validation parameter from config
					
		Returns:
			Dict with validation metrics
		"""
		val_cfg = self.cfg.get('validation', {})
		
		data_path = data_yaml or overrides.get('data') or val_cfg.get('data') or self.cfg.get('training', {}).get('data')
		if not data_path:
			raise ValueError(
				"Dataset YAML path must be provided via 'data_yaml' argument or "
				"'validation.data' in detector.yaml config"
			)
		
		# Map config keys to YOLO API parameter names
		param_mapping = {
			'batch_size': 'batch',
			'img_size': 'imgsz',
		}
		
		# Start with config parameters, apply mapping
		val_args = {'data': data_path}
		for key, value in val_cfg.items():
			if key != 'data' and value is not None:
				mapped_key = param_mapping.get(key, key)
				val_args[mapped_key] = value
		
		# Apply overrides with mapping
		for key, value in overrides.items():
			if key != 'data' and value is not None:
				mapped_key = param_mapping.get(key, key)
				val_args[mapped_key] = value
		
		print(f"\n{'='*60}")
		print(f"Starting Validation")
		print(f"{'='*60}")
		print(f"Dataset: {data_path}")
		print(f"Split: {val_args['split']}")
		print(f"Batch size: {val_args['batch']}")
		print(f"Image size: {val_args['imgsz']}")
		print(f"{'='*60}\n")
		
		results = self.model.val(**val_args)
		
		return {
			'mAP50': results.box.map50 if hasattr(results, 'box') else None,
			'mAP50-95': results.box.map if hasattr(results, 'box') else None,
			'precision': results.box.mp if hasattr(results, 'box') else None,
			'recall': results.box.mr if hasattr(results, 'box') else None,
			'results': results
		}
	
	def export_model(self, output_path: Optional[str] = None, **overrides) -> str:
		"""
		Export model to different formats using parameters from detector.yaml config.
		Args:
			output_path: Custom output path (overrides auto path)
			**overrides: Override any export parameter from config
			
		Returns:
			Path to exported model
		"""
		# Get export config from detector.yaml
		export_cfg = self.cfg.get('export', {})
		
		# Start with config parameters (excluding None values)
		export_args = {}
		for key, value in export_cfg.items():
			if value is not None:
				export_args[key] = value
		
		# Apply overrides
		for key, value in overrides.items():
			if value is not None:
				export_args[key] = value
		
		# Get format for printing
		export_format = export_args.get('format', 'onnx')
		
		print(f"\n{'='*60}")
		print(f"Exporting Model to {export_format.upper()}")
		print(f"{'='*60}")
		print(f"Format: {export_format}")
		print(f"Half precision: {export_args.get('half', False)}")
		print(f"{'='*60}\n")
		
		exported_path = self.model.export(**export_args)
		
		if output_path and exported_path:
			import shutil
			from pathlib import Path
			Path(output_path).parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(exported_path, output_path)
			print(f"✓ Model exported to: {output_path}")
			return output_path
		
		print(f"✓ Model exported to: {exported_path}")
		return exported_path