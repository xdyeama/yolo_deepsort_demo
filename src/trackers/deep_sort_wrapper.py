"""
DeepSORT tracker wrapper for deep-sort-realtime package.
Converts YOLOv8 detections to DeepSORT format and returns tracked objects.
"""
import numpy as np
from typing import Any, Dict, List, Optional, Union
from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortTracker:
    """
    Wrapper around deep-sort-realtime's DeepSort tracker.
    
    Expected detection format from YOLOv8: [x1, y1, x2, y2, conf, class_id]
    DeepSORT expects: ([x, y, w, h], confidence, class_id)
    """
    
    def __init__(self, tracker_config: Dict[str, Any]):
        """
        Initialize DeepSORT tracker with config from tracker.yaml.
        
        Args:
            tracker_config: Dictionary with DeepSORT parameters (flat structure)
        """
        self.cfg = tracker_config
        self.tracker = self._init_tracker()
        
    def _init_tracker(self) -> DeepSort:
        """Initialize DeepSORT with parameters from config."""
        return DeepSort(
            max_age=self.cfg.get('max_age', 30),
            n_init=self.cfg.get('n_init', 3),
            nn_budget=self.cfg.get('nn_budget', 100),
            max_cosine_distance=self.cfg.get('max_cosine_distance', 0.2),
            max_iou_distance=self.cfg.get('max_iou_distance', 0.7),
            nms_max_overlap=self.cfg.get('nms_max_overlap', 1.0),
            embedder=self.cfg.get('embedder', 'mobilenet'),
            embedder_model_name=self.cfg.get('embedder_model_name', None),
            embedder_wts=self.cfg.get('embedder_wts', None),
            polygon=self.cfg.get('polygon', False),
            today=None  # Let DeepSORT auto-set
        )
    
    def update(
        self, 
        detections: Union[List, np.ndarray], 
        frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections for current frame.
        
        Args:
            detections: List of detections in format [x1, y1, x2, y2, conf, class_id]
                       where (x1, y1) is top-left, (x2, y2) is bottom-right
            frame: Current frame (BGR format from OpenCV) used for re-identification
            
        Returns:
            List of confirmed tracks, each a dict with:
                - track_id: Unique track identifier
                - bbox: [x1, y1, x2, y2] in xyxy format
                - confidence: Detection confidence
                - class: Object class ID
                - state: Track state (e.g., 'Confirmed')
        """
        # Convert YOLOv8 detections (xyxy) to DeepSORT format (xywh)
        ds_detections = []
        
        for det in detections:
            if len(det) < 5:
                continue
            
            # Correct order: x1, y1, x2, y2, conf (NOT x1, x2, y1, y2)
            x1, y1, x2, y2, conf = det[:5]
            cls = int(det[5]) if len(det) > 5 else 0
            
            # Convert xyxy to xywh (top-left x, top-left y, width, height)
            bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
            # DeepSORT expects: (bbox, confidence, class)
            ds_detections.append((bbox_xywh, float(conf), cls))
        
        # Update tracker with detections and frame for re-ID
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)
        
        # Extract confirmed tracks
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            # Get bbox in ltrb (left, top, right, bottom) = xyxy format
            ltrb = track.to_ltrb()
            
            results.append({
                'track_id': track.track_id,
                'bbox': ltrb.tolist() if isinstance(ltrb, np.ndarray) else ltrb,
                'confidence': track.get_det_conf() if track.get_det_conf() else 0.0,
                'class': int(track.get_det_class()) if track.get_det_class() is not None else 0,
                'state': str(track.state)
            })
        
        return results
    
    def reset(self):
        """Reset tracker to initial state (clears all tracks)."""
        self.tracker = self._init_tracker()
        
    @property
    def active_track_count(self) -> int:
        """Return number of currently active tracks."""
        return len([t for t in self.tracker.tracks if t.is_confirmed()])
