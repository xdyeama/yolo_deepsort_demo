"""
Visualization utilities for drawing tracking results on frames.
Supports bounding boxes, track IDs, class labels, confidence scores, and trails.
"""
import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple


# COCO class names (for YOLOv8)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_color_by_id(track_id: int) -> Tuple[int, int, int]:
    """
    Generate consistent color for each track ID using golden ratio method.
    
    Args:
        track_id: Unique track identifier
        
    Returns:
        BGR color tuple
    """
    track_id = int(track_id) if not isinstance(track_id, int) else track_id
    golden_ratio = 0.618033988749895
    hue = (track_id * golden_ratio) % 1.0
    
    # Convert HSV to RGB (OpenCV uses BGR)
    hue_int = int(hue * 179)  # OpenCV hue is 0-179
    hsv = np.uint8([[[hue_int, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def get_color_by_class(class_id: int) -> Tuple[int, int, int]:
    """
    Get consistent color for each class.
    
    Args:
        class_id: Object class identifier
        
    Returns:
        BGR color tuple
    """
    colors = [
        (255, 0, 0),     # Blue for person
        (0, 255, 0),     # Green for bicycle
        (0, 0, 255),     # Red for car
        (255, 255, 0),   # Cyan for motorcycle
        (255, 0, 255),   # Magenta for airplane
        (0, 255, 255),   # Yellow for bus
        (128, 0, 128),   # Purple for train
        (255, 165, 0),   # Orange for truck
    ]
    return colors[class_id % len(colors)]


class Visualizer:
    """
    Handles visualization of tracking results with trails support.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: Visualization config (from runtime.yaml under 'draw')
        """
        self.cfg = config or {}
        
        # Drawing parameters
        self.thickness = int(self.cfg.get('thickness', 2))
        self.font_scale = float(self.cfg.get('font_scale', 0.5))
        self.show_conf = bool(self.cfg.get('show_conf', True))
        self.show_class = bool(self.cfg.get('show_class', True))
        self.color_mode = self.cfg.get('color_mode', 'id')  # 'id', 'class', 'random'
        
        # Trail parameters
        self.trails_enabled = bool(self.cfg.get('trails', False))
        self.trail_length = int(self.cfg.get('trail_length', 32))
        self.trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.trail_length))
        
        # Font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        class_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Draw all tracks on the frame.
        
        Args:
            frame: Input frame (BGR)
            tracks: List of track dicts with 'track_id', 'bbox', 'confidence', 'class'
            class_names: Optional list of class names (defaults to COCO)
            
        Returns:
            Annotated frame
        """
        frame = frame.copy()
        class_names = class_names or COCO_CLASSES
        
        # Draw trails first (underneath boxes)
        if self.trails_enabled:
            self._draw_trails(frame)
        
        # Draw each track
        for track in tracks:
            track_id = track['track_id']
            print(f"Track id: {track_id}")
            bbox = track['bbox']  # [x1, y1, x2, y2]
            conf = track.get('confidence', 0.0)
            cls = track.get('class', 0)
            
            # Get color based on mode
            if self.color_mode == 'id':
                color = get_color_by_id(track_id)
            elif self.color_mode == 'class':
                color = get_color_by_class(cls)
            else:
                color = (0, 255, 0)  # Default green
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
            
            # Update trail with center point
            if self.trails_enabled:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                self.trails[track_id].append((center_x, center_y))
            
            # Prepare label text
            label_parts = [f"ID:{track_id}"]
            
            if self.show_class and cls < len(class_names):
                label_parts.append(class_names[cls])
            
            if self.show_conf:
                label_parts.append(f"{conf:.2f}")
            
            label = " ".join(label_parts)
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.thickness
            )
            
            # Position label above box, or below if too close to top
            if y1 > 30:
                label_y1 = y1 - label_h - baseline - 5
                label_y2 = y1
                text_y = y1 - baseline - 5
            else:
                label_y1 = y2
                label_y2 = y2 + label_h + baseline + 5
                text_y = y2 + label_h + 5
            
            # Draw filled rectangle for label background
            cv2.rectangle(
                frame,
                (x1, label_y1),
                (x1 + label_w + 10, label_y2),
                color,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1 + 5, text_y),
                self.font,
                self.font_scale,
                (255, 255, 255),  # White text
                self.thickness - 1 if self.thickness > 1 else 1,
                cv2.LINE_AA
            )
        
        return frame
    
    def _draw_trails(self, frame: np.ndarray):
        """
        Draw tracking trails for all active tracks.
        
        Args:
            frame: Frame to draw on (modified in-place)
        """
        for track_id, trail in self.trails.items():
            if len(trail) < 2:
                continue
            
            # Get color for this track
            if self.color_mode == 'id':
                color = get_color_by_id(track_id)
            else:
                color = (200, 200, 200)  # Gray for trails
            
            # Draw lines between consecutive points
            points = list(trail)
            for i in range(1, len(points)):
                # Fade trail: older points have lower alpha (thinner lines)
                alpha = i / len(points)
                thickness = max(1, int(self.thickness * alpha))
                
                cv2.line(
                    frame,
                    points[i - 1],
                    points[i],
                    color,
                    thickness,
                    cv2.LINE_AA
                )
    
    def reset_trails(self):
        """Clear all tracking trails."""
        self.trails.clear()
    
    def draw_info(
        self,
        frame: np.ndarray,
        fps: Optional[float] = None,
        frame_idx: Optional[int] = None,
        track_count: Optional[int] = None
    ) -> np.ndarray:
        """
        Draw runtime information on frame (FPS, frame count, track count).
        
        Args:
            frame: Input frame
            fps: Current FPS
            frame_idx: Current frame index
            track_count: Number of active tracks
            
        Returns:
            Frame with info overlay
        """
        frame = frame.copy()
        info_lines = []
        
        if fps is not None:
            info_lines.append(f"FPS: {fps:.1f}")
        
        if frame_idx is not None:
            info_lines.append(f"Frame: {frame_idx}")
        
        if track_count is not None:
            info_lines.append(f"Tracks: {track_count}")
        
        # Draw info in top-right corner
        y_offset = 30
        for i, line in enumerate(info_lines):
            text_size = cv2.getTextSize(line, self.font, self.font_scale, self.thickness)[0]
            x = frame.shape[1] - text_size[0] - 10
            y = y_offset + i * 30
            
            # Draw background
            cv2.rectangle(
                frame,
                (x - 5, y - text_size[1] - 5),
                (frame.shape[1] - 5, y + 5),
                (0, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                line,
                (x, y),
                self.font,
                self.font_scale,
                (0, 255, 0),
                self.thickness,
                cv2.LINE_AA
            )
        
        return frame


def draw_detections(
    frame: np.ndarray,
    detections: List[List[float]],
    class_names: Optional[List[str]] = None,
    thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Draw raw detections (without tracking) on frame.
    
    Args:
        frame: Input frame (BGR)
        detections: List of detections [x1, y1, x2, y2, conf, class_id]
        class_names: Optional class names
        thickness: Box line thickness
        font_scale: Font scale for labels
        
    Returns:
        Annotated frame
    """
    frame = frame.copy()
    class_names = class_names or COCO_CLASSES
    
    for det in detections:
        x1, y1, x2, y2, conf = det[:5]
        cls = int(det[5]) if len(det) > 5 else 0
        
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Use class color
        color = get_color_by_class(cls)
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        label = f"{class_name} {conf:.2f}"
        
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10 if y1 > 30 else y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
    
    return frame
