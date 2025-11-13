"""
Track serializer for saving tracking results to JSON/CSV formats.
Supports frame-centric storage with batch writing at end of processing.
"""
import json
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# Import COCO class names from visualizer
from src.io.visualizer import COCO_CLASSES


class TrackSerializer:
    """
    Serializes tracking results to JSON and CSV formats.
    
    Frame structure:
    {
        "frame": 0,
        "timestamp": 0.0,
        "track_count": 2,
        "tracks": [
            {
                "track_id": 1,
                "bbox": [x1, y1, x2, y2],
                "bbox_format": "xyxy",
                "confidence": 0.95,
                "class": 0,
                "class_name": "person",
                "state": "Confirmed"
            }
        ]
    }
    """
    
    def __init__(
        self,
        output_dir: str,
        save_json: bool = True,
        save_csv: bool = False,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize track serializer.

        Args:
            output_dir: Directory to save output files
            save_json: Whether to save JSON format
            save_csv: Whether to save CSV format
            class_names: List of class names (defaults to COCO)
        """
        self.output_dir = Path(output_dir)
        self.save_json = save_json
        self.save_csv = save_csv
        self.class_names = class_names or COCO_CLASSES
        self.frames: List[Dict[str, Any]] = []
        self.source_info: Dict[str, Any] = {}
    
    def add_frame(
        self, 
        frame_idx: int, 
        timestamp: float, 
        tracks: List[Dict[str, Any]]
    ):
        """
        Add a frame's tracking data to the buffer.

        Args:
            frame_idx: Index of the frame
            timestamp: Timestamp of the frame in seconds
            tracks: List of track dicts from DeepSORT wrapper
        """
        # Convert tracks to serializable format with class names
        serialized_tracks = []
        for track in tracks:
            class_id = track.get('class', 0)
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            
            serialized_tracks.append({
                "track_id": track['track_id'],
                "bbox": track['bbox'],  # [x1, y1, x2, y2]
                "bbox_format": "xyxy",
                "confidence": track.get('confidence', 0.0),
                "class": class_id,
                "class_name": class_name,
                "state": track.get('state', 'Unknown')
            })
        
        frame_data = {
            "frame": frame_idx,
            "timestamp": round(timestamp, 3),
            "track_count": len(serialized_tracks),
            "tracks": serialized_tracks
        }
        self.frames.append(frame_data)

    def _save_json(self, path: Optional[str] = None) -> str:
        """
        Save tracks to JSON file with frame-centric structure.

        Args:
            path: Optional custom path (defaults to output_dir/tracks.json)
            
        Returns:
            Path to saved file
        """
        if not self.frames:
            print("Warning: No frames to save")
            return ""
        
        json_path = Path(path) if path else self.output_dir / "tracks.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(json_path, 'w') as f:
                json.dump(self.frames, f, indent=2)
            print(f"Saved tracks to {json_path}")
            return str(json_path)
        except Exception as e:
            print(f"Error saving JSON file: {e}")
            return ""

    def save_csv(self, path: Optional[str] = None) -> str:
        """
        Save tracks to CSV file (flattened: one row per track per frame).

        Args:
            path: Optional custom path (defaults to output_dir/tracks.csv)
            
        Returns:
            Path to saved file
        """
        if not self.frames:
            print("Warning: No frames to save")
            return ""
        
        csv_path = Path(path) if path else self.output_dir / "tracks.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten frame/track structure
        rows = []
        for frame_data in self.frames:
            frame_idx = frame_data['frame']
            timestamp = frame_data['timestamp']
            
            for track in frame_data['tracks']:
                bbox = track['bbox']
                rows.append({
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'track_id': track['track_id'],
                    'x1': bbox[0],
                    'y1': bbox[1],
                    'x2': bbox[2],
                    'y2': bbox[3],
                    'width': bbox[2] - bbox[0],
                    'height': bbox[3] - bbox[1],
                    'confidence': track['confidence'],
                    'class': track['class'],
                    'class_name': track['class_name'],
                    'state': track['state']
                })
        
        if not rows:
            print("Warning: No tracks to save")
            return ""
        
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved tracks to {csv_path}")
            return str(csv_path)
        except Exception as e:
            print(f"Error saving CSV file: {e}")
            return ""

    def save_metadata(
        self, 
        config: Optional[Dict[str, Any]] = None,
        source_info: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None
    ) -> str:
        """
        Save session metadata to JSON file.

        Args:
            config: Optional config snapshot
            source_info: Optional source info dict (source, fps, frame_size, etc.)
            path: Optional custom path (defaults to output_dir/metadata.json)
            
        Returns:
            Path to saved file
        """
        metadata_path = Path(path) if path else self.output_dir / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Compute statistics
        unique_tracks = set()
        for frame in self.frames:
            for track in frame['tracks']:
                unique_tracks.add(track['track_id'])
        
        duration = self.frames[-1]['timestamp'] if self.frames else 0.0
        
        metadata = {
            "total_frames": len(self.frames),
            "duration_sec": round(duration, 3),
            "unique_track_count": len(unique_tracks),
        }
        
        # Add source info if provided
        if source_info:
            metadata.update(source_info)
        
        # Add config snapshot if provided
        if config:
            metadata["config_snapshot"] = config
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to {metadata_path}")
            return str(metadata_path)
        except Exception as e:
            print(f"Error saving metadata file: {e}")
            return ""

    def save_all(
        self,
        config: Optional[Dict[str, Any]] = None,
        source_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Save all enabled formats and metadata.

        Args:
            config: Optional config snapshot for metadata
            source_info: Optional source info for metadata
            
        Returns:
            Dict with paths to saved files
        """
        saved_paths = {}
        
        if self.save_json:
            path = self._save_json()
            if path:
                saved_paths['json'] = path
        
        if self.save_csv:
            path = self.save_csv()
            if path:
                saved_paths['csv'] = path
        
        # Always save metadata if we have frames
        if self.frames:
            path = self.save_metadata(config, source_info)
            if path:
                saved_paths['metadata'] = path
        
        return saved_paths
    
    def reset(self):
        """Clear accumulated frame data."""
        self.frames.clear()


def load_tracks_json(path: str) -> List[Dict[str, Any]]:
    """
    Load tracks from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        List of frame dicts
    """
    with open(path, 'r') as f:
        return json.load(f)


def load_tracks_csv(path: str) -> List[Dict[str, Any]]:
    """
    Load tracks from CSV file.
    
    Args:
        path: Path to CSV file
        
    Returns:
        List of row dicts
    """
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)