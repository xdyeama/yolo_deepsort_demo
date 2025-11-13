"""
Offline tracking loop for batch video processing.
Processes all frames sequentially with progress tracking and complete output saving.

Now uses YOLO's built-in tracking (ByteTrack/BoT-SORT) for real-time performance.
DeepSORT code is commented out but kept for reference.
"""
import cv2
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from src.io.video_reader import VideoReader
from src.io.video_writer import VideoWriter
from src.io.serializer import TrackSerializer
from src.io.visualizer import Visualizer, COCO_CLASSES
from src.models.yolov8_detector import YOLODetector
# COMMENTED OUT: DeepSORT is replaced by YOLO's built-in tracking
# from src.trackers.deep_sort_wrapper import DeepSortTracker

def apply_filters(
    detections: List[List[float]],
    filters: Dict[str, Any],
) -> List[List[float]]:
    """
    Apply pre-tracking filters to detections.
    
    Args:
        detections: List of [x1, y1, x2, y2, conf, class_id]
        filters: Filter config dict
        
    Returns:
        Filtered detections
    """
    if not filters:
        return detections
    
    filtered_detections = []
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det

        # Confidence filter
        if conf < filters.get('min_detector_confidence', 0.25):
            continue
        
        # Box area filter
        area = (x2 - x1) * (y2 - y1)
        if area < filters.get('min_box_area', 100.0):
            continue
        
        # Class allowlist filter
        allowlist = filters.get('class_allowlist')
        if allowlist is not None and int(class_id) not in allowlist:
            continue
        
        filtered_detections.append(det)
    
    return filtered_detections


def run_offline_tracking(
    source: str,
    config: Dict[str, Any],
    detector: YOLOv8Detector,
    tracker: Optional[Any] = None,  # DEPRECATED: tracker parameter kept for backward compatibility
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Run offline tracking on video file or image directory.
    
    Now uses YOLO's built-in tracking (ByteTrack/BoT-SORT) instead of DeepSORT for better performance.
    
    Args:
        source: Path to video file or image directory
        config: Merged config dict (runtime + detector + tracker)
        detector: Initialized YOLOv8Detector instance with tracking enabled
        tracker: DEPRECATED - No longer used. YOLO's built-in tracker is used instead.
        output_dir: Directory for outputs (run_dir)
        
    Returns:
        Stats dict with processed_frames, total_time, avg_fps, saved_paths
    
    Note:
        The 'tracker' parameter is kept for backward compatibility but is no longer used.
        To use DeepSORT instead, uncomment the DeepSORT code blocks below.
    """
    try:
        reader = VideoReader(source)
        visualizer = Visualizer(config.get('draw', {}))
        
        # Initialize writer if saving video
        writer = None
        if config.get('save_video', True):
            writer = VideoWriter(
                output_path=str(Path(output_dir) / 'output.mp4'),
                fps=reader.fps or 30.0,
                frame_size=reader.frame_size,
                fourcc=config.get('video_writer', {}).get('fourcc', 'mp4v'),
                enabled=True
            )
        
        # Initialize serializer if saving tracks
        serializer = None
        if config.get('save_tracks_json') or config.get('save_tracks_csv'):
            serializer = TrackSerializer(
                output_dir=output_dir,
                save_json=config.get('save_tracks_json', True),
                save_csv=config.get('save_tracks_csv', False),
                class_names=config.get('class_names', COCO_CLASSES)
            )

        total_frames = reader.total_frames if hasattr(reader, 'total_frames') else None
        progress_bar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
        
        try:
            for frame_data in reader.frames():
                frame_idx = frame_data['index']
                timestamp = frame_data['timestamp']
                frame = frame_data['frame']

                # Apply stride (skip frames)
                if frame_idx % config.get('stride', 1) != 0:
                    progress_bar.update(1)
                    continue

                # Skip first n frames if configured
                if frame_idx < config.get('skip_first_n', 0):
                    progress_bar.update(1)
                    continue

                # Stop at max_frames if configured
                max_frames = config.get('max_frames')
                if max_frames and frame_idx >= max_frames:
                    break

                # ===== NEW: YOLO Built-in Tracking (ByteTrack/BoT-SORT) =====
                # Direct tracking with YOLO - no separate detection step needed
                tracks = detector.track_frame(frame)
                
                # ===== OLD: DeepSORT Tracking (COMMENTED OUT) =====
                # # Detection
                # detections = detector.predict_frame(frame)
                # # Apply filters
                # detections = apply_filters(detections, config.get('filters', {}))
                # # Tracking with DeepSORT
                # tracks = tracker.update(detections, frame)
                # ==================================================
                
                # Apply filters to tracks (optional - can filter by confidence, class, etc.)
                # Note: YOLO tracking already applies confidence filtering during tracking
                if config.get('filters'):
                    # Convert tracks to detection format for filtering
                    track_dets = [[t['bbox'][0], t['bbox'][1], t['bbox'][2], t['bbox'][3], 
                                   t['confidence'], t['class']] for t in tracks]
                    filtered_dets = apply_filters(track_dets, config.get('filters', {}))
                    # Reconstruct tracks from filtered detections
                    filtered_track_ids = set(range(len(filtered_dets)))
                    tracks = [t for i, t in enumerate(tracks) if i in filtered_track_ids]

                # Visualization
                annotated = visualizer.draw_tracks(frame, tracks)

                # Optional: Add info overlay (FPS, frame count, track count)
                annotated = visualizer.draw_info(
                    annotated,
                    fps=progress_bar.format_dict.get('rate'),
                    frame_idx=frame_idx,
                    track_count=len(tracks)
                )

                # Write output video
                if writer:
                    writer.write(annotated)

                # Save tracks to serializer
                if serializer:
                    serializer.add_frame(frame_idx, timestamp, tracks)

                # Display window (if enabled and not headless)
                if config.get('show', False) and not config.get('headless', False):
                    cv2.imshow(config.get('window_name', 'Tracking'), annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(config.get('exit_key', 'q')):
                        print("\nUser interrupted")
                        break

                # Update progress bar
                progress_bar.update(1)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving partial results...")
        finally:
            progress_bar.close()
            reader.release()
            if writer:
                writer.release()
            if config.get('show', False):
                cv2.destroyAllWindows()

        # Save tracks
        saved_paths = {}
        if serializer:
            saved_paths = serializer.save_all(
                config=config,
                source_info={
                    'source': source,
                    'fps': reader.fps,
                    'frame_size': list(reader.frame_size) if reader.frame_size else None
                }
            )
        
        # Return statistics
        stats = {
            'processed_frames': progress_bar.n,
            'total_time': progress_bar.format_dict.get('elapsed', 0.0),
            'avg_fps': progress_bar.format_dict.get('rate', 0.0),
            'saved_paths': saved_paths,
            'output_video': str(writer.path) if writer and hasattr(writer, 'path') else None
        }
        return stats
    except FileNotFoundError:
        print(f"Error: Could not find source file: {source}")
        return {}
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return {}