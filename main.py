"""
Main CLI entry point.
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

from src.utils.config import load_config
from src.utils.devices import configure_torch_environment
from src.models.yolo_detector import YOLODetector
from src.pipeline.offline_loop import run_offline_tracking

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="YOLO Multi-Object Tracking Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    # I/O
    parser.add_argument(
        '--source', '-s',
        type=str,
        default='0',
        help='Source input (video file, image directory, or webcam index (0) or RTSP URL)'
    )
    # Output
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='data/outputs',
        help='Output directory for tracking results'
    )
    # Configs
    parser.add_argument(
        '--cfg',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--detector_cfg',
        type=str,
        default='configs/detector.yaml',
        help='Path to detector config file'
    )
    parser.add_argument(
        '--tracker_cfg',
        type=str,
        default='configs/tracker.yaml',
        help='Path to tracker config file'
    )
    parser.add_argument(
        '--runtime_cfg',
        type=str,
        default='configs/runtime.yaml',
        help='Path to runtime config file'
    )

    # Detector cfg overrides
    parser.add_argument(
        '--conf',
        type=float,
        help='Detection confidence threshold'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='Intersection over Union threshold for NMS'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=640,
        help='Inference image size'
    )
    parser.add_argument(
        '--classes',
        type=int,
        nargs='+',
        help='Filter by class IDs (e.g. 0 for person)'
    )

    # Runtime overrides
    parser.add_argument('--show', action='store_true', help='Display tracking window')
    parser.add_argument('--no-show', action='store_true', help='Disable display window')
    parser.add_argument('--save-video', action='store_true', help='Save output video')
    parser.add_argument('--no-save-video', action='store_true', help='Disable video saving')
    parser.add_argument('--save-tracks', action='store_true', help='Save tracks to JSON/CSV')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--stride', type=int, help='Process every Nth frame')
    
    # Device
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu', 'mps'], help='Device to use')
    parser.add_argument('--half', action='store_true', help='Use FP16 precision')
    
    return parser.parse_args()


def build_cli_overrides(args):
    """Build config overrides dict from CLI arguments."""
    overrides = {}
    
    # Source
    overrides['source'] = args.source
    
    # Detector overrides
    if args.conf is not None:
        overrides['detector.conf_threshold'] = args.conf
    if args.iou is not None:
        overrides['detector.iou_threshold'] = args.iou
    if args.img_size is not None:
        overrides['detector.img_size'] = args.img_size
    if args.classes is not None:
        overrides['detector.classes'] = args.classes
    
    # Runtime overrides
    if args.show:
        overrides['show'] = True
    if args.no_show:
        overrides['show'] = False
    if args.save_video:
        overrides['save_video'] = True
    if args.no_save_video:
        overrides['save_video'] = False
    if args.save_tracks:
        overrides['save_tracks_json'] = True
    if args.max_frames is not None:
        overrides['max_frames'] = args.max_frames
    if args.stride is not None:
        overrides['stride'] = args.stride
    
    # Device
    if args.device is not None:
        overrides['device'] = args.device
    if args.half:
        overrides['half'] = True
    
    return overrides

def create_run_directory(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pattern = config.get('paths', {}).get('run_dir_pattern', 'runs/track_{timestamp}')
    run_dir = pattern.format(timestamp=timestamp)
    
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")
    return run_dir


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load and merge configs
    print("Loading configuration...")
    cli_overrides = build_cli_overrides(args)
    
    config = load_config(
        default_yaml_path=args.cfg,
        detector_yaml_path=args.detector_cfg,
        tracker_yaml_path=args.tracker_cfg,
        runtime_yaml_path=args.runtime_cfg,
        cli_overrides=cli_overrides
    )
    
    # Create output directory
    output_dir = args.output_dir if args.output_dir else create_run_directory(config)

    # Configure torch environment (device, seeds, etc.)
    print("Configuring PyTorch environment...")
    device = configure_torch_environment(config)
    print(f"Using device: {device}")
    
    # Init detector with built-in tracking
    print("Loading YOLOv8/YOLOv11/YOLOv12 with built-in tracking...")
    detector_config = config.get('detector', {})
    detector = YOLODetector(detector_config, use_tracking=True)
    tracker_type = detector_config.get('tracker_type', 'bytetrack.yaml')
    print(f"✓ Detector loaded: {detector_config.get('model', 'yolo12s.pt')}")
    print(f"✓ Built-in tracker: {tracker_type.replace('.yaml', '')} (ByteTrack/BoT-SORT)")
    
    
    source = config.get('source', '0')
    print(f"\nSource: {source}")
    print("-" * 60)
    
    # tracking pipeline 
    print("Starting tracking pipeline...")
    stats = run_offline_tracking(
        source=source,
        config=config,
        detector=detector,
        output_dir=output_dir
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("TRACKING COMPLETE")
    print("=" * 60)
    print(f"Processed frames: {stats.get('processed_frames', 0)}")
    
    # Handle None values in stats
    total_time = stats.get('total_time') or 0.0
    avg_fps = stats.get('avg_fps') or 0.0
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    
    if stats.get('output_video'):
        print(f"\n✓ Output video: {stats['output_video']}")
    
    if stats.get('saved_paths'):
        print("\n✓ Saved files:")
        for key, path in stats['saved_paths'].items():
            print(f"  - {key}: {path}")
    
    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())