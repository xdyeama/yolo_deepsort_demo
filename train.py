"""
Training script for YOLO12s on custom dataset.
Trains the model and saves weights to weights/ directory.
"""
import argparse
import sys
from pathlib import Path

from src.utils.config import load_config
from src.utils.devices import configure_torch_environment
from src.models.yolo_detector import YOLODetector


def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(
        description='Train YOLO12s on custom dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to dataset YAML file (e.g., data/custom_dataset.yaml)'
    )
    
    # Model
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='yolo12s.pt',
        help='Model to use for training (yolo12s.pt, yolo11s.pt, yolo8s.pt)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Image size for training'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu', 'mps'],
        help='Device to use for training'
    )
    
    # Output
    parser.add_argument(
        '--name',
        type=str,
        default='custom_model',
        help='Name for this training run'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='yolo_training',
        help='Project directory for training runs'
    )
    parser.add_argument(
        '--weights-dir',
        type=str,
        default='weights',
        help='Directory to save final weights'
    )
    
    # Training options
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='Use pretrained weights (transfer learning)'
    )
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Train from scratch (no pretrained weights)'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='auto',
        choices=['SGD', 'Adam', 'AdamW', 'auto'],
        help='Optimizer to use'
    )
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (epochs without improvement)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of dataloader workers'
    )
    
    # Validation
    parser.add_argument(
        '--val',
        action='store_true',
        help='Run validation after training'
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default='configs/detector.yaml',
        help='Path to detector config file'
    )
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("YOLO Custom Training")
    print("="*60)
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config(
        default_yaml_path='configs/default.yaml',
        detector_yaml_path=args.config
    )
    
    # Get detector config
    detector_config = config.get('detector', {})
    
    # Override model if specified
    if args.model:
        detector_config['model'] = args.model
    
    # Configure environment
    print("Configuring environment...")
    if args.device != 'auto':
        detector_config['device'] = args.device
    
    device = configure_torch_environment(detector_config)
    print(f"Using device: {device}")
    
    # Initialize detector (without tracking for training)
    print(f"\nInitializing {args.model}...")
    detector = YOLODetector(detector_config, use_tracking=False)
    
    # Check if dataset YAML exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"\n❌ Error: Dataset YAML not found: {args.data}")
        print("\nCreate a dataset YAML file with the following format:")
        print("""
# dataset.yaml
path: /path/to/dataset  # dataset root dir
train: images/train     # train images (relative to 'path')
val: images/val         # val images (relative to 'path')
test: images/test       # test images (optional)

# Classes
names:
  0: person
  1: car
  2: bicycle
  # ... add your classes
        """)
        return 1
    
    # Determine pretrained setting
    pretrained = not args.no_pretrained if args.no_pretrained else args.pretrained
    
    # Build override dict from CLI arguments (only non-default values)
    overrides = {}
    if args.epochs != 100:
        overrides['epochs'] = args.epochs
    if args.batch != 16:
        overrides['batch_size'] = args.batch
    if args.img_size != 640:
        overrides['img_size'] = args.img_size
    if args.weights_dir != 'weights':
        overrides['save_dir'] = args.weights_dir
    if args.project != 'yolo_training':
        overrides['project'] = args.project
    if args.name != 'custom_model':
        overrides['name'] = args.name
    if args.optimizer != 'auto':
        overrides['optimizer'] = args.optimizer
    if args.lr0 != 0.01:
        overrides['lr0'] = args.lr0
    if args.patience != 50:
        overrides['patience'] = args.patience
    if args.workers != 8:
        overrides['workers'] = args.workers
    if not pretrained:
        overrides['pretrained'] = False
    
    # Start training
    print(f"\n{'='*60}")
    print("Starting Training...")
    print(f"Config-based training with CLI overrides")
    print(f"{'='*60}\n")
    
    try:
        # Train using config + overrides
        results = detector.train(
            data_yaml=str(data_path),
            **overrides
        )
        
        # Print results
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        
        if results['best_weights_path']:
            print(f"✓ Best weights: {results['best_weights_path']}")
        
        if results['last_weights_path']:
            print(f"✓ Last checkpoint: {results['last_weights_path']}")
        
        print(f"✓ Training directory: {results['training_dir']}")
        
        if results.get('metrics'):
            metrics = results['metrics']
            if metrics.get('mAP50'):
                print(f"\nFinal Metrics:")
                print(f"  mAP@0.5: {metrics['mAP50']:.4f}")
                if metrics.get('mAP50-95'):
                    print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
        
        # Run validation if requested
        if args.val and results['best_weights_path']:
            print(f"\n{'='*60}")
            print("Running Validation...")
            print(f"{'='*60}\n")
            
            # Load best weights for validation
            detector_config['model'] = results['best_weights_path']
            val_detector = YOLODetector(detector_config, use_tracking=False)
            
            # Use config-based validation
            val_results = val_detector.validate(data_yaml=str(data_path))
            
            print(f"\nValidation Results:")
            print(f"  mAP@0.5: {val_results['mAP50']:.4f}")
            print(f"  mAP@0.5:0.95: {val_results['mAP50-95']:.4f}")
            print(f"  Precision: {val_results['precision']:.4f}")
            print(f"  Recall: {val_results['recall']:.4f}")
        
        print(f"\n{'='*60}")
        print("Done!")
        print(f"{'='*60}\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n Training interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

