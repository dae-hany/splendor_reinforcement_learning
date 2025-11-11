"""
Analyze TensorBoard logs to understand training progress
"""

import os
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def analyze_tensorboard_logs(log_dir):
    """Analyze TensorBoard event files."""
    
    # Find event file
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not event_files:
        print(f"No event files found in {log_dir}")
        return
    
    event_file = os.path.join(log_dir, event_files[0])
    print(f"Analyzing: {event_file}")
    print("=" * 80)
    
    # Load event file
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    # Get available tags
    print("\nAvailable scalar tags:")
    scalar_tags = ea.Tags()['scalars']
    for tag in scalar_tags:
        print(f"  - {tag}")
    
    # Analyze key metrics
    print("\n" + "=" * 80)
    print("TRAINING METRICS SUMMARY")
    print("=" * 80)
    
    metrics_to_analyze = [
        'charts/episodic_return',
        'charts/episodic_length',
        'losses/td_loss',
        'losses/q_values',
        'charts/SPS',
        'charts/epsilon'
    ]
    
    for tag in metrics_to_analyze:
        if tag in scalar_tags:
            events = ea.Scalars(tag)
            values = [e.value for e in events]
            steps = [e.step for e in events]
            
            if len(values) > 0:
                print(f"\n{tag}:")
                print(f"  Total data points: {len(values)}")
                print(f"  Steps range: {min(steps)} - {max(steps)}")
                print(f"  Mean: {np.mean(values):.4f}")
                print(f"  Std: {np.std(values):.4f}")
                print(f"  Min: {np.min(values):.4f}")
                print(f"  Max: {np.max(values):.4f}")
                
                # Show first and last few values
                print(f"  First 5 values: {[f'{v:.2f}' for v in values[:5]]}")
                print(f"  Last 5 values: {[f'{v:.2f}' for v in values[-5:]]}")
                
                # Check for improvement
                if len(values) >= 10:
                    early_mean = np.mean(values[:len(values)//3])
                    late_mean = np.mean(values[-len(values)//3:])
                    change = late_mean - early_mean
                    change_pct = (change / (abs(early_mean) + 1e-8)) * 100
                    print(f"  Early mean (first 1/3): {early_mean:.4f}")
                    print(f"  Late mean (last 1/3): {late_mean:.4f}")
                    print(f"  Change: {change:.4f} ({change_pct:+.1f}%)")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = r"runs\SplendorSolo-v0__train_dqn_splendor__42__1762780970"
    
    analyze_tensorboard_logs(log_dir)
