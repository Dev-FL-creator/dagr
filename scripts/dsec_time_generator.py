#!/usr/bin/env python3
"""
Unified script to generate correct start_end_time.yaml files for DSEC sequences
Combines functionality of fix_time_range.py and generate_start_end_time.py
"""
import h5py
import yaml
from pathlib import Path
import argparse
import numpy as np

# Try to import blosc compression support
try:
    import hdf5plugin
except ImportError:
    print("Warning: hdf5plugin not available, some HDF5 files might not be readable")


def generate_dsec_time_yaml(sequence_path, verbose=True):
    """
    Generate start_end_time.yaml from DSEC events.h5 file
    
    Args:
        sequence_path: Path to DSEC sequence directory
        verbose: Whether to print detailed information
        
    Returns:
        tuple: (start_time, end_time) or (None, None) if failed
    """
    sequence_path = Path(sequence_path)
    events_file = sequence_path / "events" / "left" / "events.h5"
    yaml_file = sequence_path / "start_end_time.yaml"
    
    if not events_file.exists():
        print(f" Events file not found: {events_file}")
        return None, None
    
    if verbose:
        print(f" Analyzing events file: {events_file}")
    
    try:
        with h5py.File(events_file, 'r') as f:
            if verbose:
                print(" HDF5 file datasets:")
                for key in f.keys():
                    shape_info = f[key].shape if hasattr(f[key], 'shape') else 'Group'
                    print(f"  {key}: {shape_info}")
            
            start_time, end_time = None, None
            
            # Method 1: Use ms_to_idx + t_offset (most reliable for DSEC)
            if 'ms_to_idx' in f and 't_offset' in f:
                ms_to_idx = f['ms_to_idx'][:]
                t_offset = f['t_offset'][()]
                
                # Find valid time range
                valid_mask = ms_to_idx >= 0
                if valid_mask.any():
                    first_valid_ms = np.where(valid_mask)[0][0]
                    last_valid_ms = np.where(valid_mask)[0][-1]
                    
                    # Apply offset and convert to microseconds
                    start_time = int(t_offset + first_valid_ms * 1000)
                    end_time = int(t_offset + last_valid_ms * 1000)
                    
                    if verbose:
                        print(f" Method 1 (ms_to_idx + t_offset):")
                        print(f"  Valid ms range: {first_valid_ms} to {last_valid_ms}")
                        print(f"  T_offset: {t_offset} μs")
                        print(f"  Final range: {start_time} to {end_time} μs")
            
            # Method 2: Direct timestamp reading (fallback)
            if start_time is None and 'events' in f and 't' in f['events']:
                timestamps = f['events']['t']
                t_offset = f['t_offset'][()] if 't_offset' in f else 0
                
                start_time = int(timestamps[0] + t_offset)
                end_time = int(timestamps[-1] + t_offset)
                
                if verbose:
                    print(f" Method 2 (direct timestamps):")
                    print(f"  Raw range: {timestamps[0]} to {timestamps[-1]} μs")
                    print(f"  T_offset: {t_offset} μs")
                    print(f"  Final range: {start_time} to {end_time} μs")
            
            # Method 3: Default values (last resort)
            if start_time is None:
                start_time, end_time = 0, 60000000
                print(f"  Using default time range: 0 to 60 seconds")
            
    except Exception as e:
        print(f" Error reading HDF5 file: {e}")
        print("  Using default time range: 0 to 60 seconds")
        start_time, end_time = 0, 60000000
    
    # Create and write YAML file
    yaml_content = {
        'start_time': int(start_time),
        'end_time': int(end_time)
    }
    
    try:
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        duration = (end_time - start_time) / 1000000
        print(f"  Generated: {yaml_file}")
        print(f"  Start time: {start_time} μs ({start_time/1000000:.2f} seconds)")
        print(f"  End time: {end_time} μs ({end_time/1000000:.2f} seconds)")
        print(f"  Duration: {duration:.2f} seconds")
        
        return start_time, end_time
        
    except Exception as e:
        print(f" Error writing YAML file: {e}")
        return None, None


def process_multiple_sequences(dataset_root, verbose=False):
    """
    Process all sequences in a dataset root directory
    
    Args:
        dataset_root: Path to DSEC dataset root (contains train/test folders)
        verbose: Whether to print detailed information
    """
    dataset_root = Path(dataset_root)
    processed = 0
    failed = 0
    
    for split in ["train", "test"]:
        split_dir = dataset_root / split
        if split_dir.exists():
            print(f"\n Processing {split} sequences...")
            for sequence_dir in split_dir.iterdir():
                if sequence_dir.is_dir():
                    print(f"\n Processing sequence: {sequence_dir.name}")
                    result = generate_dsec_time_yaml(sequence_dir, verbose)
                    if result[0] is not None:
                        processed += 1
                    else:
                        failed += 1
    
    print(f"\n Summary: {processed} sequences processed, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate start_end_time.yaml for DSEC sequences")
    parser.add_argument("--sequence_path", type=str, 
                       help="Path to single DSEC sequence directory")
    parser.add_argument("--dataset_root", type=str,
                       help="Path to DSEC dataset root (will process all sequences)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print detailed information")
    
    args = parser.parse_args()
    
    if args.dataset_root:
        process_multiple_sequences(args.dataset_root, args.verbose)
    elif args.sequence_path:
        start_time, end_time = generate_dsec_time_yaml(args.sequence_path, args.verbose)
        if start_time is not None:
            print("\n Successfully generated start_end_time.yaml")
            print("Now you can run the visualization script!")
        else:
            print("\n Failed to generate start_end_time.yaml")
    else:
        print(" Please provide either --sequence_path or --dataset_root")
        parser.print_help()