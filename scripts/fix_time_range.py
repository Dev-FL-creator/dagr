#!/usr/bin/env python3
"""
Script to find the correct time offset for DSEC events and fix start_end_time.yaml
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


def find_correct_time_range(sequence_path):
    """
    Find the correct time range by reading ms_to_idx mapping
    
    Args:
        sequence_path: Path to DSEC sequence directory
    """
    sequence_path = Path(sequence_path)
    events_file = sequence_path / "events" / "left" / "events.h5"
    yaml_file = sequence_path / "start_end_time.yaml"
    
    if not events_file.exists():
        print(f"Events file not found: {events_file}")
        return None, None
    
    print(f"Analyzing events file: {events_file}")
    
    try:
        with h5py.File(events_file, 'r') as f:
            print("HDF5 file datasets:")
            for key in f.keys():
                print(f"  {key}: {f[key].shape if hasattr(f[key], 'shape') else 'Group'}")
            
            # Try to read ms_to_idx to find valid time range
            if 'ms_to_idx' in f:
                ms_to_idx = f['ms_to_idx'][:]
                print(f"ms_to_idx array length: {len(ms_to_idx)}")
                print(f"ms_to_idx range: {ms_to_idx.min()} to {ms_to_idx.max()}")
                
                # Find first and last valid indices (non-negative values)
                valid_mask = ms_to_idx >= 0
                if valid_mask.any():
                    first_valid_ms = np.where(valid_mask)[0][0]
                    last_valid_ms = np.where(valid_mask)[0][-1]
                    
                    # Convert milliseconds to microseconds
                    start_time = int(first_valid_ms * 1000)
                    end_time = int(last_valid_ms * 1000)
                    
                    print(f"Valid time range found:")
                    print(f"  First valid ms: {first_valid_ms}")
                    print(f"  Last valid ms: {last_valid_ms}")
                    print(f"  Start time: {start_time} μs ({start_time/1000000:.2f} seconds)")
                    print(f"  End time: {end_time} μs ({end_time/1000000:.2f} seconds)")
                    print(f"  Duration: {(end_time - start_time)/1000000:.2f} seconds")
                    
                else:
                    print("No valid indices found in ms_to_idx")
                    return None, None
            
            # Also check t_offset if available
            if 't_offset' in f:
                t_offset = f['t_offset'][()]
                print(f"t_offset: {t_offset}")
                
                # Adjust start_time with offset
                if 'ms_to_idx' in f and valid_mask.any():
                    start_time = int(t_offset + first_valid_ms * 1000)
                    end_time = int(t_offset + last_valid_ms * 1000)
                    
                    print(f"Adjusted with t_offset:")
                    print(f"  Start time: {start_time} μs ({start_time/1000000:.2f} seconds)")
                    print(f"  End time: {end_time} μs ({end_time/1000000:.2f} seconds)")
            
            return start_time, end_time
            
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        return None, None


def create_yaml_file(sequence_path, start_time, end_time):
    """Create start_end_time.yaml with correct time range"""
    yaml_file = Path(sequence_path) / "start_end_time.yaml"
    
    yaml_content = {
        'start_time': int(start_time),
        'end_time': int(end_time)
    }
    
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Generated: {yaml_file}")
    print(f"Content:")
    print(f"  start_time: {start_time}")
    print(f"  end_time: {end_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix DSEC start_end_time.yaml with correct time range")
    parser.add_argument("--sequence_path", type=str, required=True, 
                       help="Path to DSEC sequence directory")
    
    args = parser.parse_args()
    
    start_time, end_time = find_correct_time_range(args.sequence_path)
    
    if start_time is not None and end_time is not None:
        create_yaml_file(args.sequence_path, start_time, end_time)
        print("\n✅ Successfully generated correct start_end_time.yaml")
        print("Now you can run the visualization script again.")
    else:
        print("\n❌ Could not determine correct time range")
        print("Please check the events.h5 file structure.")