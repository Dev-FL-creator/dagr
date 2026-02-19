#!/usr/bin/env python3
"""
Script to generate correct start_end_time.yaml files for DSEC sequences
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


def generate_start_end_time_yaml(sequence_path):
    """
    Generate start_end_time.yaml from events.h5 file or create default values
    
    Args:
        sequence_path: Path to DSEC sequence directory
    """
    sequence_path = Path(sequence_path)
    events_file = sequence_path / "events" / "left" / "events.h5"
    yaml_file = sequence_path / "start_end_time.yaml"
    
    if not events_file.exists():
        print(f"Events file not found: {events_file}")
        return
    
    print(f"Reading events from: {events_file}")
    
    try:
        # Try to read from HDF5 file
        with h5py.File(events_file, 'r') as f:
            print("HDF5 file structure:")
            def print_structure(name, obj):
                print(f"  {name}: {type(obj).__name__}")
            f.visititems(print_structure)
            
            # Try to read timestamps and offset
            if 'events' in f and 't' in f['events']:
                try:
                    timestamps = f['events']['t']
                    t_offset = f['t_offset'][()] if 't_offset' in f else 0
                    
                    # Get actual timestamp range with offset
                    start_time = int(timestamps[0] + t_offset)
                    end_time = int(timestamps[-1] + t_offset)
                    
                    print(f"Successfully read timestamps from events/t")
                    print(f"T_offset: {t_offset}")
                    print(f"Raw timestamps range: {timestamps[0]} to {timestamps[-1]}")
                    print(f"Adjusted timestamps range: {start_time} to {end_time}")
                except Exception as e:
                    print(f"Could not read timestamps: {e}")
                    raise
            else:
                raise ValueError("No suitable timestamp data found")
                
    except Exception as e:
        print(f"Could not read HDF5 file: {e}")
        print("Creating default time range...")
        
        # Create default time range (60 seconds)
        start_time = 0
        end_time = 60000000
    
    # Create YAML content
    yaml_content = {
        'start_time': start_time,
        'end_time': end_time
    }
    
    # Write YAML file
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Generated: {yaml_file}")
    print(f"  Start time: {start_time} μs ({start_time/1000000:.2f} seconds)")
    print(f"  End time: {end_time} μs ({end_time/1000000:.2f} seconds)")
    print(f"  Duration: {(end_time - start_time) / 1000000:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate start_end_time.yaml for DSEC sequences")
    parser.add_argument("--sequence_path", type=str, required=True, 
                       help="Path to DSEC sequence directory")
    
    args = parser.parse_args()
    generate_start_end_time_yaml(args.sequence_path)