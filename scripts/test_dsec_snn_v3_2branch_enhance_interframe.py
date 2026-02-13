import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

for k in ('HDF5_VOL_CONNECTOR', 'HDF5_PLUGIN_PATH'):
    os.environ.pop(k, None)

import hdf5plugin
import h5py

import torch
import wandb
import numpy as np
import tqdm
from torch_geometric.data import DataLoader
from torch.utils.data import Subset
from pprint import pprint

from dagr.utils.args import FLAGS
from dagr.data.dsec_data import DSEC
from dagr.data.augment import Augmentations
from dagr.model.networks.dagr_fusion_seperate_heads_v3_2branch_enhance import DAGR
from dagr.model.networks.ema import ModelEMA
from dagr.utils.logging import set_up_logging_directory, log_hparams
from dagr.utils.testing import run_test_with_visualization
import torch_geometric
import random


def to_npy(detections):
    """Convert detections to numpy format for saving"""
    n_boxes = len(detections['boxes'])
    dtype = np.dtype([('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('class_confidence', '<f4')])
    data = np.zeros(shape=(n_boxes,), dtype=dtype)
    data['t'] = detections['t']
    data['x'] = detections['boxes'][:,0]
    data['y'] = detections['boxes'][:,1]
    data['w'] = detections['boxes'][:,2] - data['x']
    data['h'] = detections['boxes'][:,3] - data['y']
    data['class_id'] = detections['labels']
    data['class_confidence'] = detections['scores']
    return data


def save_detections(directory, detections):
    """Save detections grouped by sequence"""
    sequence_detections_map = dict()
    for d in tqdm.tqdm(detections, desc="compiling detections for saving..."):
        s = d['sequence']
        if s not in sequence_detections_map:
            sequence_detections_map[s] = to_npy(d)
        else:
            sequence_detections_map[s] = np.concatenate([sequence_detections_map[s], to_npy(d)])

    for s, detections in sequence_detections_map.items():
        detections = detections[detections['t'].argsort()]
        np.save(directory / f"detections_{s}.npy", detections)


if __name__ == '__main__':
    # Set random seeds for reproducibility
    seed = 42
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    args = FLAGS()
    
    output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory, exp_name=args.exp_name)
    
    project = f"low_latency-{args.dataset}-{args.task}-interframe"
    print(f"PROJECT: {project}")
    log_hparams(args)
    
    print("init datasets")
    dataset_path = args.dataset_directory / args.dataset
    
    augmentations = Augmentations(args)
    
    forced_scale = 2
    adjusted_min_bbox_diag = 15
    adjusted_min_bbox_height = 10
    
    print(f"Testing with scale={forced_scale} (original resolution)")
    
    test_dataset = DSEC(
        root=dataset_path, 
        split="test", 
        transform=augmentations.transform_testing, 
        debug=False,
        min_bbox_diag=adjusted_min_bbox_diag,
        min_bbox_height=adjusted_min_bbox_height,
        only_perfect_tracks=True,
        no_eval=args.no_eval,
        scale=forced_scale
    )
    
    test_loader = DataLoader(
        test_dataset, 
        follow_batch=['bbox', 'bbox0'], 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0, 
        drop_last=True
    )
    
    print("init net")
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)
    model = model.cuda()
    
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Testing with {num_params} number of parameters.")
    
    ema = ModelEMA(model)
    
    assert "checkpoint" in args, "Please provide checkpoint path via --checkpoint argument"
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint)
    
    try:
        ema.ema.load_state_dict(checkpoint['ema'])
        print("Checkpoint loaded successfully (strict=True)")
    except Exception as e:
        print(f"Strict loading failed: {e}")
        try:
            ema.ema.load_state_dict(checkpoint['ema'], strict=False)
            print("Checkpoint loaded successfully (strict=False)")
        except Exception as e2:
            print(f"Loading failed completely: {e2}")
            exit(1)
    
    # Cache LUTs if needed (for some model types)
    if hasattr(ema.ema, 'cache_luts'):
        ema.ema.cache_luts(radius=getattr(args, 'radius', 2), height=test_dataset.height, width=test_dataset.width)
    
    print("starting interframe testing...")
    
    detections = []
    with torch.no_grad():
        # Test across different time windows
        for n_us in np.linspace(0, 50000, args.num_interframe_steps):
            print(f"\nTesting with time window: {int(n_us)} μs")
            test_loader.dataset.set_num_us(int(n_us))
            
            # Run test with visualization and detection compilation
            metrics, detections_one_offset = run_test_with_visualization(
                test_loader, 
                ema.ema, 
                dataset=args.dataset, 
                name=f"{args.exp_name}_interframe", 
                compile_detections=True,
                no_eval=args.no_eval
            )
            
            detections.extend(detections_one_offset)
            
            if metrics is not None:
                pprint(f"Time Window: {int(n_us)} μs \t mAP: {metrics['mAP']}")
                
                # Log to wandb if initialized
                if wandb.run is None:
                    wandb.init(project=project, name=f"{args.exp_name}_interframe")
                
                log_data = {f"interframe/time_{int(n_us)}_us/{k}": v for k, v in metrics.items()}
                wandb.log(log_data)
    
    # Save all detections
    print("\nSaving detections...")
    save_detections(output_directory, detections)
    
    print(f"\nInterframe testing completed!")
    print(f"Detection files saved to: {output_directory}")
    print(f"Use visualize_detections.py to visualize results")