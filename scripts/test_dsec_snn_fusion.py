import os
import torch
import wandb
import numpy as np
import tqdm
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

for k in ('HDF5_VOL_CONNECTOR', 'HDF5_PLUGIN_PATH'):
    os.environ.pop(k, None)

import hdf5plugin
import h5py

from torch_geometric.data import DataLoader
from torch.utils.data import Subset
from dagr.utils.args import FLAGS
from dagr.data.dsec_data import DSEC
from dagr.data.augment import Augmentations
#双分支单检测头
#from dagr.model.networks.dagr_fusion import DAGR
#双分支双检测头
from dagr.model.networks.dagr_fusion_seperate_heads import DAGR
from dagr.model.networks.ema import ModelEMA
from dagr.utils.logging import set_up_logging_directory, log_hparams
from dagr.utils.buffers import DetectionBuffer, format_data

def run_test(loader: DataLoader, model: torch.nn.Module, dry_run_steps: int=-1, dataset="gen1"):
    model.eval()
    
    base_dataset = loader.dataset
    while isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset

    mapcalc = DetectionBuffer(height=base_dataset.height, width=base_dataset.width, classes=base_dataset.classes)

    for i, data in enumerate(tqdm.tqdm(loader, desc="Testing")):
        data = data.cuda()
        data = format_data(data)

        with torch.no_grad():
            detections, targets = model(data)
        
        if i % 10 == 0:
            torch.cuda.empty_cache()

        mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])

        if dry_run_steps > 0 and i == dry_run_steps:
            break

    torch.cuda.empty_cache()
    return mapcalc

if __name__ == '__main__':
    import torch_geometric
    import random
    
    seed = 42
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    args = FLAGS()
    
    output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory, exp_name=args.exp_name)
    
    project = f"low_latency-{args.dataset}-{args.task}"
    print(f"PROJECT: {project}")
    log_hparams(args)
    
    augmentations = Augmentations(args)
    
    print("init datasets")
    dataset_path = args.dataset_directory / args.dataset
    
    test_dataset = DSEC(root=dataset_path, split="test", transform=augmentations.transform_testing, debug=False, min_bbox_diag=15, min_bbox_height=10)
    
    sampler = np.random.permutation(np.arange(len(test_dataset)))
    test_loader = DataLoader(test_dataset, sampler=sampler, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    print("init net")
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)
    
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Testing with {num_params} number of parameters.")
    
    model = model.cuda()
    
    print("Creating temporal modules with dummy forward pass...")
    dummy_data = next(iter(test_loader))
    dummy_data = dummy_data.cuda()
    dummy_data = format_data(dummy_data)
    
    with torch.no_grad():
        try:
            model(dummy_data)
            print("Dummy forward pass completed - temporal modules created")
        except Exception as e:
            print(f"Dummy forward pass failed: {e}")
    
    print("Checking temporal modules:")
    if hasattr(model, 'event_backbone') and hasattr(model.event_backbone, '_temporal_modules'):
        print(f"Event backbone temporal modules: {list(model.event_backbone._temporal_modules.keys())}")
    if hasattr(model, 'img_backbone') and hasattr(model.img_backbone, '_temporal_modules'):
        print(f"Image backbone temporal modules: {list(model.img_backbone._temporal_modules.keys())}")
    
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
    
    print("starting to test")
    with torch.no_grad():
        mapcalc = run_test(test_loader, ema.ema, dataset=args.dataset)
        metrics = mapcalc.compute()
        
        if wandb.run is None:
            wandb.init(project=project, name=args.exp_name)
        
        log_data = {f"testing/metric/{k}": v for k, v in metrics.items()}
        wandb.log(log_data)

        main_metrics = ["mAP", "mAP_50", "mAP_75", "mAP_S", "mAP_M", "mAP_L"]
        summary_parts = []
        for k in main_metrics:
            if k in metrics:
                try:
                    summary_parts.append(f"{k}={metrics[k]:.4f}")
                except Exception:
                    summary_parts.append(f"{k}={metrics[k]}")
        
        if summary_parts:
            formatted_metrics = ", ".join(summary_parts)
            print(f"[Test Results] {formatted_metrics}")
        
        print("\nAll metrics:")
        for k, v in metrics.items():
            try:
                print(f"  {k}: {v:.4f}")
            except Exception:
                print(f"  {k}: {v}")