#Temporal Transformer聚合
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
from dagr.model.networks.dagr_snn import DAGR
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
    
    print("init datasets")
    dataset_path = args.dataset_directory / args.dataset
    
    augmentations = Augmentations(args)
    test_dataset = DSEC(root=dataset_path, split="test", transform=augmentations.transform_testing, debug=False, min_bbox_diag=15, min_bbox_height=10)
    
    sampler = np.random.permutation(np.arange(len(test_dataset)))
    test_loader = DataLoader(test_dataset, sampler=sampler, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    print("init net")
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)
    model = model.cuda()
    
    #print("Creating temporal modules with dummy forward pass...")
    dummy_data = next(iter(test_loader))
    dummy_data = dummy_data.cuda()
    dummy_data = format_data(dummy_data)
    
    with torch.no_grad():
        try:
            model(dummy_data)
            #print("Dummy forward pass completed - temporal modules created")
        except Exception as e:
            print(f"Dummy forward pass failed: {e}")
    
    print("Checking temporal modules:")
    if hasattr(model, 'backbone') and hasattr(model.backbone, '_temporal_modules'):
        print(f"Temporal modules found: {list(model.backbone._temporal_modules.keys())}")
    
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




#TCM 聚合
# # avoid matlab error on server
# import os
# import torch
# import wandb
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# import numpy as np
# import random
# from torch_geometric.data import DataLoader

# from dagr.utils.args import FLAGS
# from dagr.data.dsec_data import DSEC
# from dagr.data.augment import Augmentations

# from dagr.model.networks.dagr_snn import DAGR
# from dagr.model.networks.ema import ModelEMA

# from dagr.utils.logging import set_up_logging_directory, log_hparams
# from dagr.utils.testing import run_test_with_visualization


# def warmup_tcm_modules(model: torch.nn.Module, height: int, width: int, T: int = 4, device: str = "cuda"):
#     # 仅当包含 TCM 时才初始化
#     bb = getattr(model, "backbone", None)
#     if bb is None or not hasattr(bb, "_tcm"):
#         return
#     # 获取通道与输出尺寸
#     out_chs = getattr(bb, "out_channels", [256, 512])  # p4, p5
#     if hasattr(bb, "get_output_sizes"):
#         sizes = bb.get_output_sizes()  # [[Hp4,Wp4],[Hp5,Wp5]]
#     else:
#         sizes = [[max(1, height // 16), max(1, width // 16)],
#                  [max(1, height // 32), max(1, width // 32)]]
#     # 懒初始化 _tcm.p4 / _tcm.p5
#     bb.tcm_enabled = True
#     with torch.no_grad():
#         # p4
#         if len(out_chs) >= 1:
#             C4, (H4, W4) = int(out_chs[0]), sizes[0]
#             x_tbchw = torch.zeros((T, 1, C4, H4, W4), device=device, dtype=torch.float32)
#             try:
#                 bb._aggregate_time(x_tbchw, "p4")
#             except Exception:
#                 pass
#         # p5
#         if len(out_chs) >= 2:
#             C5, (H5, W5) = int(out_chs[1]), sizes[1]
#             x_tbchw = torch.zeros((T, 1, C5, H5, W5), device=device, dtype=torch.float32)
#             try:
#                 bb._aggregate_time(x_tbchw, "p5")
#             except Exception:
#                 pass


# if __name__ == '__main__':
#     import torch_geometric

#     # Seed
#     seed = 42
#     torch_geometric.seed.seed_everything(seed)
#     torch.random.manual_seed(seed)
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)

#     # Args
#     args = FLAGS()

#     # Logging
#     output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory)
#     project = f"low_latency-{args.dataset}-{args.task}"
#     print(f"PROJECT: {project}")
#     log_hparams(args)

#     # Dataset
#     print("init datasets")
#     augment = Augmentations(args)
#     dataset_path = args.dataset_directory / args.dataset
#     eval_split = 'test'
#     print(f"[Test] eval_split = {eval_split}")
#     test_dataset = DSEC(
#         root=dataset_path,
#         split=eval_split,
#         transform=augment.transform_testing,
#         debug=False,
#         min_bbox_diag=15,
#         min_bbox_height=10
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         follow_batch=['bbox', 'bbox0'],
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=4,
#         drop_last=False
#     )
#     print(f"[Test] dataset size = {len(test_dataset)}, classes = {getattr(test_dataset, 'classes', None)}")

#     # Model + EMA（强制启用 TCM）
#     print("init net")
#     setattr(args, 'tcm_enabled', True)
#     model = DAGR(args, height=test_dataset.height, width=test_dataset.width).cuda()
#     ema = ModelEMA(model)

#     # Checkpoint（文件路径）
#     assert hasattr(args, 'checkpoint') and args.checkpoint, "--checkpoint is required"
#     if not os.path.isfile(args.checkpoint):
#         raise FileNotFoundError(args.checkpoint)
#     print(f"[Checkpoint] file = {args.checkpoint}")

#     # 用虚拟张量预热 TCM，避免 _tcm 键变成 unexpected
#     T_bins = int(getattr(args, 'snn_temporal_bins', 4))
#     warmup_tcm_modules(ema.ema, test_dataset.height, test_dataset.width, T=T_bins, device="cuda")

#     # 加载 EMA 权重（不做 _tcm 过滤）
#     ckpt = torch.load(args.checkpoint, map_location='cpu')
#     state = ckpt.get('ema') or ckpt.get('model') or ckpt if isinstance(ckpt, dict) else ckpt
#     res = ema.ema.load_state_dict(state, strict=False)
#     try:
#         print("[Load][EMA] missing:", getattr(res, "missing_keys", []))
#         print("[Load][EMA] unexpected:", getattr(res, "unexpected_keys", []))
#     except Exception:
#         pass

#     # Eval
#     with torch.no_grad():
#         metrics = run_test_with_visualization(test_loader, ema.ema, dataset=args.dataset)
#         wandb.log({f"testing/metric/{k}": v for k, v in metrics.items()})
#         main_metrics = ["mAP", "mAP_50", "mAP_75", "mAP_S", "mAP_M", "mAP_L"]
#         summary = ", ".join([f"{k}={metrics[k]:.4f}" for k in main_metrics if k in metrics])
#         print(summary if summary else str(metrics))


#简单mean聚合
# # avoid matlab error on server
# import os
# import torch
# import wandb
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# from torch_geometric.data import DataLoader
# from dagr.utils.args import FLAGS

# from dagr.data.dsec_data import DSEC
# from dagr.data.augment import Augmentations

# from dagr.model.networks.dagr_snn import DAGR
# from dagr.model.networks.ema import ModelEMA

# from dagr.utils.logging import set_up_logging_directory, log_hparams
# from dagr.utils.testing import run_test_with_visualization 

# if __name__ == '__main__':
#     import torch_geometric
#     import random
#     import numpy as np
    
#     seed = 42
#     torch_geometric.seed.seed_everything(seed)
#     torch.random.manual_seed(seed)
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
    
#     args = FLAGS()
    
#     output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory)
    
#     project = f"low_latency-{args.dataset}-{args.task}"
#     print(f"PROJECT: {project}")
#     log_hparams(args)
    
#     print("init datasets")

#     dataset_path = args.dataset_directory / args.dataset
#     test_dataset = DSEC(root=dataset_path, split="test", transform=Augmentations.transform_testing, debug=False, min_bbox_diag=15, min_bbox_height=10)
    
#     num_iters_per_epoch = 1
    
#     sampler = np.random.permutation(np.arange(len(test_dataset)))
#     test_loader = DataLoader(test_dataset, sampler=sampler, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    
#     print("init net")
#     # load a dummy sample to get height, width
#     model = DAGR(args, height=test_dataset.height, width=test_dataset.width)
#     model = model.cuda()
#     ema = ModelEMA(model)
    
#     assert "checkpoint" in args
#     checkpoint = torch.load(args.checkpoint)
#     ema.ema.load_state_dict(checkpoint['ema'])
    
#     with torch.no_grad():
#         metrics = run_test_with_visualization(test_loader, ema.ema, dataset=args.dataset)
#         log_data = {f"testing/metric/{k}": v for k, v in metrics.items()}
#         wandb.log(log_data)

#         main_metrics = ["mAP", "mAP_50", "mAP_75", "mAP_S", "mAP_M", "mAP_L"]
#         formatted_metrics = ", ".join([f"{k}={metrics[k]:.4f}" for k in main_metrics if k in metrics])
#         print(formatted_metrics)