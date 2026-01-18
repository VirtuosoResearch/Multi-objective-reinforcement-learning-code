import torch
from torch import autograd
import numpy as np
import json
import os
import re

from torch.nn.attention import SDPBackend, sdpa_kernel

@torch.no_grad()
def _rademacher_like(params):
    # +1/-1 with equal prob; keep dtype/device to match param
    return [torch.empty_like(p).bernoulli_(0.5).mul_(2).add_(-1) for p in params]

def hessian_trace_vhv(loss, params, num_v=10):
    """
    Estimate trace(H) of the Hessian of `loss` wrt `params` using Hutchinson.
    Args:
        loss: scalar tensor
        params: iterable of Tensors with requires_grad=True
        num_v: number of probe vectors (more -> lower variance)
    Returns:
        trace_estimate (scalar tensor on same device)
    """
    # First grad; must keep graph for the second derivative
    g = autograd.grad(loss, params, create_graph=True)
    trace_est = 0.0
    for _ in range(num_v):
        v = [ _rademacher_like(p) for p in params ]
        # v^T * g (a scalar)
        vg = sum((vi * gi).sum() for vi, gi in zip(v, g))
        # ∂(v^T g)/∂θ  = H v
        Hv = autograd.grad(vg, params, retain_graph=True)
        # v^T (H v)
        vHv = sum((vi * hvi).sum() for vi, hvi in zip(v, Hv))
        trace_est = trace_est + vHv
    # Free the graph when done
    trace_est = trace_est / float(num_v)
    return trace_est

def get_validation_hessian(model, use_nso, model_path, model_name, valid_loader, datamodels_num, loss_fn, load_batch_fn, device='cuda'):
    pattern = re.compile(rf"^{re.escape(model_name)}_\w+\.pth$")
    model_file_list = []
    # for file in os.listdir(model_path):
    #     if pattern.match(file):
    #         model_file = os.path.join(model_path, file)
    #         model_file_list.append(model_file)
    num_models = 0
    for file in os.listdir(model_path):
        if pattern.match(file):
            num_models += 1
    for i in range(num_models):
        if use_nso:
            model_file = os.path.join(model_path, f"nso_model_{i}.pth")
        else:
            model_file = os.path.join(model_path, f"model_{i}.pth")
        model_file_list.append(model_file)

    # Compute Hessian trace by trace_vhv
    num_probes = 10
    trace_list = []
    record_loss = 0
    with sdpa_kernel(SDPBackend.MATH):
        for i in range(datamodels_num, num_models):
            model_file = model_file_list[i]
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
            model.eval()
            params = [p for p in model.parameters() if p.requires_grad]

            trace_sum = 0.0
            for _ in range(num_probes):
                v = _rademacher_like(params)

                vHv_sum = 0.0
                count = 0
                test_loss = 0
                for batch in valid_loader:
                    inputs, targets, batch_size = load_batch_fn(batch, device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    test_loss += loss.item() * batch_size
                    g = autograd.grad(loss, params, create_graph=True)
                    vg = sum((vi * gi).sum() for vi, gi in zip(v, g))
                    Hv = autograd.grad(vg, params, retain_graph=False, create_graph=False)
                    vHv = sum((vi * hvi).sum() for vi, hvi in zip(v, Hv))
                    vHv_sum += vHv.detach() * batch_size
                    count += batch_size
                    # test loss


                    del loss, outputs, g, vg, Hv, vHv
                test_loss /= max(count, 1)
                trace_sum += (vHv_sum / max(count, 1))
            trace = trace_sum / float(num_probes)
            trace_list.append(trace)

        trace_avg = sum(trace_list) / len(trace_list)

    return trace_avg.item(), test_loss 

def get_per_sample_hessian(model, use_nso, model_path, model_name, test_dataset, datamodels_num, loss_fn, load_batch_fn, device='cuda', num_probes=10):
    """
    Compute per test sample Hessian trace for each model.
    
    Args:
        model: The model to compute Hessian for
        use_nso: Whether to use NSO models
        model_path: Path to model files
        model_name: Name pattern for model files
        test_dataset: Test dataset
        datamodels_num: Number of models to use for training
        loss_fn: Loss function
        load_batch_fn: Function to load batch data
        device: Device to use
        num_probes: Number of probe vectors for Hutchinson estimator
    
    Returns:
        Dictionary with per-sample Hessian traces for each model
    """
    pattern = re.compile(rf"^{re.escape(model_name)}_\w+\.pth$")
    model_file_list = []
    
    num_models = 0
    for file in os.listdir(model_path):
        if pattern.match(file):
            num_models += 1
    for i in range(num_models):
        if use_nso:
            model_file = os.path.join(model_path, f"nso_model_{i}.pth")
        else:
            model_file = os.path.join(model_path, f"model_{i}.pth")
        model_file_list.append(model_file)

    # Create a single sample dataloader for per-sample computation
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,  # Use 0 to avoid multiprocessing issues
        pin_memory=False
    )

    per_sample_hessians = {}
    
    with sdpa_kernel(SDPBackend.MATH):
        for i in range(datamodels_num, num_models):
            model_file = model_file_list[i]
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
            model.eval()
            params = [p for p in model.parameters() if p.requires_grad]

            sample_hessians = []
            
            for sample_idx, batch in enumerate(test_loader):
                inputs, targets, batch_size = load_batch_fn(batch, device)
                
                # Compute Hessian trace for this single sample
                trace_sum = 0.0
                for _ in range(num_probes):
                    v = _rademacher_like(params)
                    
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    
                    g = autograd.grad(loss, params, create_graph=True)
                    vg = sum((vi * gi).sum() for vi, gi in zip(v, g))
                    Hv = autograd.grad(vg, params, retain_graph=False, create_graph=False)
                    vHv = sum((vi * hvi).sum() for vi, hvi in zip(v, Hv))
                    trace_sum += vHv.detach()
                    
                    del loss, outputs, g, vg, Hv, vHv
                
                trace = trace_sum / float(num_probes)
                sample_hessians.append(trace.item())
                
                if sample_idx % 100 == 0:
                    print(f"Model {i}, Sample {sample_idx}/{len(test_dataset)}")
            
            per_sample_hessians[f'model_{i}'] = sample_hessians
            print(f"Completed Hessian computation for model {i}")

    return per_sample_hessians


def get_training_hessian(model, use_nso, model_path, model_name, train_loaders, datamodels_num, loss_fn, load_batch_fn, device='cuda'):
    pattern = re.compile(rf"^{re.escape(model_name)}_\w+\.pth$")
    model_file_list = []
    # for file in os.listdir(model_path):
    #     if pattern.match(file):
    #         model_file = os.path.join(model_path, file)
    #         model_file_list.append(model_file)
    num_models = 0
    for file in os.listdir(model_path):
        if pattern.match(file):
            num_models += 1
    for i in range(num_models):
        if use_nso:
            model_file = os.path.join(model_path, f"nso_model_{i}.pth")
        else:
            model_file = os.path.join(model_path, f"model_{i}.pth")
        model_file_list.append(model_file)

    # Compute Hessian trace by trace_vhv
    num_probes = 10
    trace_list = []
    with sdpa_kernel(SDPBackend.MATH):
        for i in range(datamodels_num, num_models):
            model_file = model_file_list[i]
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
            model.eval()
            params = [p for p in model.parameters() if p.requires_grad]

            trace_sum = 0.0
            for _ in range(num_probes):
                v = _rademacher_like(params)

                vHv_sum = 0.0
                count = 0

                for batch in train_loaders[i]:
                    inputs, targets, batch_size = load_batch_fn(batch, device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)

                    g = autograd.grad(loss, params, create_graph=True)
                    vg = sum((vi * gi).sum() for vi, gi in zip(v, g))
                    Hv = autograd.grad(vg, params, retain_graph=False, create_graph=False)
                    vHv = sum((vi * hvi).sum() for vi, hvi in zip(v, Hv))
                    vHv_sum += vHv.detach() * batch_size
                    count += batch_size

                    del loss, outputs, g, vg, Hv, vHv

                trace_sum += (vHv_sum / max(count, 1))
            trace = trace_sum / float(num_probes)
            trace_list.append(trace)

        trace_avg = sum(trace_list) / len(trace_list)

    return trace_avg.item()

def get_per_sample_hessian(model, use_nso, model_path, model_name, test_dataset, datamodels_num, loss_fn, load_batch_fn, device='cuda', num_probes=10):
    """
    Compute per test sample Hessian trace for each model.
    
    Args:
        model: The model to compute Hessian for
        use_nso: Whether to use NSO models
        model_path: Path to model files
        model_name: Name pattern for model files
        test_dataset: Test dataset
        datamodels_num: Number of models to use for training
        loss_fn: Loss function
        load_batch_fn: Function to load batch data
        device: Device to use
        num_probes: Number of probe vectors for Hutchinson estimator
    
    Returns:
        Dictionary with per-sample Hessian traces for each model
    """
    pattern = re.compile(rf"^{re.escape(model_name)}_\w+\.pth$")
    model_file_list = []
    
    num_models = 0
    for file in os.listdir(model_path):
        if pattern.match(file):
            num_models += 1
    for i in range(num_models):
        if use_nso:
            model_file = os.path.join(model_path, f"nso_model_{i}.pth")
        else:
            model_file = os.path.join(model_path, f"model_{i}.pth")
        model_file_list.append(model_file)

    # Create a single sample dataloader for per-sample computation
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,  # Use 0 to avoid multiprocessing issues
        pin_memory=False
    )

    per_sample_hessians = {}
    
    with sdpa_kernel(SDPBackend.MATH):
        for i in range(datamodels_num, num_models):
            model_file = model_file_list[i]
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
            model.eval()
            params = [p for p in model.parameters() if p.requires_grad]

            sample_hessians = []
            
            for sample_idx, batch in enumerate(test_loader):
                inputs, targets, batch_size = load_batch_fn(batch, device)
                
                # Compute Hessian trace for this single sample
                trace_sum = 0.0
                for _ in range(num_probes):
                    v = _rademacher_like(params)
                    
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    
                    g = autograd.grad(loss, params, create_graph=True)
                    vg = sum((vi * gi).sum() for vi, gi in zip(v, g))
                    Hv = autograd.grad(vg, params, retain_graph=False, create_graph=False)
                    vHv = sum((vi * hvi).sum() for vi, hvi in zip(v, Hv))
                    trace_sum += vHv.detach()
                    
                    del loss, outputs, g, vg, Hv, vHv
                
                trace = trace_sum / float(num_probes)
                sample_hessians.append(trace.item())
                
                if sample_idx % 100 == 0:
                    print(f"Model {i}, Sample {sample_idx}/{len(test_dataset)}")
            
            per_sample_hessians[f'model_{i}'] = sample_hessians
            print(f"Completed Hessian computation for model {i}")

    return per_sample_hessians
