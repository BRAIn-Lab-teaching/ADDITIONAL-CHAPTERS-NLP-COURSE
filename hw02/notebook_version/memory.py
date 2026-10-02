"""Непересекающиеся категории CUDA-памяти в конце одного forward."""

import torch

from hw02.notebook_version.runtime import _autocast


def partition_allocations(blocks, groups):
    """Каждую живую аллокацию относим ровно к одной категории."""
    owners = {}
    for category, addresses in groups.items():
        for address in addresses:
            owners.setdefault(address, category)
    result = dict.fromkeys((*groups, "other"), 0)
    for block in blocks:
        if block["state"] == "active_allocated":
            result[owners.get(block["address"], "other")] += block["size"]
    return result


def initialise_state(optimizers):
    """Материализовать ленивое состояние на нулевых градиентах при LR=0."""
    for optimizer in optimizers:
        if optimizer.state:
            continue
        rates = [group["lr"] for group in optimizer.param_groups]
        try:
            for group in optimizer.param_groups:
                group["lr"] = 0.0
                for parameter in group["params"]:
                    if parameter.requires_grad:
                        parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
        finally:
            for group, rate in zip(optimizer.param_groups, rates):
                group["lr"] = rate


class _SavedTensor:
    def __init__(self, tensor, tracker):
        self.tensor = tensor.detach()
        self.address = tensor.untyped_storage().data_ptr()
        self.tracker = tracker

    def __del__(self):
        references = self.tracker.references[self.address] - 1
        if references:
            self.tracker.references[self.address] = references
        else:
            del self.tracker.references[self.address]


class SavedStorages:
    """Адреса storage, сохранённых autograd; views не считаются повторно."""

    def __init__(self):
        self.references = {}

    @property
    def addresses(self):
        return set(self.references)

    def pack(self, tensor):
        packed = _SavedTensor(tensor, self)
        self.references[packed.address] = self.references.get(packed.address, 0) + 1
        return packed

    def unpack(self, packed):
        return packed.tensor


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


def _addresses(tensors, device):
    return {tensor.untyped_storage().data_ptr() for tensor in tensors
            if tensor is not None and tensor.device == device and tensor.numel()}


def measure_memory(model, optimizers, batch, device):
    """Контролируемый проход без изменения весов, с полными буферами градиентов.

    Снимок соответствует концу forward очередного микробатча accumulation,
    когда градиенты предыдущих микробатчей уже материализованы. Это не
    декомпозиция исторического пика обучения.
    """
    device = torch.device("cuda", torch.cuda.current_device()) if device.index is None else device
    parameters = list(model.parameters())
    original_gradients = [parameter.grad for parameter in parameters]
    was_training = model.training
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device)
    try:
        initialise_state(optimizers)
        for parameter in parameters:
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        inputs = {key: value.to(device) for key, value in batch.items()}
        saved = SavedStorages()
        model.train()
        with torch.autograd.graph.saved_tensors_hooks(saved.pack, saved.unpack):
            with _autocast(device):
                dtype = str(torch.get_autocast_dtype("cuda")).removeprefix("torch.")
                output = model(**inputs)
            torch.cuda.synchronize(device)
            groups = {
                "parameters": _addresses([*parameters, *model.buffers()], device),
                "gradients": _addresses((p.grad for p in parameters), device),
                "optimizer": _addresses(_tensors([opt.state for opt in optimizers]), device),
                "inputs": _addresses(inputs.values(), device),
                "saved_for_backward": saved.addresses,
            }
            blocks = [block for segment in torch.cuda.memory_snapshot()
                      if segment["device"] == device.index for block in segment["blocks"]]
            components = partition_allocations(blocks, groups)
            allocated = torch.cuda.memory_allocated(device)
            assert sum(components.values()) == allocated
            result = {
                "phase": "end_of_forward_with_gradient_buffers",
                "components_bytes": components, "allocated_bytes": allocated,
                "reserved_bytes": torch.cuda.memory_reserved(device),
                "batch_shape": list(inputs["input_ids"].shape),
                "gpu": torch.cuda.get_device_name(device),
                "dtype": dtype, "weights_updated": False,
            }
            output.loss.backward()
        return result
    finally:
        for parameter, gradient in zip(parameters, original_gradients):
            parameter.grad = gradient
        model.train(was_training)
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
