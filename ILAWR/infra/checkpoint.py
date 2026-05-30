import os
import torch
import torch.distributed as dist

from .dist_utils import is_main_process, barrier


def _strip_ddp_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        new_state[k.removeprefix('module.')] = v
    return new_state


def save_checkpoint(path, model, optimizer=None, scaler=None,
                    ema=None, step=0, best_metrics=None, extra=None):
    if not is_main_process():
        return

    state = {
        'step': step,
        'model': _strip_ddp_prefix(model.state_dict()),
    }
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    if ema is not None:
        state['ema'] = ema.state_dict()
    if best_metrics is not None:
        state['best_metrics'] = best_metrics
    if extra is not None:
        state.update(extra)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scaler=None,
                    ema=None, map_location='cpu', strict=True):
    if not os.path.isfile(path):
        return 0

    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    model_state = ckpt.get('model', ckpt.get('net', {}))
    model_state = _strip_ddp_prefix(model_state)
    model.load_state_dict(model_state, strict=strict)

    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scaler is not None and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    if ema is not None and 'ema' in ckpt:
        ema.load_state_dict(ckpt['ema'])

    barrier()
    return ckpt.get('step', 0)


if __name__ == '__main__':
    import torch.nn as nn
    model = nn.Linear(10, 10)
    path = '/tmp/test_ckpt.pth'
    save_checkpoint(path, model, step=100, best_metrics={'psnr': 30.0})
    step = load_checkpoint(path, model)
    print(f'resumed from step={step}')
    os.remove(path)
