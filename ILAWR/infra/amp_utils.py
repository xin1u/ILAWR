import torch
from torch.amp import autocast, GradScaler


class AMPContext:
    def __init__(self, enabled=True, dtype='fp16'):
        self.enabled = enabled
        if dtype == 'fp16':
            self.amp_dtype = torch.float16
        elif dtype == 'bf16':
            self.amp_dtype = torch.bfloat16
        else:
            raise ValueError(f'Unsupported AMP dtype: {dtype}')

        self.scaler = GradScaler(enabled=enabled and dtype == 'fp16')

    def forward_context(self):
        return autocast('cuda', enabled=self.enabled, dtype=self.amp_dtype)

    def backward_and_step(self, loss, optimizer, max_norm=1.0, parameters=None):
        self.scaler.scale(loss).backward()
        if parameters is not None:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        self.scaler.step(optimizer)
        self.scaler.update()

    def state_dict(self):
        return self.scaler.state_dict()

    def load_state_dict(self, state):
        self.scaler.load_state_dict(state)


if __name__ == '__main__':
    amp_ctx = AMPContext(enabled=True, dtype='fp16')
    print(f'dtype={amp_ctx.amp_dtype}, scaler_enabled={amp_ctx.scaler.is_enabled()}')

    amp_bf16 = AMPContext(enabled=True, dtype='bf16')
    print(f'dtype={amp_bf16.amp_dtype}, scaler_enabled={amp_bf16.scaler.is_enabled()}')
