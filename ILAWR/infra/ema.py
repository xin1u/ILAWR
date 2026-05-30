import copy
from collections import OrderedDict

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self._backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self):
        return {'decay': self.decay, 'shadow': copy.deepcopy(self.shadow)}

    def load_state_dict(self, state):
        self.decay = state['decay']
        self.shadow = state['shadow']


if __name__ == '__main__':
    model = nn.Linear(10, 10)
    ema = ModelEMA(model, decay=0.999)
    model.weight.data.fill_(1.0)
    ema.update(model)
    print(f'shadow weight mean: {ema.shadow["weight"].mean().item():.6f}')
    print(f'state_dict keys: {list(ema.state_dict().keys())}')
