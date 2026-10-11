"""FP32 model EMA, updated once per successful optimizer step (YOLOX decay form)."""
import copy
import math
import torch


class ModelEMA:
    def __init__(self, model, decay=0.9998, tau=2000.0):
        if not 0 < decay < 1 or tau <= 0:
            raise ValueError('Invalid EMA decay/tau')
        self.model = copy.deepcopy(model).eval().float().requires_grad_(False)
        self.decay, self.tau, self.updates = float(decay), float(tau), 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / self.tau))
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            if value.is_floating_point():
                assert value.dtype == torch.float32
                value.mul_(decay).add_(source[name].detach().float(), alpha=1-decay)
            else:
                value.copy_(source[name])

    def state_dict(self):
        return dict(state_dict=self.model.state_dict(), updates=self.updates,
                    decay=self.decay, tau=self.tau)

    def load_state_dict(self, state):
        if (state['decay'], state['tau']) != (self.decay, self.tau):
            raise ValueError('EMA contract differs')
        if type(state['updates']) is not int or state['updates'] < 0:
            raise ValueError('Invalid EMA update counter')
        self.model.load_state_dict(state['state_dict'], strict=True)
        self.updates = state['updates']
