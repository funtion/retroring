import math
import torch


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class WarmCosine:
    def __init__(self, warmup=4e3, tmax=1e5, eta_min=5e-4):
        if warmup is None:
            self.warmup = 0
        else:
            warmup_step = int(warmup)
            assert warmup_step > 0
            self.warmup = warmup_step
            self.lr_step = (1 - eta_min) / warmup_step
        self.tmax = int(tmax)
        self.eta_min = eta_min

    def step(self, step):
        if step >= self.warmup:
            return (
                self.eta_min
                + (1 - self.eta_min)
                * (1 + math.cos(math.pi * (step - self.warmup) / self.tmax))
                / 2
            )

        else:
            return self.eta_min + self.lr_step * step


class WarmLinear:
    def __init__(self, warmup=4e3, tmax=1e5, eta_min=5e-4):
        if warmup is None:
            self.warmup_step = 0
        else:
            warmup_step = int(warmup)
            assert warmup_step > 0
            self.warmup_step = warmup_step
            self.warmup_lr_step = (1 - eta_min) / warmup_step
        self.decay_lr_step = (eta_min - 1) / (tmax - self.warmup_step)
        self.eta_min = eta_min

    def step(self, step):
        if step >= self.warmup_step:
            return max(self.eta_min, 1 + self.decay_lr_step * (step - self.warmup_step))
        else:
            return max(self.eta_min, self.eta_min + self.warmup_lr_step * step)
