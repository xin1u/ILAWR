import math


def cosine_decay_lr(step, total_steps, base_lr=2e-4):
    return 0.5 * (1 + math.cos(step * math.pi / total_steps)) * base_lr
