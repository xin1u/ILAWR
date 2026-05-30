from contextlib import contextmanager


class GradAccumulator:
    def __init__(self, accum_steps=1):
        self.accum_steps = accum_steps

    def is_accumulation_step(self, step):
        return (step % self.accum_steps) != 0

    def should_step(self, step):
        return (step % self.accum_steps) == 0

    def scale_loss(self, loss):
        return loss / self.accum_steps

    @contextmanager
    def no_sync_if_accumulating(self, model, step):
        if self.is_accumulation_step(step) and hasattr(model, 'no_sync'):
            with model.no_sync():
                yield
        else:
            yield


if __name__ == '__main__':
    ga = GradAccumulator(accum_steps=4)
    for step in range(1, 9):
        print(f'step={step}, accumulating={ga.is_accumulation_step(step)}, '
              f'should_step={ga.should_step(step)}')
