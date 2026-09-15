def create_emulator(config):
    return StepFixture()


class StepFixture:
    def set_step(self, step):
        self.step = step

    def infer(self, inputs, outputs):
        outputs['y'][:] = inputs['x'] + self.step
