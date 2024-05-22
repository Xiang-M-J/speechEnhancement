from trainer_base import TrainerBase
from trainer_utils import Args


class Test(TrainerBase):
    def get_loss_fn(self, *args, **kwargs):
        pass

    def train_step(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def test_step(self, *args, **kwargs):
        pass

    def __init__(self, args):
        TrainerBase.__init__(self, args)


arg = Args("test")
arg.save = False
test = Test(arg)
