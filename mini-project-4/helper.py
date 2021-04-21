from pathlib import Path

BASE_DIR = Path("/home/dataset/imagenet_2010")
TRAIN_SET = BASE_DIR / "train"
TEST_SET = BASE_DIR / "test"
VAL_SET = BASE_DIR / "val"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
