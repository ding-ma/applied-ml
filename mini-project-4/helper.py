from pathlib import Path

BASE_DIR = Path("/home/dataset/ILSVRC/Data/CLS-LOC")
# BASE_DIR = Path("/home/dataset/small_image_net") # only 50 images
TRAIN_SET = BASE_DIR / "train"
TEST_SET = BASE_DIR / "test"
VAL_SET = BASE_DIR / "val"  # the validation dataset will be used a testset for our experiments


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
