from train import get_method
from options import Options
from utils import *


def main():
    opt = Options(isTrain=False)
    opt.parse()

    worker = AELaplaceWorker(opt)
    worker.set_gpu_device()
    worker.set_seed()
    worker.set_network_loss()
    worker.set_logging(test=True)
    worker.load_checkpoint()
    worker.set_dataloader(test=True)
    worker.run_laplace()

if __name__ == "__main__":
    main()
