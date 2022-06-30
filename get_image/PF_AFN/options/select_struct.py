import torch


'''
python test.py --name demo --resize_or_crop None --batchSize 1 --gpu_ids 0
'''

class Selector:
    def __init__(self):

        self.name = "real"
        self.gpu_ids = [0]
        self.norm = 'instance'
        self.use_dropout = False
        self.data_type = 32
        self.verbose = False

        self.batchSize = 1
        self.loadSize = 512
        self.fineSize = 512
        self.input_nc = 3
        self.output_nc = 3

        self.dataroot = "./dataset/"
        self.resize_or_crop = 'scale_width'
        self.serial_batches = False
        self.no_flip = False
        self.nThreads = 1
        self.max_dataset_size = float("inf")

        self.display_winsize = 512
        self.tf_log = False

        self.warp_checkpoint = 'checkpoints/PFAFN/warp_model_final.pth'
        self.gen_checkpoint = "checkpoints/PFAFN/gen_model_final.pth"
        self.phase = 'real'
        self.isTrain = False



