#!/usr/bin/env python3

class Config:

    def __init__(self,
                 input_ch=3,
                 padded_im_size=32,
                 num_classes=10,
                 im_size=32,
                 epc_seed=0,
                 resnet_type='small'
                 ):

        super(Config, self).__init__()

        self.input_ch = input_ch
        self.padded_im_size = padded_im_size
        self.num_classes = num_classes
        self.im_size = im_size
        self.epc_seed = epc_seed
        self.resnet_type = resnet_type
