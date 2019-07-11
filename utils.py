# -*- coding: utf-8 -*-
import os
import gzip
import numpy as np
import torch

from continuous_fine_tune import CFT
from network import Network

def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model(weight_path, continuous_fine_tuning=False):
    weight_path = os.path.abspath(weight_path)
    if not os.path.exists(weight_path):
        raise RuntimeError("Could not find file: " + weight_path + ". Did you remember to download the pretrained model?")
    if weight_path.endswith('.gz'):
        with gzip.open(weight_path, 'rb') as f_in:
            state = torch.load(f_in, get_device())
    else:
        state = torch.load(weight_path, get_device())
    net = Network()
    net.to(net.device)
    net.load_state_dict(state['network'])

    if continuous_fine_tuning:
        net.cft = CFT(state, net)
    return net

def clear_dir(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

class FrameHandler():
    def __init__(self, frame_shape):
        if frame_shape is None:
            self.orig_size = None
            return
        must_divide = 32
        self.orig_size = frame_shape
        self.padding = [((-dim%must_divide)//2,)*2 for dim in frame_shape[:2]]
        self.padding.append((0, 0))

    def bgr_to_tensor(self, bgr_frame):
        return self.rgb_to_tensor(np.flip(bgr_frame, 2))

    def rgb_to_tensor(self, frame):
        if self.orig_size is None:
            self.__init__(frame.shape)

        frame = np.pad(frame, self.padding, 'reflect')
        frame = torch.from_numpy(frame).to(get_device())
        frame = (frame.permute((2, 0, 1))).type(torch.float32)/255
        return frame

    def tensor_to_numpy_bgr(self, frame):
        return np.flip(self.tensor_to_numpy_rgb(frame), 2)

    def tensor_to_numpy_rgb(self, frame):
        frame = (frame.permute((1, 2, 0)))*255
        for dim in range(2):
            if (self.padding[dim][0]+self.padding[dim][1]) > 0:
                frame = frame.narrow(dim, self.padding[dim][0], self.orig_size[dim])
        frame = frame.detach().cpu().numpy()
        return frame
    