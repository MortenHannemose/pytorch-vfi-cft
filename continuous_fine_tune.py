# -*- coding: utf-8 -*-
import numpy as np
from torch.optim import Adam
from torch.utils.data.dataloader import default_collate
import torch.nn as nn

import loss_functions

class CFT():
    def __init__(self, state, net, iters=50, lr=0.0001):
        self.state = state
        self.net = net
        self.iters = iters
        self.optimizer = Adam(net.parameters(), lr=lr)
        loss_weights = {'l1': 1, 'l1_0': 1, 'l1_2': 1, 'tv': 10/255, 'feat': 1/255, 'MoLin': 1/255/255, 'pyra': 10}
        self.cyclic = Cyclic(net, loss_weights)
        self.cyclic.to(net.device)

    def finetune_4(self, real_frames):
        self.net.train()
        #extract frames from queue
        real_frames = [frame for frame_no, frame in real_frames.queue]
        assert(len(real_frames) == 4)

        self.net.load_state_dict(self.state['network'])
        self.optimizer.load_state_dict(self.state['optimizer'])

        mf = 10
        max_win_size = min(real_frames[0].shape[1:3])-mf*3
        size = max_win_size - max_win_size%32
        size = min(256, size)

        for it in range(self.iters):
            ims_crop = []
            dx = np.random.randint(-mf, mf)
            dy = np.random.randint(-mf, mf)
            left_bounds = (max(0-3*dx, 0), real_frames[0].shape[2]-size+min(0, -3*dx))
            top_bounds = (max(0-3*dy, 0), real_frames[0].shape[1]-size+min(0, -3*dy))
            left = np.random.randint(left_bounds[0], left_bounds[1])
            top = np.random.randint(top_bounds[0], top_bounds[1])

            for i in range(4):
                top_bound = top+i*dy
                left_bound = left+i*dx
                ims_crop.append(real_frames[i][:, top_bound:top_bound+size, left_bound:left_bound+size])
            f = default_collate(ims_crop)
            f1_tru = f[1:-1]

            cat = lambda a, b: [a, b] if np.random.rand() > 0.5 else [b, a]
            f_5 = self.net(f[:-1], f[1:])['output_im']
            f_5a, f_5b = cat(f_5[:-1], f_5[1:])

            loss_cyc = self.cyclic(f_5a, f_5b, f1_tru)
            loss = loss_cyc['SMloss'].mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.net.eval()

class Cyclic(nn.Module):
    def __init__(self, net, loss_weights):
        super(Cyclic, self).__init__()
        self.net = net
        self.loss_weights = loss_weights
        self.feature_loss = 'feat' in loss_weights
        self.loss_fun = loss_functions.SMLoss(loss_weights)

    def forward(self, im0, im1, im2):
        out_dict = self.net(im0, im2)
        self.compute_pyramids(out_dict, im0, im2)
        loss_cyc = self.loss_fun(out_dict, im1)
        return loss_cyc

    def compute_pyramids(self, out_dict, input0, input2):
        flow_pyramid = []
        prev = out_dict['uvm']
        for level in range(4):
            prev = self.net.down(prev)/2
            flow_pyramid.append(prev)
        interp0_pyramid = []
        interp2_pyramid = []
        input0_scaled = input0
        input2_scaled = input2
        for flow in flow_pyramid:
            input0_scaled = self.net.down(input0_scaled)
            input2_scaled = self.net.down(input2_scaled)
            interp0_pyramid.append(self.net.interpolation(flow, input0_scaled, 0)[0])
            interp2_pyramid.append(self.net.interpolation(flow, input2_scaled, 1)[0])
        out_dict['interp0_pyramid'] = interp0_pyramid
        out_dict['interp2_pyramid'] = interp2_pyramid
    