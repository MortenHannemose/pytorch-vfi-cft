# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
if torch.cuda.is_available():
    from torch.cuda import LongTensor, FloatTensor
else:
    from torch import LongTensor, FloatTensor
import utils

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        if not torch.cuda.is_available():
            print("Running on CPU!")

        self.device = utils.get_device()

        def conv_ReLU(n_in, n_out, kernel_size=3, stride=1, pad=1, bias=True):
            return [nn.Conv2d(n_in, n_out, kernel_size, stride, pad, bias=bias),
                    nn.ReLU(True)]

        def basic(in_channels, out_channels, layers=3):
            modules = conv_ReLU(in_channels, out_channels)
            for i in range(layers-1):
                modules.extend(conv_ReLU(out_channels, out_channels))
            return nn.Sequential(*modules)

        def basic_up(in_channels, out_channels, layers=1):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                              basic(in_channels, out_channels, layers))

        self.conv1 = basic(6, 32)
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = basic(32, 64)
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = basic(64, 128)
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        self.conv4 = basic(128, 256)
        self.pool4 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = basic(256, 512)
        self.pool5 = nn.AvgPool2d(kernel_size=2)

        self.conv6 = basic(512, 512)

        self.upsample_and_deconv1 = basic_up(512, 512)
        self.deconv1 = basic(512, 512, layers=2)

        self.upsample_and_deconv2 = basic_up(512, 256)
        self.deconv2 = basic(256, 256, layers=2)

        self.upsample_and_deconv3 = basic_up(256, 128)
        self.deconv3 = basic(128, 128, layers=2)

        self.upsample_and_deconv4 = basic_up(128, 64)
        self.deconv4 = basic(64, 64, layers=2)

        self.upsample5 = basic_up(64, 32, layers=3)

        self.uvm = nn.Conv2d(32, 6, kernel_size=3, padding=1, bias=False)

        self.down = nn.AvgPool2d(kernel_size=2)

    def update_im_size(self, im_size):
        self.im_size = im_size

    def interpolation(self, uvm, image, index):
        u, v = torch.index_select(uvm, dim=1, index=LongTensor([0+3*index,
                                  1+3*index])).permute(0, 2, 3, 1).split(1, dim=3)
        row_num = FloatTensor()
        col_num = FloatTensor()
        im_size = image.shape[2:4]
        torch.arange(im_size[0], out=row_num)
        torch.arange(im_size[1], out=col_num)
        row_num = row_num.view(1, im_size[0], 1, 1)
        col_num = col_num.view(1, 1, im_size[1], 1)
        x_norm = 2*(u+col_num)/(im_size[1]-1)-1
        y_norm = 2*(v+row_num)/(im_size[0]-1)-1
        xy_norm = torch.clamp(torch.cat((x_norm, y_norm), dim=3), -1, 1)
        interp = nn.functional.grid_sample(image, xy_norm)
        w = torch.index_select(uvm, dim=1, index=LongTensor([3*index+2]))+0.5
        return interp, w, u, v

    def forward(self, input0, input2):
        var_join_input = torch.cat([input0, input2], dim=1)

        var_conv1 = self.conv1(var_join_input)
        var_pool1 = self.pool1(var_conv1)

        var_conv2 = self.conv2(var_pool1)
        var_pool2 = self.pool2(var_conv2)

        var_conv3 = self.conv3(var_pool2)
        var_pool3 = self.pool3(var_conv3)

        var_conv4 = self.conv4(var_pool3)
        var_pool4 = self.pool4(var_conv4)

        var_conv5 = self.conv5(var_pool4)
        var_pool5 = self.pool5(var_conv5)

        var_conv6 = self.conv6(var_pool5)

        var_upsample_and_deconv1 = self.upsample_and_deconv1(var_conv6)
        var_join1 = var_upsample_and_deconv1 + var_conv5
        var_deconv1 = self.deconv1(var_join1)

        var_upsample_and_deconv2 = self.upsample_and_deconv2(var_deconv1)
        var_join2 = var_upsample_and_deconv2 + var_conv4
        var_deconv2 = self.deconv2(var_join2)

        var_upsample_and_deconv3 = self.upsample_and_deconv3(var_deconv2)
        var_join3 = var_upsample_and_deconv3 + var_conv3
        var_deconv3 = self.deconv3(var_join3)

        var_upsample_and_deconv4 = self.upsample_and_deconv4(var_deconv3)
        var_join4 = var_upsample_and_deconv4 + var_conv2
        var_deconv4 = self.deconv4(var_join4)

        var_upsample5 = self.upsample5(var_deconv4)
        var_uvm = self.uvm(var_upsample5)

        interp0, w0, u0, v0 = self.interpolation(var_uvm, input0, 0)
        interp2, w2, u2, v2 = self.interpolation(var_uvm, input2, 1)

        output = w0 * interp0 + w2 * interp2
        out_dict = {'output_im': output, 'interp0': interp0, 'interp2': interp2,
                    'u0': u0, 'v0': v0, 'u2': u2, 'v2': v2, 'w0': w0, 'w2': w2,
                    'uvm': var_uvm}
        return out_dict
