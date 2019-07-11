# -*- coding: utf-8 -*-
import cv2
import utils
from torch.utils.data.dataloader import default_collate

weight_path = 'VFI_CFT_weights.pt.gz'
net = utils.load_model(weight_path)

fh = utils.FrameHandler(None)
#OpenCV uses the bgr color order
im0 = fh.bgr_to_tensor(cv2.imread('images/im0.png'))
im2 = fh.bgr_to_tensor(cv2.imread('images/im2.png'))

#Add minibatch dimension to tensors
im0 = default_collate([im0])
im2 = default_collate([im2])

out_dict = net(im0, im2)
#get the output image, and only the first image in the minibatch
im1 = out_dict['output_im'][0]

im1_np = fh.tensor_to_numpy_bgr(im1)
cv2.imwrite("images/im1.png", im1_np)