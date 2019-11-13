# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
from queue import Queue
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import default_collate

import utils

def SlowMovie(vid_in_path, slow_factor=2, continuous_fine_tuning=False, 
              tmp_dir = './slowed_movie_frames/', cpu_max_size = 320):
    if np.log2(slow_factor)%1!=0:
        raise ValueError("Slow factor must be a power of 2!")
    def write_frame(frame, frame_no):
        out_path = os.path.join(tmp_dir, '{0:09d}.bmp'.format(frame_no))
        assert(not os.path.exists(out_path))
        cv2.imwrite(out_path, frame)

    def recursive_predict_and_write(f0, f2, frame0_no, frame_diff):
        #Interpolate the middle frame
        with torch.no_grad():
            f1 = net(f0, f2)['output_im']
        offset = frame_diff//2
        #Recursively predict the extra frames
        if frame_diff > 2:
            recursive_predict_and_write(f0, f1, frame0_no, offset)
            recursive_predict_and_write(f1, f2, frame0_no+offset, offset)
        #Write the predicted frame
        write_frame(fh.tensor_to_numpy_bgr(f1[0]), frame0_no+offset)

    cap = cv2.VideoCapture(vid_in_path)
    if cap is None or not cap.isOpened():
        raise RuntimeError('Unable to open video: ', os.path.abspath(vid_in_path))
    vid_dir, vid_in_name = os.path.split(vid_in_path)
    vid_out_name = os.path.splitext(vid_in_name)[0] + '_{0}x_slow.mp4'.format(slow_factor)
    vid_out_path = vid_dir + vid_out_name
    assert(not os.path.exists(vid_out_path))

    #Total number of frames
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames > 220:
        n_frames = 220
        print("Limiting number of frames to 220. Feel free to remove.")

    real_frames = Queue()
    
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    utils.clear_dir(tmp_dir)
    fh = utils.FrameHandler(None)

    def compute_inbetween_frames(real_frames, idx):
        frame0_no, frame0 = real_frames.queue[idx]
        frame1_no, frame1 = real_frames.queue[idx+1]
        #Convert the frames to tensors with minibatch size 1
        frame0 = default_collate([frame0])
        frame1 = default_collate([frame1])
        recursive_predict_and_write(frame0, frame1, frame0_no*slow_factor, slow_factor)

    video_long_edge = max(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
                          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not torch.cuda.is_available() and video_long_edge > cpu_max_size:
        print("Because we are using CPU, all images will be resized to {0} on the long edge for speed.".format(cpu_max_size))
        scale =  cpu_max_size/video_long_edge
        
    for frame_num in tqdm(range(n_frames)):
        ret, frame = cap.read()
        
        if not torch.cuda.is_available() and video_long_edge > cpu_max_size:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            
        assert(ret)
        write_frame(frame, frame_num*slow_factor)
        real_frames.put((frame_num, fh.bgr_to_tensor(frame)))

        if real_frames.qsize() == 4:
            #We have four real frames in the queue, which lets us finetune if we want
            if continuous_fine_tuning:
                net.cft.finetune_4(real_frames)
            #Compute the intermediate frames between the two middle frames in the queue
            compute_inbetween_frames(real_frames, 1)

            #In the very beginning and end of our video, we need to predict the first and last frame pair, respectively
            if real_frames.queue[0][0] == 0:
                compute_inbetween_frames(real_frames, 0)
            if real_frames.queue[3][0] == n_frames-1:
                compute_inbetween_frames(real_frames, 2)
            real_frames.get()

    fps = cap.get(cv2.CAP_PROP_FPS)
    ffmpeg_command = 'ffmpeg -f image2 -r {fps} -i "{tmp}%09d.bmp" -c h264 -crf 17 -y "{vid_out_path}"'.format(fps=fps, tmp=tmp_dir, vid_out_path=vid_out_path)
    print('Running', ffmpeg_command)
    print("if it doesn't generate a movie you probably don't have ffmpeg installed.")
    os.system(ffmpeg_command)
        

#%%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weight", dest="weight_path", default='VFI_CFT_weights.pt.gz')
    parser.add_argument("-m", "--movie", dest="movie_path", default="rain.mp4")
    parser.add_argument("-f", "--factor", dest="slow_factor", default=2, type=int)
    parser.add_argument("-c", "--cft", dest="continuous_fine_tuning", default=False, type=bool)
    args = parser.parse_args()

    net = utils.load_model(args.weight_path, args.continuous_fine_tuning)
    SlowMovie(args.movie_path, args.slow_factor, args.continuous_fine_tuning)
