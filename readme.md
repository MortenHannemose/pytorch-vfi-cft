# pytorch-vfi-cft
Want to convert your video to slowmotion? Now you can!

This method generates extra frames, so you can convert an existing video to a higher framerate.

The method uses CNNs (convolutional neural networks), so we recommend running in on a GPU.

---

This is a reference implementation of [Video Frame Interpolation via Cyclic Fine-Tuning and Asymmetric Reverse Flow](http://www.imm.dtu.dk/~jerf/papers/vfi_cft_arf.pdf). 

If you use our work please cite the paper:

    @inproceedings{hannemose2019video,
      title={Video Frame Interpolation via Cyclic Fine-Tuning and Asymmetric Reverse Flow},
      author={Hannemose, Morten and Jensen, Janus N{\o}rtoft and Einarsson, Gudmundur and Wilm, Jakob and Dahl, Anders Bjorholm and Frisvad, Jeppe Revall},
      booktitle={Scandinavian Conference on Image Analysis},
      pages={311--323},
      year={2019},
      organization={Springer}
    }
    
## Example result
<img src='./misc/soapbox_comparison.gif' alt="gif showing an example result">
   
## Usage
To convert a video to slowmotion use `slow-movie.py`

Example to convert `rain.mp4` to 4x slowmotion:

	python slow_movie.py -m rain.mp4 -f 4
    
This will output the movie as `bmp` files and put them in the folder `slowed_movie_frames`.
To convert the generated frames into a video you must have `ffmpeg` installed. [Instructions here](https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg).

## Pretrained model
You can download our trained model from [http://people.compute.dtu.dk/mohan/vfi-cft/VFI_CFT_weights.pt.gz](http://people.compute.dtu.dk/mohan/vfi-cft/VFI_CFT_weights.pt.gz).

This file should be placed in the root of the repository.

## Interpolation from two images
To interpolate the middle frame from only two frames, please see `simple_example.py`.
This is also a good starting ground for modifying our code.

## Requirements
The code is tested under:
* Python 3.6
* pytorch 1.1.0

It will most likely work with other versions, but we have not tested it.

## Issues
This repository is actively maintained, so feel free to open an issue if you run into problems.