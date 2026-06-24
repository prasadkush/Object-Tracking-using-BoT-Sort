# BoT-SORT



## Highlights 🚀


## Coming Soon
## Abstract




### Object Tracking of fruits 

[Watch the demo](https://drive.google.com/file/d/1z1sPhgd9UmggBCXD4KC3Jut7bXOSv7je/view?usp=drive_link)


## Installation

The code was tested on Ubuntu 22.04

BoT-SORT code is based on ByteTrack and FastReID. <br>
Visit their installation guides for more setup options.
 


## Data Preparation


## Training

## Demo

Demo with BoT-SORT(-ReID) based YOLOX and multi-class.

```shell
cd <BoT-SORT_dir>
``

Demo with BoT-SORT(-ReID) based YOLOv7 and multi-class.
```shell
cd <BoT-SORT_dir>
python3 tools/mc_demo_yolov7.py --weights pretrained/yolov7-d6.pt --source <path_to_video/images> --fuse-score --agnostic-nms (--with-reid)
```


## Citation

```
@article{aharon2022bot,
  title={BoT-SORT: Robust Associations Multi-Pedestrian Tracking},
  author={Aharon, Nir and Orfaig, Roy and Bobrovsky, Ben-Zion},
  journal={arXiv preprint arXiv:2206.14651},
  year={2022}
}
```


## Acknowledgement

A large part of the codes, ideas and results are borrowed from 
[ByteTrack](https://github.com/ifzhang/ByteTrack), 
[StrongSORT](https://github.com/dyhBUPT/StrongSORT),
[FastReID](https://github.com/JDAI-CV/fast-reid),
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and
[YOLOv7](https://github.com/wongkinyiu/yolov7). 
Thanks for their excellent work!











