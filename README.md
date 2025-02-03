## ByteTracker re-implement

1. first download yolox weights with the Introduction of [YOLOX](README_YOLOX.md)

2. use following command to run tracker 
```
python ./track/main.py -n [model_name] -v [video_path] -p [ckpt path] --lconf 0.25 -hconf 0.35
```
the meaning of parameters :
```
options:
  -h, --help               show this help message and exit
  --name NAME, -n NAME     model_name
  --video VIDEO, -v VIDEO  video_path
  --ckpt CKPT, -p CKPT     ckpt_path
  --experiment_name EXPERIMENT_NAME
  --lconf LCONF            lower conf
  --hconf HCONF            higher conf
  --nms NMS                test nms threshold
  --tsize TSIZE            test img size
```

## TODO List 
- [x] base function of byte tracker
- [] kalman filter 
