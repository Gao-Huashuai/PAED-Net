# PAEDNet

This is an experimental **T**ensor **F**low implementation of **PAEDNet** for panorama object detection, mainly based on the work of [TFFRCNN](https://github.com/CharlesShang/TFFRCNN) 

### Installation: 
For detailed installation, please refer to https://github.com/CharlesShang/TFFRCNN

### Demo

After successfully completing, you'll be ready to run the demo.
The demo samples are from WHU panoramic dataset and the download link is http://study.rsgis.whu.edu.cn/pages/download/

To run the demo
```Shell
cd $PAEDNet
python ./faster_rcnn/demo.py
```
The detection results of panoramic images will be saved in path (`./data/detection_result`).

### Evaluation
```Shell
cd $PAEDNet
python ./faster_rcnn/test_net.py
```
The download data can be put into `./data/panoramic/`
The evaluation model was trained WHU panoramic dataset with 30000 iteration steps, which can be downloaded at https://pan.baidu.com/s/1j4-WmYAYLEbV0TyICmfPEQ




