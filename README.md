# GauGau_yolov3_realtime

## Download 

yolov3.weights : https://pjreddie.com/media/files/yolov3.weights
yolov3-tiny.weights : https://pjreddie.com/media/files/yolov3-tiny.weights

## Create some files:
yolov3.cfg: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
yolov3-tiny.cfg: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg
coco.names: https://github.com/muhammadshiraz/YOLO-Real-Time-Object-Detection/blob/master/coco.names

## Initialize virtual environment on the commmand line
Create virtual envirnoment
```
py -m venv .venv
```
Upgarde pip
```
py -m pip install --upgrade pip
```
Set permission for program
```
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser
```
Install packages from requirements.txt
```
pip install -r .\requirements.txt
```
Freeze 
```
python -m pip freeze > requirements.txt
```

## Some problems:
* Problem 1:  self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
IndexError: invalid index to scalar variable.

getUnconnectedOutLayers() returns an integer, not an iterable. Instead, use
outputlayers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
## Reference:
https://github.com/muhammadshiraz/YOLO-Real-Time-Object-Detection