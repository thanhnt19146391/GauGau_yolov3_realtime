# GauGau_yolov3_realtime
## Installations
```
 pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

```

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
## Progress
Yolov5


## Some problems:
* Problem 1:  

self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

IndexError: invalid index to scalar variable.

Solution: getUnconnectedOutLayers() returns an integer, not an iterable. Instead, use

outputlayers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

* Problem 2: Also slowly 

Solution: build custom model with less object (person .etc)

* Problem 3: ERROR: image auto rotate when convert png to jpg

Solution: 

If an image has an EXIF Orientation tag, other than 1, return a new image that is transposed accordingly. The new image will have the orientation data removed. Otherwise, return a copy of the image.

```
im = ImageOps.exif_transpose(im)
```

https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.exif_transpose

## Reference:

https://github.com/muhammadshiraz/YOLO-Real-Time-Object-Detection

https://github.com/NSTiwari/YOLOv3-Custom-Object-Detection

https://github.com/ultralytics/yolov5