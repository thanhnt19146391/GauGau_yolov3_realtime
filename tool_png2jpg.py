import glob
from PIL import Image, ImageOps

import os

folder_path = 'D:\\GauGau\\GauGau_yolov3_realtime\\dataset\\human_detection_dataset\\1\\*.png'

new_folder_path = 'D:\\GauGau\\GauGau_yolov3_realtime\\dataset\\human_detection_dataset\\images\\'

# Count number of file in new folder
print(os.path.join(new_folder_path, '*.jpg'))
cnt = len(glob.glob(os.path.join(new_folder_path, '*.jpg')))
print(f'Number of *.jpg: {cnt}')

for path in glob.glob(folder_path):
    file_name = path.split('\\')[-1]
    file_index = file_name.split('.')[0] 
    file_path = new_folder_path + file_index + '.jpg'
    print(file_path)
    im = Image.open(path)
    
    im = ImageOps.exif_transpose(im)

    rgb_im = im.convert('RGB')
    rgb_im.save(file_path)
