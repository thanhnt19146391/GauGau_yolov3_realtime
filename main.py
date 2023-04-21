import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch

class ThanhNT_GUI(tk.Tk):
    def __init__(self, frames = None, *arg, **kwargs):
        """ Backend """
        # self.load_yolov3()
        self.load_yolov5()

        self.frames = frames

        if frames == None:
            self.vid = cv2.VideoCapture(0)
        else:
            self.max_index = len(frames)    
            self.frame_index = -1

        """ Frontend """
        # Init Tk
        tk.Tk.__init__(self, *arg, **kwargs)

        # Configure window size as maximum
        self.state('zoomed')

        self.frame_label = tk.Label(
            master = self
        )
        self.frame_label.grid(column = 0, row = 0)

        self.update_frame()

    def load_yolov5(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', force_reload=True)  # yolov5n - yolov5x6 or custom

    def detect_objects_yolov5(self, src):
        frame = src.copy()

        results = self.model(frame)  # inference
        results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
        df = results.pandas().xyxy[0]
        print(df)
        print('-------------------------------------------')

        for i in range(df.shape[0]):
            xmin, ymin, xmax, ymax, confidence, num_class, name = df.iloc[i]
            cv2.rectangle(
                img = frame, 
                pt1 = (round(xmin), round(ymin)),
                pt2 = (round(xmax), round(ymax)),
                color = (255, 255, 255),
                thickness = 2
            )
            cv2.putText(
                    img = frame, 
                    text = f'{name} ({round(confidence * 100)}%)', 
                    org = (round(xmin) + 5, round(ymin) + 25), 
                    fontFace = cv2.FONT_HERSHEY_PLAIN, 
                    fontScale = 2,
                    color = (255, 255, 255), 
                    thickness = 2)

        return frame

    def createPhotoImage(self, mat):
        img = Image.fromarray(mat)
        w, h = img.size
        dst = ImageTk.PhotoImage(image = img)
        return dst

        
    def update_frame(self):
        if self.frames == None:
            ret, frame = self.vid.read()
            if ret: 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return
        else:
            self.frame_index = (self.frame_index + 1) % self.max_index
            # self.frame_index = (self.frame_index + 1) % 10
            print(f'Frame {self.frame_index}')
            frame = self.frames[self.frame_index]

        dst_frame = self.detect_objects_yolov5(src = frame)
        # dst_frame = frame
        
        self.photoImage = self.createPhotoImage(dst_frame)
        self.frame_label.config(image = self.photoImage)
        self.after(ms = 42, func = self.update_frame)

    


def ThanhNT_main():
    """
    ThanhNT: load video and detect realtime
    """    
  
    
    # define our new video
    video_filename = 'elderly.mp4'

    cap = cv2.VideoCapture(video_filename)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret: 
           frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           frames.append(frame) 
        else:
            break

    print(f'Number of frame: {len(frames)}')
    

    app = ThanhNT_GUI(frames = None)
    app.mainloop()
    

    
if __name__ == '__main__':
    ThanhNT_main()
   

    