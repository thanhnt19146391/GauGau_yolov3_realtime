import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

class ThanhNT_GUI(tk.Tk):
    def __init__(self, frames = None, *arg, **kwargs):
        """ Backend """
        self.load_yolo()

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

    def load_yolo(self):
        # Load Yolo
        self.net = cv2.dnn.readNet("weights/yolov3_training_1000.weights", "cfg/yolov3_testing.cfg")
        self.classes = []
        with open("classes.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()

        # ?
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # Random bounding color for number of classes
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect_objects(self, src):
        frame = src.copy()

        height, width, channels = frame.shape
        
        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        # blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

         # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[3] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 1.8)
                    y = int(center_y - h / 1.8)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(
                    img = frame, 
                    text = label + " " + str(round(confidence, 2)), 
                    org = (x, y + 30), 
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
        dst_frame = self.detect_objects(src = frame)
        # dst_frame = frame
        
        self.photoImage = self.createPhotoImage(dst_frame)
        self.frame_label.config(image = self.photoImage)
        self.after(ms = 42, func = self.update_frame)

    


def ThanhNT_main():
    """
    ThanhNT: load video and detect realtime
    """    
  
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
    """

    app = ThanhNT_GUI(frames = None)
    app.mainloop()
    

    

if __name__ == '__main__':
    ThanhNT_main()
   

    