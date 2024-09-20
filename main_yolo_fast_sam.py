from ultralytics import YOLO, FastSAM
import pyrealsense2 as rs
import cv2
import math
import time
import torch  

from ultralytics.utils.plotting import Annotator, colors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import threading


import os
import sys
import wave
import json
import pyaudio
from vosk import Model, KaldiRecognizer

class VideoSegmentation():
    def __init__(self):
        self.seg_object = None
        self.device = "0"
        self.read_lock = threading.Lock()
        self.curr_model = "YOLO"

        #Set frame times and font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fps_curr = 0
        self.fps_avg = 0
        self.f_count = 0
        self.prev_frame_time = time.time()
        self.new_frame_time = 0
        self.prev_frame_time_avg = time.time()
        self.new_frame_time_avg = 0

        #create threads
        self.t1 = threading.Thread(target=self.webcam_video)
        self.t2 = threading.Thread(target=self.voice_input, daemon=True)

        #start threads
        self.t1.start()
        self.t2.start()

        #Objects to detect
        self.obj_list = ["bottle","book", "cup", "person"]

        #Load yolo model
        self.model = YOLO("yolo-Weights/yolov8n-seg.pt")
        self.model_sam = FastSAM("FastSAM-s.pt")
        self.names = self.model.model.names

    def setup_webcam(self):
         # Start video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.fps_cam =  self.cap.get(cv2.CAP_PROP_FPS)

    def setup_realsense(self):
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        # device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline = pipeline
        self.pipeline.start(config)

    def webcam_video(self):

        #Using webcam setup
        # self.setup_webcam()

        #Using Intel Realsense
        self.setup_realsense()

        #record video of output
        # result = cv2.VideoWriter('filename.avi',  
        #                  cv2.VideoWriter_fourcc(*'MJPG'), 
        #                  10, (640,480)) 


        while True:
            self.f_count += 1

            # Read image from webcam
            # success, img = self.cap.read()


            #Read image from realsense
            # Wait for a coherent pair of color frames
            frames = self.pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            img = np.asanyarray(color_frame.get_data())

            with self.read_lock:
                # Check which model to use
                if self.curr_model == "YOLO":
                    self.segment_YOLO(img)
                elif self.curr_model == "FASTSAM":
                    self.segment_FASTSAM(img)
                
            #     # object details
                # Calculate fps
                self.new_frame_time = time.time()
                fps = 1/(self.new_frame_time-self.prev_frame_time) 
                self.prev_frame_time = self.new_frame_time
                fps = int(fps) 
                fps = str(fps)
                self.fps_curr = fps
                #Put fps on the frame
                cv2.putText(img, "FPS: " + fps, (7, 30), self.font, 1, (100, 255, 0), 2, cv2.LINE_AA) 


            # Check if user input and paste on frame accordingly
            with self.read_lock:
                if self.seg_object is not None:
                    cv2.putText(img, self.seg_object, (7, 400), self.font, 1, (100, 255, 0), 2, cv2.LINE_AA)
                else:
                    pass

            #Show Image
            # result.write(img)
            cv2.namedWindow('Camera', cv2.WINDOW_KEEPRATIO) 
            cv2.imshow('Camera', img)
                
            if cv2.waitKey(1) == ord('q'):
                self.new_frame_time_avg = time.time()
                self.fps_avg = self.f_count/(self.new_frame_time_avg-self.prev_frame_time_avg)
                break

    def user_input(self):
        while True:
            user_input = input("Enter Object to segment:")
            # with self.read_lock:
            #     self.seg_object = user_input
            self.execute_command(user_input)

    def voice_input(self):
        #Implement keyword
        keyword = "hello"

        model = Model(model_name="vosk-model-small-en-in-0.4")
        recognizer = KaldiRecognizer(model, 16000)

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
        stream.start_stream()
        try:
            while True:
                print("Listening for keyword...")
                words = self.recognize_speech_from_mic(recognizer, stream)
                print(f"You said: {words}")
                if keyword in words:
                    print(f"Keyword detected: {keyword}")
                    print("Listening for command...")

                    # Listen to command
                    command = self.recognize_speech_from_mic(recognizer, stream)
                    if command:
                        print(f"You said: {command}")
                        self.execute_command(command)
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def recognize_speech_from_mic(self,recognizer, stream):
        while True:
            data = stream.read(4000, exception_on_overflow = False)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                return json.loads(result)["text"]
        return ""

    def stop(self):
        self.t1.join()
        #Using webcam
        # self.cap.release()

        #Using realsense
        self.pipeline.stop()

        cv2.destroyAllWindows()

    def execute_command(self, command):
        if command == "model number one":
            with self.read_lock:
                self.curr_model = "YOLO"
        elif command =="model number two":
            with self.read_lock:
                self.curr_model = "FASTSAM"
        else:
            with self.read_lock:
                self.seg_object = command

    def segment_YOLO(self, img):
        results = None
        if self.seg_object == "reset":
                    self.seg_object == None
                    # print(f"Object ----- {self.seg_object}")
                    results = self.model.predict(img, verbose=False, stream=True, device=self.device)
        elif self.seg_object is not None:
            #Get ID of object
            try:
                cls_id = list(self.names.values()).index(self.seg_object)
                results = self.model.predict(img, classes=cls_id, verbose=False, stream=True, device=self.device)
            except:
                results = self.model.predict(img, verbose=False,stream=True, device=self.device)
                # print("The object is not in scope")
                pass

        else:
            results = self.model.predict(img, verbose=False, stream=True, device=self.device)
        
        # Plot YOLO results on frame
        annotator = Annotator(img, line_width=2)
        for r in results:
            if r.masks is not None:
                clss = r.boxes.cls.cpu().tolist()
                masks = r.masks.xy
                for mask, cls in zip(masks, clss):
                    annotator.seg_bbox(mask=mask, mask_color=colors(int(cls), True), label=self.names[int(cls)])


    def segment_FASTSAM(self,img):
        if self.seg_object is not None:
            results = self.model_sam.predict(img, texts=self.seg_object, verbose=False, conf=0.6, stream=True, device=self.device)
        else:
            results = self.model_sam.predict(img, verbose=False, conf=0.6, stream=True, device=self.device)
        # img_array = results[0].plot(boxes=False)
        # img =  img_array
        annotator = Annotator(img, line_width=2)
        
        for r in results:
            if r.masks is not None:
                masks = r.masks.xy
                for mask in masks:
                    annotator.seg_bbox(mask=mask)
        

#Start function

if __name__ == "__main__":
    vid_seg =  VideoSegmentation()
    vid_seg.stop()

    print(f"Last FPS is - {vid_seg.fps_curr}")
    print(f"Average FPS is - {vid_seg.fps_avg}")
    print(f"Cuda device count {torch.cuda.current_device()}")
