# Real-Time Interactive Object Segmentation System with Voice Control

## Description

This repository contains the code and implementation for a real-time object segmentation system that leverages voice commands and natural language descriptions to enhance object detection and segmentation in video streams.
The system integrates both standard webcams and Intel RealSense cameras, combining YOLO-based object detection with vision-language models such as FastSAM.
Users can interact with the system via voice commands to control segmentation tasks, switch between detection models, and refine segmentation results.

## Features

* **Real-Time Object Segmentation:** Segment objects in live video streams from a webcam or Intel RealSense camera.
* **Voice Control:** Users can interact with the system via predefined voice commands for hands-free operation.
* **Multi-Model Architecture:** Combines YOLO for fast, predefined object detection with vision-language models (FastSAM) for generalizing to more complex object categories.
* **Switchable Detection Modes:** Allows users to switch between YOLO and FastSAM when more detailed or flexible segmentation is required.

## Setup and Installation

1. Clone Repository
2. Install requirements.txt with

   ```pip install -r requirements.txt```

## Usage

1. **Run the application:** ```python main_yolo_fast_sam.py```
2. **Control the System:**
   * **Keyword:** Set as "Hello", after which the system listens for the command
   * **Switching Models:** Use to command "model number one" to switch to YOLO, and command "model number two" to use FastSAM
   * **Segmenting Object:** After ensuring the keyword was detected, describe the object to be segmented. The model will refine the mask to just segment the described object
3. **Closing the System:** Press key q on the keyboard to close the system

## Possibile Modifications
* **Speech-to-text model:** Currently a pre-trained Vosk model for Indian accent english is used. This can be changed to suit the user's requirement.
* **Keyboard input:** The method ```user_input()``` can be used for keyboard input instead of voice input. Replace the t2 thread target in the ```__init__``` method as ```self.t2 = threading.Thread(target=self.user_input, daemon=True)```

## Future Improvements
* Add functionality to save frames as annotated data after applying FastSAM model with user input
* Add functionality to retrain YOLO model on the collected annotated data which adds a new object category to predict
* Combine SAM and CLIP encoders for faster processing of frames
* Use multithreading for seperate threads for I/O operations, that is, capturing frame, segmenting frame, displaying frame. Currently two threads use for microphone and video capturing respectively
* Add more voice commands, for example to start and stop the system

## License
This project is licensed under the MIT License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
