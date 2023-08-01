# Object Detection with Streamlit 🕵️

![Object Detection Demo](demo.gif)

## Overview

This is an object detection application built with Streamlit and powered by a pre-trained MobileNetSSD model. The model has been trained to detect 20 different objects, including aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dining table, dog, horse, motorbike, person, potted plant, sheep, sofa, train, and TV monitor.

## How It Works

1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Run the Streamlit app with `streamlit run Object_detection.py`.
4. The app will launch in your default web browser.
5. Upload an image file (e.g., JPG, JPEG, PNG) using the file uploader in the app.
6. The object detection model will process the image and display the original image along with bounding boxes around the detected objects and their confidence scores.

## Requirements

- Python 3.7 or higher
- Streamlit 1.0.0 or higher
- OpenCV (cv2) package
- TensorFlow 2.0 or higher

## Demo

A live demo of the app is available [here](https://objectdetection-wseat6jukwzkukaqepxcte.streamlit.app/).

## Note

Please make sure to upload an image file for the application to work correctly.

## Credits

The MobileNetSSD model used in this app is based on the original [Caffe implementation](https://github.com/chuanqi305/MobileNet-SSD) by chuanqi305
