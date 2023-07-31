import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import tensorflow as tf
import streamlit as st
from PIL import Image
from flask import Flask, Response
# image_path = 'room.jpg'
def detect_objects(image_data, min_confidence=0.2):
    prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
    model_path = 'models/MobileNetSSD_deploy.caffemodel'

    min_confidence = 0.2

    classes = ["background","aeroplane", "bicycle", "bird", "boat", "bottle","bus", "car", "cat","chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa","train", "tvmonitor"]

    #colors for rectangle object detector
    np.random.seed(543210)
    colors = np.random.uniform(0,255, size=(len(classes),3)) #get colors of each guess

    #cv2.dnn: This is the module within OpenCV that provides deep neural network functionalities.
    #readNetFromCaffe(prototxt_path, model_path): This function is used to read a deep learning model from Caffe framework.
    #Caffe is a deep learning framework developed by the Berkeley Vision and Learning Center (BVLC) and is widely used for image classification, object detection, and other computer vision tasks.
    #prototxt_path: This is the file path to the "deploy" prototxt file of the Caffe model. The deploy prototxt file defines the network architecture.
    #model_path: This is the file path to the pre-trained Caffe model. The model file contains the learned weights of the network.

    net = cv2.dnn.readNetFromCaffe(prototxt_path,model_path)

    # image = cv2.imread(image_path)
    image = np.array(image_data)
    try:
        height = image.shape[0]
        width = image.shape[1]
        # print(image.shape[2]) #shows the colors if is colored image then is 3 if is grayscale then 1

        ##RESIZING THE IMAGE
        blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),0.007,(300,300), 130)

        ##PREDICTION
        net.setInput(blob)
        detected_object = net.forward()
        # print(detected_object[0][0][0]) #data of the first object detected, detected_object[0][0][1] is the second etc
        # print(detected_object.shape[3])

        thickness_scale = image.shape[0] / 1000  # Scale the thickness based on the height
        font_scale = 2.0 * thickness_scale  # Scale the font size based on the thickness
        text_position_scale = image.shape[0] / 1000  # Scale the text position based on the height

        thickness = 4
        font_scale = 1
        ##CHECK ALL THE PREDICTIONS -> CHOSE THE ONES WITH GOOD PREDICTION VALUE -> SET THE VISUALISATION
        for i in range(detected_object.shape[2]):
            #how confident is the model about the object that detected in the img
            confidence = detected_object[0][0][i][2] #eg detected_object[0][0][2][2] get the confidence of the 3rd object detection

            if confidence > min_confidence:
                class_index = int(detected_object[0, 0, i, 1]) #give wat class was the object eg 1 -> aeroplane
                ##VISUALASATION
                #We take the position of the project detected from detected_object[0, 0, i, 3-6]
                # eg. [ 0.         18.          0.99582505  0.15535903  0.489421    0.7193363  0.985309  ]
                upper_left_x = int(detected_object[0, 0, i, 3] * width)
                upper_left_y = int(detected_object[0, 0, i, 4] * height)
                lower_right_x = int(detected_object[0, 0, i, 5] * width)
                lower_right_y = int(detected_object[0, 0, i,6] * height)

                prediction_text =f"{classes[class_index]}: {int(confidence*100)}%"
                # Calculate the scaled thickness
                scaled_thickness = int(thickness * thickness_scale)
                # Calculate the scaled font size
                scaled_font_size = font_scale * thickness_scale
                # 15 is the distance from the box for the text
                scaled_text_position = int(15 * text_position_scale)

                cv2.rectangle(image,(upper_left_x,upper_left_y),(lower_right_x,lower_right_y),colors[class_index],scaled_thickness)
                if upper_left_y > 30:
                    cv2.putText(image, prediction_text,(upper_left_x,upper_left_y - scaled_text_position), cv2.FONT_HERSHEY_SIMPLEX, scaled_font_size, colors[class_index],scaled_thickness)
                else:
                    cv2.putText(image, prediction_text, (upper_left_x, upper_left_y + scaled_text_position),cv2.FONT_HERSHEY_SIMPLEX, scaled_font_size, colors[class_index],scaled_thickness)
        # resized_image = cv2.resize(image, (1500, 1000))
        desired_width = 1500
        aspect_ratio = image.shape[1] / image.shape[0]
        desired_height = int(desired_width / aspect_ratio)

        # resized_image = cv2.resize(image, (desired_width, desired_height))
        # cv2.imshow(f"Detected Objects from the image named {image_path}", resized_image)
        # cv2.waitKey(0) #close the picture with any key
        # cv2.destroyAllWindows() #destroy all the related windows

        return image
    except Exception as e:
        st.error(f"error during object detection {e}")
        return None

# Streamlit App
def main():
    st.set_page_config(layout="wide")  # Set the layout to wide
    st.title("Object Detection with Streamlit 🕵️")
    st.write("Upload an image and see the detected objects")

    # Explanation section
    st.markdown("""
    ### How It Works

    This is an object detection application using a pre-trained MobileNetSSD model. The model has been trained to detect
    20 different objects, including aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dining table, dog,
    horse, motorbike, person, potted plant, sheep, sofa, train, and TV monitor.

    When you upload an image, the application will run the object detection model on the image and display the original
    image along with bounding boxes around the detected objects and their confidence scores.

    ### Note
    Please make sure to upload an image file (e.g., JPG, JPEG, PNG) for the application to work correctly.

    """)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Perform object detection on the uploaded image
            image = Image.open(uploaded_file)
            detected_image = detect_objects(image)
            if detected_image is not None:
                st.image(image, caption='Original Image', use_column_width=True)
                st.image(detected_image, caption='Detected Image', use_column_width=True)
        except Exception as e:
            st.error(f"Error during object detection: {e}")

if __name__ == "__main__":
     main()


