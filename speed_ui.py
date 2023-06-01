import streamlit as st
from PIL import Image
import cv2
import numpy as np
import requests


def detect_cars(image):
    image_arr = np.array(image)
    gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dilated = cv2.dilate(blur, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    car_cascade_src = 'cars.xml'
    car_cascade = cv2.CascadeClassifier(car_cascade_src)
    cars = car_cascade.detectMultiScale(closing, 1.1, 1)

    cnt = 0
    for (x, y, w, h) in cars:
        cv2.rectangle(image_arr, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cnt += 1

    return cnt, image_arr


def main():
    st.title("Car Detection and Counter")

    option = st.sidebar.selectbox(
        "Select Input Method", ("Image URL", "Upload Image"))

    if option == "Image URL":
        image_url = st.text_input("Enter the URL of the image:")
        if image_url:
            image = Image.open(requests.get(image_url, stream=True).raw)
            image = image.resize((450, 250))
            car_count, annotated_image = detect_cars(image)

            st.image(image, caption='Original Image', use_column_width=True)
            st.image(annotated_image, caption='Annotated Image',
                     use_column_width=True)
            st.write("Number of cars detected:", car_count)

    elif option == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = image.resize((450, 250))
            car_count, annotated_image = detect_cars(image)

            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.image(annotated_image, caption='Annotated Image',
                     use_column_width=True)
            st.write("Number of cars detected:", car_count)


if __name__ == '__main__':
    main()
