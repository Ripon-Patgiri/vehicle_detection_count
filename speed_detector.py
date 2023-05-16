# Import required Libraries
from PIL import Image
import cv2
import numpy as np
import requests

# Fetch the image from the internet
# Resize the image and convert into a NumPy Array
image = Image.open(requests.get(
    'https://a57.foxnews.com/media.foxbusiness.com/BrightCove/854081161001/201805/2879/931/524/854081161001_5782482890001_5782477388001-vs.jpg', stream=True).raw)
image = image.resize((450, 250))
image_arr = np.array(image)
image

# Save the image
Image.fromarray(image_arr).save('car_image.jpg')

# To get better output perform transformations on image

# convert image to grayscale
grey = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
Image.fromarray(grey).save('car_image_grayscale.jpg')

# Apply GaussianBlur to remove noise from image
blur = cv2.GaussianBlur(grey, (5, 5), 0)
Image.fromarray(blur).save('car_image_gaussiabblur.jpg')

# Dilate the images to fill the missing parts of the images
dilated = cv2.dilate(blur, np.ones((3, 3)))
Image.fromarray(dilated).save('car_image_dilation.jpg')

# Perform Morphology Transformation with the kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
Image.fromarray(closing).save('car_image_morph.jpg')

# Detecting cars using car cascade
car_cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_src)
cars = car_cascade.detectMultiScale(closing, 1.1, 1)
cars

# Draw rectange around detected cars
cnt = 0
for (x, y, w, h) in cars:
    cv2.rectangle(image_arr, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cnt += 1
print(cnt, " cars found")
Image.fromarray(image_arr)
