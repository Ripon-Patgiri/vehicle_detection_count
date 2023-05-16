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
