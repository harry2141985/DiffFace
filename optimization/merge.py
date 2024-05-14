import dlib
import cv2
import json
import piexif
import numpy as np

def merge_faces(image1_path, image2_path, output="./data/result.jpg"):
  # Load the images
  image1 = cv2.imread("./data/targ/001.png")
  image2 = cv2.imread("./output/0_2.png", cv2.IMREAD_UNCHANGED)

  # %% Read in exif data
  exif_dict = piexif.load("./data/aligned/targ/001.jpg")
  # Extract the serialized data
  user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
  # Deserialize
  d = json.loads(user_comment)
  
  height, width, _ = image1.shape

  fac = width / d["parentw"]
  # Define the region where image2 will be placed
  x_offset = round(d["x"] * fac) # X-coordinate offset
  y_offset = round(d["y"] * fac) # Y-coordinate offset
  img2_height = round(d["h"] * fac)
  img2_width = round(d["w"] * fac)

  x_end = x_offset + img2_width
  y_end = y_offset + img2_height

  # Resize image2
  image2 = cv2.resize(image2, (img2_width, img2_height))

  image1[y_offset:y_end, x_offset:x_end] = image2

  cv2.imwrite(output, image1)
  return image1