import dlib
import cv2
import json
import piexif
import numpy as np

def merge_faces(extension):
  output = "./data/result.png"
  # Load the images
  image1 = cv2.imread("./data/dst." + extension) # the source full image
  image2 = cv2.imread("./data/pred.png", cv2.IMREAD_UNCHANGED)

  # %% Read in exif data
  exif_dict = piexif.load("./data/dst/aligned/dst.jpg")
  # Extract the serialized data
  user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
  # Deserialize
  d = json.loads(user_comment)
  print(d)

  # Get the inverted transform array
  # Deserialize the JSON string to a nested Python list
  inverse = json.loads(d["inverse"])
  # Convert the transform_matrix_list back to a NumPy array
  inverse = np.array(inverse)

  height, width, _ = image1.shape
  fac = width / d["parentw"]
  # Define the region where image2 will be placed
  x_offset = round(d["x"]) # X-coordinate offset
  y_offset = round(d["y"]) # Y-coordinate offset
  img2_height = round(d["h"] * fac)
  img2_width = round(d["w"] * fac)

  x_end = x_offset + img2_width
  y_end = y_offset + img2_height

  # Resize image2
  image2 = cv2.resize(image2, (img2_width, img2_height))

  # Apply the inverse transformation to the transformed image
  image2 = cv2.warpAffine(
      image2,
      inverse,
      (image2.shape[1], image2.shape[0]),
  )

  # Resize image2
  #image2 = cv2.resize(image2, (img2_width, img2_height))

  image1[y_offset:y_end, x_offset:x_end] = image2

  cv2.imwrite(output, image1)
  return image1