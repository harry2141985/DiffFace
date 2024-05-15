import dlib
import cv2
import json
import piexif
import numpy as np

def merge_faces(args, extension):
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
  pheight = d["parenth"]
  pwidth = d["parentw"]

  # TODO: Remove this code
  #fac = width / d["parentw"]
  # Define the region where image2 will be placed
  #x_offset = round(d["x"]) # X-coordinate offset
  #y_offset = round(d["y"]) # Y-coordinate offset
  #img2_height = round(d["h"] * fac)
  #img2_width = round(d["w"] * fac)
  #x_end = x_offset + img2_width
  #y_end = y_offset + img2_height

  # Apply the inverse transformation to the transformed image
  image2 = cv2.warpAffine(
    image2,
    inverse,
    (pwidth, pheight),
    borderMode=0
  )

  # Resize image2 to dst dimentions
  image2 = cv2.resize(image2, (width, height))

  # Create a mask based on black pixels of image2
  mask = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)[1]

  # Expand the mask to cover potential border
  kernel = np.ones((20, 20), np.uint8)
  mask = cv2.dilate(mask, kernel, iterations=1)

  # Invert the mask to make black pixels transparent
  mask_inv = cv2.bitwise_not(mask)

  # Apply the mask to the images
  image1_part = cv2.copyTo(image1, mask)
  image2_part = cv2.copyTo(image2, mask_inv)

  # Blend the two images together
  result = cv2.add(image1_part, image2_part)

  cv2.imwrite(output, result)
  return image1