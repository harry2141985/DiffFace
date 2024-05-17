import dlib
import os
import cv2
import json
import piexif
import numpy as np

def merge_faces(args, extension):
  input_path = "./data/dst/preded/"
  output_path = "./data/dst/merged/"
  reference_path = "./data/dst/"
  aligned_path = "./data/dst/aligned/"

  # for each image in input path
  for filename in os.listdir(input_path):
    file_path = os.path.join(input_path, filename)
    if os.path.isfile(file_path):
      image1 = os.path.join(reference_path, filename) # the dst full image
      image1 = cv2.imread(image1)
      image2 = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

      # %% Read in exif data
      aligned_image = os.path.join(aligned_path, filename)
      exif_dict = piexif.load(aligned_image)
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

      cv2.imwrite(os.path.join(output_path, filename), result)