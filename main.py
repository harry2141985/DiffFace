import os
import shutil
import sys
import glob
from optimization.image_editor import ImageEditor
from optimization.arguments import get_arguments
from face_crop_plus.cropper import Cropper
from optimization.merge import merge_faces


def get_file_extension(directory, filename):
    file_pattern = f"{directory}/{filename}.*"
    files = glob.glob(file_pattern)
    
    if files:
        _, ext = os.path.splitext(files[0])
        return ext[1:]  # Remove the dot from the extension
    
    return None


if __name__ == "__main__":
    
    # we get the arguments collected in optimization/arguments.py
    args = get_arguments()

    os.makedirs(os.path.dirname("./data/src/aligned"), exist_ok=True)
    os.makedirs(os.path.dirname("./data/dst/aligned"), exist_ok=True)

    src_ext = get_file_extension("./data", "src")
    dst_ext = get_file_extension("./data", "dst")

    if not src_ext:
      print(f"No src found")
      sys.exit()

    if not dst_ext:
      print(f"No dst found")
      sys.exit()

    # We currently only support images (jpg, png)
    shutil.copy2("./data/src."+src_ext , "./data/src/src."+src_ext)
    shutil.copy2("./data/dst."+dst_ext, "./data/dst/dst."+dst_ext)

    # We align and crop images and put them into /data/aligned
    cropper = Cropper(face_factor=0.7, strategy="largest", output_size=args.crop_size)
    cropper.process_dir(input_dir="./data/src", output_dir="./data/src/aligned")
    cropper.process_dir(input_dir="./data/dst", output_dir="./data/dst/aligned")
    
    # We make the output dirs
    # We set defaults for guided diffusion image training
    # We load checkpoint models (model.pt, arcface, faceparser and GazeEstimator)
    #image_editor = ImageEditor(args)

    # This is the main function
    #image_editor.edit_image_by_prompt()

    # merge
    merged_image = merge_faces(dst_ext)

