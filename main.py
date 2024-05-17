import os
import shutil
import sys
import glob
import ffmpeg
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
    os.makedirs(os.path.dirname("./data/debug"), exist_ok=True)

    src_ext = get_file_extension("./data", "src")
    dst_ext = get_file_extension("./data", "dst")

    if not src_ext:
      print(f"No src found")
      sys.exit()

    if not dst_ext:
      print(f"No dst found")
      sys.exit()

    do_edit = not args.merge_crop_only and not args.merge_only
    do_crop = not args.merge_only

    # We only support jpg and png for src
    if src_ext in ["jpg", "png"]:
      shutil.copy2("./data/src."+src_ext , "./data/src/src."+src_ext)
    else:
      print(f"src can only be jpg or png")
      sys.exit()

    # We only support jpg, png and mp4 for dst
    if dst_ext in ["jpg", "png"]:
      shutil.copy2("./data/dst."+dst_ext, "./data/dst/dst."+dst_ext)
    elif dst_ext in ["mp4"]:
      if not args.no_extract:
        # extract frames to images
        print(f"Extracting video frames..")
        input_path = "./data/dst." + dst_ext
        output_path = "./data/dst"
        ffmpeg.input(input_path)
        kwargs = {'pix_fmt': 'rgb24'}
        ffmpeg.output(str(output_path / ('%5d.png')), **kwargs )
        try:
          ffmpeg = ffmpeg.run()
        except:
          print(f"ffmpeg fail, job commandline:" + str(ffmpeg.compile()))
    else:
      print(f"dst can only be jpg, png or mp4")
      sys.exit()


    if do_crop:
      print("Requested Crop")

      # We align and crop images and put them into /data/aligned
      cropper = Cropper(face_factor=0.7, strategy="largest", output_size=args.crop_size)
      cropper.process_dir(input_dir="./data/src", output_dir="./data/src/aligned")
      cropper.process_dir(input_dir="./data/dst", output_dir="./data/dst/aligned")
      
    if do_edit:
      print("Requested Edit")

      # We make the output dirs
      # We set defaults for guided diffusion image training
      # We load checkpoint models (model.pt, arcface, faceparser and GazeEstimator)
      image_editor = ImageEditor(args)

      # This is the main function
      image_editor.edit_image_by_prompt()

    # merge
    print("Requested Merge")
    merged_image = merge_faces(args, dst_ext)

