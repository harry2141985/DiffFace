from optimization.image_editor import ImageEditor
from optimization.arguments import get_arguments
from face_crop_plus import Cropper
from optimization.merge import merge_faces

if __name__ == "__main__":
    
    # we get the arguments collected in optimization/arguments.py
    args = get_arguments()

    # We align and crop images and put them into /data/aligned
    cropper = Cropper(face_factor=0.7, strategy="largest", output_size=args.crop_size)
    cropper.process_dir(input_dir="./data/src", output_dir="./data/aligned/src")
    cropper.process_dir(input_dir="./data/targ", output_dir="./data/aligned/targ")
    
    # We make the output dirs
    # We set defaults for guided diffusion image training
    # We load checkpoint models (model.pt, arcface, faceparser and GazeEstimator)
    image_editor = ImageEditor(args)

    # This is the main function
    image_editor.edit_image_by_prompt()

    # merge
    merged_image = merge_faces("./data/targ/001.png", "./output/0_0.png", "./output/result.png")

