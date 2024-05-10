from optimization.image_editor import ImageEditor
from optimization.arguments import get_arguments

if __name__ == "__main__":
    
    # we get the arguments collected in optimization/arguments.py
    args = get_arguments()
    
    # We make the output dirs
    # We set defaults for guided diffusion image training
    # We load checkpoint models (model.pt, arcface, faceparser and GazeEstimator)
    image_editor = ImageEditor(args)

    # This is the main function
    image_editor.edit_image_by_prompt()
