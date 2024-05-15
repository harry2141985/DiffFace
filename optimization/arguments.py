import argparse
from re import S

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Larger batch sizes can accelerate the training process by processing more samples in parallel but may require more memory. 
    # However, smaller batch sizes can provide more accurate gradient estimates and better generalization but may require more iterations to converge.
    # default=1
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Current version of the code only supports a batch_size of 1",
        default=1,
    )

    # Skipping steps in the diffusion process refers to bypassing or not performing certain intermediate steps and directly applying the diffusion operation at a later stage. 
    # This approach is often used to speed up the inference process or reduce computational requirements.
    # Skipping too many steps may result in a loss of fine-grained details or accuracy, while fewer skipped steps may provide more accurate results but at the cost of increased computation.
    # default=25
    parser.add_argument(
        "--skip_timesteps",
        type=int,
        help="How many steps to skip during the diffusion.",
        default=25,
    )

    # DDPM = Denoising Diffusion Probabilistic Model
    # DDIM = Denoising Diffusion Implicit Model
    # DDIM are a more efficient class of iterative implicit probabilistic models with the same training procedure as DDPMs
    # Default is DDPM add --ddim if you want to use it
    parser.add_argument(
        "--ddim",
        help="Indicator for using DDIM instead of DDPM",
        action="store_true",
    )

    # Timestep respacing is a technique used in diffusion models to control the density of sampling during the diffusion process. 
    # In diffusion models, the goal is to transform an input distribution to a target distribution through a series of iterative steps. 
    # Timestep respacing refers to adjusting the spacing or density of these steps to achieve specific properties or characteristics in the generated samples.
    # For more details read https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
    # default=100
    parser.add_argument(
        "--timestep_respacing",
        type=str,
        help="How to respace the intervals of the diffusion process (number between 1 and 1000).",
        default="100",
    )

    # Add this argument if you do not want to preserve the destination background
    # Default is enabled, add --no_enforce_background to disable it
    parser.add_argument(
        "--no_enforce_background",
        help="Indicator disabling the last background enforcement",
        action="store_false",
        dest="enforce_background",
    )

    # Data augmentation is a common technique used to artificially increase the diversity and size of the training dataset by applying various transformations or modifications to the existing data.
    # Increasing the number of augmentations can help improve the model's generalization ability by exposing it to a wider range of variations and reducing the risk of overfitting.
    # default=8
    parser.add_argument("--aug_num", type=int, help="The number of augmentation", default=8)

    # Random seed default=404
    parser.add_argument("--seed", type=int, help="The random seed", default=404)

    # The GPU ID default=0
    parser.add_argument("--gpu_id", type=int, help="The GPU ID", default=0)

    # The output path, default=output
    # folder will be created if doesn't exist
    parser.add_argument("--output_path", type=str, default="data/debug")

    # The output result file
    # default=output.png
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="The filename to save, must be png",
        default="output.png",
    )

    # Increasing the number of iterations allows for more fine-grained transformations and can potentially capture more details in the generated samples.
    # However, it also increases the computational cost and may require more memory.
    # default=4
    parser.add_argument("--iterations_num", type=int, help="The number of iterations", default=1)

    # Target-preserving blending is to gradually increase the mask intensity from zero to one, according to the time of the diffusion process T.
    # The masking_threshold argument sets the Target-preserving blending time.
    # default=30
    parser.add_argument("--masking_threshold", type=int, help="The number of masking threshold", default=30)

    # We give identity guidance to prevent loss of source identity in the denoising process.
    # default=6000
    parser.add_argument("--loss_weight", type=int, help="ID loss weight", default=6000)

    # Input images are aligned and cropped into data/aligned folder, to be processed by the learning models
    # default=512
    parser.add_argument("--crop_size", type=int, help="The size of the aligned and cropped images", default=256)
    
    parser.add_argument(
        "--merge_only",
        help="Merge only without cropping or editing",
        action="store_true",
    )

    parser.add_argument(
        "--merge_crop_only",
        help="Merge & crop only without editing",
        action="store_true",
    )

    args = parser.parse_args()
    return args
