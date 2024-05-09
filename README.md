# DiffFace: Diffusion-based Face Swapping with Facial Guidance

This is a fork from the original project.
- Added a colab notebook.
- Commenting the code.
- Documenting the arguments.

You can find the colab notebook in this link:
https://colab.research.google.com/drive/1GNPf5_PnX2QZASdg7P3dC76V8UgrVtWJ?usp=sharing

The pretreined weights can be found here, put them in your Drive, make sure the name is checkpoints.zip:
https://drive.google.com/file/d/1z7u38LsVaPV2ew_Ci_YLf48i-_Cge8po/view?usp=sharing

While runing the colab, you will be prompted to connect to your Drive, then the script will download and unzip the file.

## Arguments details

In the above Colab, we run the code without arguments, you can add arguments to the code like so:
```
!python main.py --masking_threshold 60
```

Here are the supported arguments:

### --batch_size
Larger batch sizes can accelerate the training process by processing more samples in parallel but may require more memory.

However, smaller batch sizes can provide more accurate gradient estimates and better generalization but may require more iterations to converge.

default=1

### --skip_timesteps
Skipping steps in the diffusion process refers to bypassing or not performing certain intermediate steps and directly applying the diffusion operation at a later stage. 

This approach is often used to speed up the inference process or reduce computational requirements.

Skipping too many steps may result in a loss of fine-grained details or accuracy, while fewer skipped steps may provide more accurate results but at the cost of increased computation.

default=25

### --ddim
DDPM = Denoising Diffusion Probabilistic Model

DDIM = Denoising Diffusion Implicit Model

DDIM are a more efficient class of iterative implicit probabilistic models with the same training procedure as DDPMs

Default is DDPM add --ddim if you want to use it

### --timestep_respacing
Timestep respacing is a technique used in diffusion models to control the density of sampling during the diffusion process. 

In diffusion models, the goal is to transform an input distribution to a target distribution through a series of iterative steps. 

Timestep respacing refers to adjusting the spacing or density of these steps to achieve specific properties or characteristics in the generated samples.

For more details read https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py

default=100 (number between 1 and 1000)

### --no_enforce_background
Add this argument if you do not want to preserve the destination background

Default is enabled, add --no_enforce_background to disable it

### --aug_num
Data augmentation is a common technique used to artificially increase the diversity and size of the training dataset by applying various transformations or modifications to the existing data.

Increasing the number of augmentations can help improve the model's generalization ability by exposing it to a wider range of variations and reducing the risk of overfitting.

default=8

### --seed
Random seed default=404

### --gpu_id
The GPU ID default=0

### --output_path
The output path, default=output

Folder will be created if doesn't exist

### --output_file
The output result file
default=output.png

### --iterations_num
Increasing the number of iterations allows for more fine-grained transformations and can potentially capture more details in the generated samples.

However, it also increases the computational cost and may require more memory.

default=4

### --masking_threshold
Target-preserving blending is to gradually increase the mask intensity from zero to one, according to the time of the diffusion process T.

The masking_threshold argument sets the Target-preserving blending time.

### --loss_weight
We give identity guidance to prevent loss of source identity in the denoising process.

default=6000

default=30

## License
This licence allows for academic and non-commercial purpose only. The entire project is under the CC-BY-NC 4.0 license.

## Citation
If you find DiffFace useful for your work please cite:
```
@inproceedings{kim2022diffface,
  title = {DiffFace: Diffusion-based Face Swapping with Facial Guidance},
  author = {K. Kim, Y. Kim, S. Cho, J. Seo, J. Nam, K. Lee, S. Kim, K. Lee},
  journal = {Arxiv},
  year = {2022}
}
```

## Acknowledgments
This code borrows heavily from [Blended-Diffusion](https://github.com/omriav/blended-diffusion) and [Guided-Diffusion](https://github.com/openai/guided-diffusion).

