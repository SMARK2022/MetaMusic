# MetaMusic Overview

**([简体中文](README.md)/EN)**

This is a project that uses Wav2CLIP and VQGAN-CLIP to generate AI music videos and images from any song. We named our first-year project group MetaMusic, which signifies the diversity of transformations and meaning expression that can be achieved through music.

The basic code uses [VQGAN-CLIP](https://github.com/nerdyrodent/VQGAN-CLIP), and the CLIP embedding for audio uses [Wav2CLIP](https://github.com/descriptinc/lyrebird-wav2clip). After reading the literature, we found that our project has similarities with [Music2Video](https://github.com/joeljang/music2video), and we have made several original modifications and optimizations.

For technical information related to this mechanism, please refer to the following images:
![quicker_ce65ebeb-0c13-42d5-b5ed-be01dc2b6309.png](https://s2.loli.net/2023/07/04/C9H8SOzauy3rhEf.png)

## Explorations

- `[×]` We have tried integrating the AudioClip model, but it didn't produce good results for music due to the training dataset's focus on real-world sound recognition.
- `[×]` We have tried integrating other VQ-GAN models, but the ImageNet dataset's realism doesn't align well with generating emotional images for songs, resulting in poor results.
- `[×]` We have attempted to transform the entire project into a CLIP-guided Diffusion model, but the video generation efficiency was too low, resulting in a significant increase in video generation time. The video module that generates frames using large models has been temporarily put aside.
- `[√]` We have made compatibility improvements to the code and performed mixed-precision computations to reduce GPU memory usage for some models.
- `[√]` We have successfully wrapped the two systems into APIs for easy usage.
- `[√]` We have implemented a visual interface using gradio.
- `[ ]` We will try to create our own dataset to train the Wav2Clip model (or possibly VQ-GAN).
- `[ ]` We will continue to explore other image generation models, including but not limited to Diffusion architecture, and integrate them into the MetaMusic impression image generation module.

## Current Results Showcase

![Visual Interface](https://s2.loli.net/2023/07/04/h3GOSofPW42Jte1.png)

## Usage Guide

The current supported device environments for this model are:

- Windows (verified) / Linux (unverified) systems
- Nvidia GPU with CUDA support / CPU

Due to the complexity and potential compatibility issues of the model code, the configuration may not be smooth. Please proceed with caution and good luck!

This guide assumes the usage of [Anaconda](https://www.anaconda.com/products/individual#Downloads) for managing virtual Python environments.

- Create a new virtual Python environment for VQGAN-CLIP:

```sh
conda create --name metamusic python=3.9
conda activate metamusic
```

- Next, download the CUDA and cudnn versions supported by your Nvidia GPU from [CUDA Toolkit - Free Tools and Training | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit) (generally, for 30 series GPUs and above, downloading CUDA 12 is sufficient).

- Then, install PyTorch in the new environment:

Visit [Start Locally | PyTorch](https://pytorch.org/get-started/locally/) and choose the appropriate PyTorch version for your computer. Copy the provided link and use it to download and install PyTorch (e.g., I used the cu118 version).

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- Install the necessary system support libraries, including but not limited to ffmpeg and MSVC, and configure the system environment variables accordingly.

- Install other required Python packages:

```sh
pip install ftfy regex tqdm omegaconf pytorch-lightning IPython kornia imageio imageio-ffmpeg einops torch_optimizer wav2clip
```

- Additionally, clone the required repositories:

```sh
git clone 'https://github.com/SMARK2022/MetaMusic.git'
```

- Due to version compatibility issues, it is recommended to use the `requirements.txt` file with version numbers for installation. Additionally, separately install `taming-transformers` using pip from the internet.

- You will also need at least one pretrained VQGAN model. (After testing, VQ-GAN ImageNet 16384 is recommended)

```sh
mkdir checkpoints

wget -o checkpoints/vqgan_imagenet_f16_16384.yaml 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
wget -o checkpoints/vqgan_imagenet_f16_16384.ckpt 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1'
```

For more information about VQGAN pretrained models, including download links, refer to <https://github.com/CompVis/taming-transformers#overview-of-pretrained-models>.

By default, the `.yaml` and `.ckpt` files are expected to be placed in the `checkpoints` directory. For more information about the dataset and models, refer to <https://github.com/CompVis/taming-transformers>.

## Generate Music Impression Images / Videos

- To generate music impression images/videos, you can directly run `gradiogui.py`, which is the visual interface file, and specify your music on the web page.

- If you need fine-grained parameter adjustments, you can use the two different APIs `api_picture` or `api_video` as needed.

**Example:**

```python
import api_picture
api_picture.generate(filemusic=....mp3, ...):
```

The model and process for generating music impression images are still being optimized, so the results may vary.

Additionally, the video generation process may take some time, so please be patient (approximately 2 hours).

## References

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```

```bibtex
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis},
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@article{wu2021wav2clip,
  title={Wav2CLIP: Learning Robust Audio Representations From CLIP},
  author={Wu, Ho-Hsiang and Seetharaman, Prem and Kumar, Kundan and Bello, Juan Pablo},
  journal={arXiv preprint arXiv:2110.11499},
  year={2021}
}
```
