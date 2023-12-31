# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun

import ffmpeg
import cv2
import warnings
import re
from subprocess import Popen, PIPE
from PIL import ImageFile, Image, PngImagePlugin, ImageChops
import imageio
import numpy as np
import kornia.augmentation as K
from CLIP import clip
from taming.models import cond_transformer, vqgan
from omegaconf import OmegaConf
from torch_optimizer import DiffGrad, AdamP, RAdam
import math
from typing import Any, Union, List
import time

# from email.policy import default
from urllib.request import urlopen
from tqdm import tqdm
import sys
import os
import wav2clip
import librosa

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.cuda.amp import autocast


# torch.use_deterministic_algorithms(True)  # NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation


def clean_file_name(filename: str):
    invalid_chars = '[\\\/:*?"<>|]'
    replace_char = "-"
    return re.sub(invalid_chars, replace_char, filename)


def generate(
    filemusic: str,
    video_output: str = "output.mp4",
    iterations_per_second: int = 30,
    output_video_fps: float = 0,
    audio_sampling_freq: int = 16000,
    display_freq: int = 20,
    size: Union[int, int] = [368, 368],
    calc_device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    init_image: str = None,
    init_noise: str = None,
    clip_model: str = "ViT-B/32",
    vqgan_config: str = f"checkpoints/vqgan_imagenet_f16_16384.yaml",
    vqgan_checkpoint: str = f"checkpoints/vqgan_imagenet_f16_16384.ckpt",
    learningrate: float = 0.1,
    cut_method: str = "latest",
    num_cuts: int = 32,
    cut_power: float = 1.0,
    seed: int = None,
    optimiser: str = "Adam",
    display_picname: str = "output.png",
    workplace: str = "workplace",
    toclean: bool = True,
    augments: list = [],
    transrate: float = 0,
    transimg: str = None,
):
    """
    ## Function Summary

    This function generates an image using the WAV2CLIP+VQGAN algorithm. It takes an audio prompt as input and iteratively updates an initial image to match the prompt using gradient descent. The generated image is saved as output.png.

    ## Arguments

    - `filemusic` (str): The file path of the audio prompt.
    - `video_output` (str, optional): The file path to save the generated video. Defaults to "output.mp4".
    - `iterations_per_second` (int, optional): The number of iterations to perform per second. Defaults to 30.
    - `output_video_fps` (float, optional): The frames per second for the output video. Defaults to 0.
    - `audio_sampling_freq` (int, optional): The sampling frequency of the audio. Defaults to 16000.
    - `display_freq` (int, optional): The number of iterations between displaying progress. Defaults to 20.
    - `size` (Union[int, int], optional): The output size of the image. Defaults to [368, 368].
    - `calc_device` (str, optional): The device to use for calculations, either "cuda:0" or "cpu". Defaults to "cuda:0" if torch.cuda.is_available() else "cpu".
    - `init_image` (str, optional): The file path of the initial image. Defaults to None.
    - `init_noise` (str, optional): The file path of the initial noise image (pixels or gradient). Defaults to None.
    - `clip_model` (str, optional): The CLIP model to use (e.g. "ViT-B/32", "ViT-B/16"). Defaults to "ViT-B/32".
    - `vqgan_config` (str, optional): The file path of the VQGAN config. Defaults to "checkpoints/vqgan_imagenet_f16_16384.yaml".
    - `vqgan_checkpoint` (str, optional): The file path of the VQGAN checkpoint. Defaults to "checkpoints/vqgan_imagenet_f16_16384.ckpt".
    - `learningrate` (float, optional): The learning rate for gradient descent. Defaults to 0.1.
    - `cut_method` (str, optional): The method to use for cutting the image. Available options are "original", "updated", "nrupdated", "updatedpooling", and "latest". Defaults to "latest".
    - `num_cuts` (int, optional): The number of cuts to perform. Defaults to 32.
    - `cut_power` (float, optional): The power to use for cutting. Defaults to 1.0.
    - `seed` (int, optional): The seed for random number generation. Defaults to None.
    - `optimiser` (str, optional): The optimiser to use for gradient descent. Available options are "Adam", "AdamW", "Adagrad", "Adamax", "DiffGrad", "AdamP", "RAdam", and "RMSprop". Defaults to "Adam".
    - `display_picname` (str, optional): The filename for the output picture. Defaults to "output.png".
    - `workplace` (str, optional): The folder name for the workspace. Defaults to "workplace".
    - `toclean` (bool, optional): Whether to reset the workspace folder. Defaults to True.
    - `augments` (list, optional): The enabled augmentations for the latest cut method. Available options are "Ji", "Sh", "Gn", "Pe", "Ro", "Af", "Et", "Ts", "Cr", "Er", and "Re". Defaults to [].
    - `transrate` (float, optional): The image transform rate. Defaults to 0.
    - `transimg` (str, optional): The file path of the image input for transformation. Defaults to None.

    ## Raises

    - `ValueError`: Exception raised if there is an error in the parameters or during execution.

    ## Returns

    - `_type_`: Description of the returned value. (Missing information)
    """
    # Setting gpu id to use

    # pip install taming-transformers doesn't work with Gumbel, but does not yet work with coco etc
    # appending the path does work with Gumbel, but gives ModuleNotFoundError: No module named 'transformers' for coco etc
    sys.path.append("taming-transformers")

    # import taming.modules

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Supress warnings

    warnings.filterwarnings("ignore")

    if not augments:
        augments = [["Af", "Pe", "Ji", "Er"]]

    # Clean up
    # input(">> Cleanup the output folder?(Press ENTER to continue)")
    os.system("mkdir " + workplace)
    if toclean:
        for file in os.listdir(workplace):
            os.remove(workplace + "/" + file)

    # Make steps directory
    steps_folder = f"./{clean_file_name(calc_device)}"
    if not os.path.exists(steps_folder):
        os.mkdir(steps_folder)

    # Fallback to CPU if CUDA is not found and make sure GPU video rendering is also disabled
    # NB. May not work for AMD cards?
    calc_device = torch.device(calc_device)

    if calc_device != "cpu" and not torch.cuda.is_available():
        calc_device = "cpu"
        print(
            "Warning: No GPU found! Using the CPU instead. The iterations will be slow."
        )
        print(
            "Perhaps CUDA/ROCm or the right pytorch version is not properly installed?"
        )

    ## Too slow
    # if calc_device != "cpu" and torch.backends.cudnn.is_available():
    #     print("Using cudnn to boost up!")
    #     torch.backends.cudnn.deterministic = True

    # Loading Img inputed by user

    if transimg != None:
        print("Using Trans Img:", transimg)
        User_img = imageio.imread(transimg)
        User_img = cv2.resize(User_img, np.array(size))
        User_img = torch.from_numpy(User_img).to(calc_device)

    # Various functions and classes
    def sinc(x):
        return torch.where(
            x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([])
        )

    def lanczos(x, a):
        cond = torch.logical_and(-a < x, x < a)
        out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
        return out / out.sum()

    def ramp(ratio, width):
        n = math.ceil(width / ratio + 1)
        out = torch.empty([n])
        cur = 0
        for i in range(out.shape[0]):
            out[i] = cur
            cur += ratio
        return torch.cat([-out[1:].flip([0]), out])[1:-1]

    # NR: Testing with different intital images
    def random_noise_image(w, h):
        random_image = Image.fromarray(
            np.random.randint(0, 255, (w, h, 3), dtype=np.dtype("uint8"))
        )
        return random_image

    # create initial gradient image
    def gradient_2d(start, stop, width, height, is_horizontal):
        if is_horizontal:
            return np.tile(np.linspace(start, stop, width), (height, 1))
        else:
            return np.tile(np.linspace(start, stop, height), (width, 1)).T

    def gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
        result = np.zeros((height, width, len(start_list)), dtype=float)

        for i, (start, stop, is_horizontal) in enumerate(
            zip(start_list, stop_list, is_horizontal_list)
        ):
            result[:, :, i] = gradient_2d(start, stop, width, height, is_horizontal)

        return result

    def random_gradient_image(w, h):
        array = gradient_3d(
            w,
            h,
            (0, 0, np.random.randint(0, 255)),
            (
                np.random.randint(1, 255),
                np.random.randint(2, 255),
                np.random.randint(3, 128),
            ),
            (True, False, False),
        )
        random_image = Image.fromarray(np.uint8(array))
        return random_image

    # Used in older MakeCutouts
    def resample(input, size, align_corners=True):
        n, c, h, w = input.shape
        dh, dw = size

        input = input.view([n * c, 1, h, w])

        if dh < h:
            kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
            pad_h = (kernel_h.shape[0] - 1) // 2
            input = F.pad(input, (0, 0, pad_h, pad_h), "reflect")
            input = F.conv2d(input, kernel_h[None, None, :, None])

        if dw < w:
            kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
            pad_w = (kernel_w.shape[0] - 1) // 2
            input = F.pad(input, (pad_w, pad_w, 0, 0), "reflect")
            input = F.conv2d(input, kernel_w[None, None, None, :])

        input = input.view([n, c, h, w])
        return F.interpolate(input, size, mode="bicubic", align_corners=align_corners)

    class ReplaceGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_forward, x_backward):
            with autocast():
                ctx.shape = x_backward.shape
                return x_forward

        @staticmethod
        def backward(ctx, grad_in):
            return None, grad_in.sum_to_size(ctx.shape)

    replace_grad = ReplaceGrad.apply

    class ClampWithGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, min, max):
            with autocast():
                ctx.min = min
                ctx.max = max
                ctx.save_for_backward(input)
                return input.clamp(min, max)

        @staticmethod
        def backward(ctx, grad_in):
            (input,) = ctx.saved_tensors
            return (
                grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0),
                None,
                None,
            )

    clamp_with_grad = ClampWithGrad.apply

    def vector_quantize(x, codebook):
        d = (
            x.pow(2).sum(dim=-1, keepdim=True)
            + codebook.pow(2).sum(dim=1)
            - 2 * x @ codebook.T
        )
        indices = d.argmin(-1)
        x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
        return replace_grad(x_q, x)

    class Prompt(nn.Module):
        def __init__(self, embed, weight=1.0, stop=float("-inf")):
            super().__init__()
            self.register_buffer("embed", embed)
            self.register_buffer("weight", torch.as_tensor(weight))
            self.register_buffer("stop", torch.as_tensor(stop))

        def forward(self, input):
            with autocast():
                input_normed = F.normalize(input.unsqueeze(1), dim=2)
                embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
                dists = (
                    input_normed.sub(embed_normed)
                    .norm(dim=2)
                    .div(2)
                    .arcsin()
                    .pow(2)
                    .mul(2)
                )
                dists = dists * self.weight.sign()
                return (
                    self.weight.abs()
                    * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
                )

    # NR: Split prompts and weights
    def split_prompt(prompt):
        vals = prompt.rsplit(":", 2)
        # vals = vals + ['', '1', '-inf'][len(vals):]
        vals = vals + ["1", "-inf"]
        return vals[0], float(vals[1]), float(vals[2])

    class MakeCutouts(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow=1.0):
            super().__init__()
            self.cut_size = cut_size
            self.cutn = cutn
            self.cut_pow = cut_pow  # not used with pooling

            # Pick your own augments & their order
            augment_list = []
            for item in augments[0]:
                if item == "Ji":
                    augment_list.append(
                        K.ColorJitter(
                            brightness=0.1,
                            contrast=0.1,
                            saturation=0.1,
                            hue=0.1,
                            p=0.7,
                        )
                    )
                elif item == "Sh":
                    augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
                elif item == "Gn":
                    augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1.0, p=0.5))
                elif item == "Pe":
                    augment_list.append(
                        K.RandomPerspective(distortion_scale=0.7, p=0.7)
                    )
                elif item == "Ro":
                    augment_list.append(K.RandomRotation(degrees=15, p=0.7))
                elif item == "Af":
                    augment_list.append(
                        K.RandomAffine(
                            degrees=15,
                            translate=0.1,
                            shear=5,
                            p=0.7,
                            padding_mode="zeros",
                            keepdim=True,
                        )
                    )  # border, reflection, zeros
                elif item == "Et":
                    augment_list.append(K.RandomElasticTransform(p=0.7))
                elif item == "Ts":
                    augment_list.append(
                        K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7)
                    )
                elif item == "Cr":
                    augment_list.append(
                        K.RandomCrop(
                            size=(self.cut_size, self.cut_size),
                            pad_if_needed=True,
                            padding_mode="reflect",
                            p=0.5,
                        )
                    )
                elif item == "Er":
                    augment_list.append(
                        K.RandomErasing(
                            scale=(0.1, 0.4),
                            ratio=(0.3, 1 / 0.3),
                            same_on_batch=True,
                            p=0.7,
                        )
                    )
                elif item == "Re":
                    augment_list.append(
                        K.RandomResizedCrop(
                            size=(self.cut_size, self.cut_size),
                            scale=(0.1, 1),
                            ratio=(0.75, 1.333),
                            cropping_mode="resample",
                            p=0.5,
                        )
                    )

            self.augs = nn.Sequential(*augment_list)
            self.noise_fac = 0.1
            # self.noise_fac = False

            # Uncomment if you like seeing the list ;)
            # print(augment_list)

            # Pooling
            self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
            self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

        def forward(self, input):
            with autocast():
                cutouts = []

                for _ in range(self.cutn):
                    # Use Pooling
                    cutout = (self.av_pool(input) + self.max_pool(input)) / 2
                    cutouts.append(cutout)

                batch = self.augs(torch.cat(cutouts, dim=0))

                if self.noise_fac:
                    facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(
                        0, self.noise_fac
                    )
                    batch = batch + facs * torch.randn_like(batch)
                return batch

    # An updated version with Kornia augments and pooling (where my version started):
    class MakeCutoutsPoolingUpdate(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow=1.0):
            super().__init__()
            self.cut_size = cut_size
            self.cutn = cutn
            self.cut_pow = cut_pow  # Not used with pooling

            self.augs = nn.Sequential(
                K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode="border"),
                K.RandomPerspective(0.7, p=0.7),
                K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
                K.RandomErasing((0.1, 0.4), (0.3, 1 / 0.3), same_on_batch=True, p=0.7),
            )

            self.noise_fac = 0.1
            self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
            self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

        def forward(self, input):
            with autocast():
                sideY, sideX = input.shape[2:4]
                max_size = min(sideX, sideY)
                min_size = min(sideX, sideY, self.cut_size)
                cutouts = []

                for _ in range(self.cutn):
                    cutout = (self.av_pool(input) + self.max_pool(input)) / 2
                    cutouts.append(cutout)

                batch = self.augs(torch.cat(cutouts, dim=0))

                if self.noise_fac:
                    facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(
                        0, self.noise_fac
                    )
                    batch = batch + facs * torch.randn_like(batch)
                return batch

    # An Nerdy updated version with selectable Kornia augments, but no pooling:
    class MakeCutoutsNRUpdate(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow=1.0):
            super().__init__()
            self.cut_size = cut_size
            self.cutn = cutn
            self.cut_pow = cut_pow
            self.noise_fac = 0.1

            # Pick your own augments & their order
            augment_list = []
            for item in augments[0]:
                if item == "Ji":
                    augment_list.append(
                        K.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7
                        )
                    )
                elif item == "Sh":
                    augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
                elif item == "Gn":
                    augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1.0, p=0.5))
                elif item == "Pe":
                    augment_list.append(
                        K.RandomPerspective(distortion_scale=0.5, p=0.7)
                    )
                elif item == "Ro":
                    augment_list.append(K.RandomRotation(degrees=15, p=0.7))
                elif item == "Af":
                    augment_list.append(
                        K.RandomAffine(
                            degrees=30,
                            translate=0.1,
                            shear=5,
                            p=0.7,
                            padding_mode="zeros",
                            keepdim=True,
                        )
                    )  # border, reflection, zeros
                elif item == "Et":
                    augment_list.append(K.RandomElasticTransform(p=0.7))
                elif item == "Ts":
                    augment_list.append(
                        K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7)
                    )
                elif item == "Cr":
                    augment_list.append(
                        K.RandomCrop(
                            size=(self.cut_size, self.cut_size),
                            pad_if_needed=True,
                            padding_mode="reflect",
                            p=0.5,
                        )
                    )
                elif item == "Er":
                    augment_list.append(
                        K.RandomErasing(
                            scale=(0.1, 0.4),
                            ratio=(0.3, 1 / 0.3),
                            same_on_batch=True,
                            p=0.7,
                        )
                    )
                elif item == "Re":
                    augment_list.append(
                        K.RandomResizedCrop(
                            size=(self.cut_size, self.cut_size),
                            scale=(0.1, 1),
                            ratio=(0.75, 1.333),
                            cropping_mode="resample",
                            p=0.5,
                        )
                    )

            self.augs = nn.Sequential(*augment_list)

        def forward(self, input):
            with autocast():
                sideY, sideX = input.shape[2:4]
                max_size = min(sideX, sideY)
                min_size = min(sideX, sideY, self.cut_size)
                cutouts = []
                for _ in range(self.cutn):
                    size = int(
                        torch.rand([]) ** self.cut_pow * (max_size - min_size)
                        + min_size
                    )
                    offsetx = torch.randint(0, sideX - size + 1, ())
                    offsety = torch.randint(0, sideY - size + 1, ())
                    cutout = input[
                        :, :, offsety : offsety + size, offsetx : offsetx + size
                    ]
                    cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
                batch = self.augs(torch.cat(cutouts, dim=0))
                if self.noise_fac:
                    facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(
                        0, self.noise_fac
                    )
                    batch = batch + facs * torch.randn_like(batch)
                return batch

    # An updated version with Kornia augments, but no pooling:
    class MakeCutoutsUpdate(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow=1.0):
            super().__init__()
            self.cut_size = cut_size
            self.cutn = cutn
            self.cut_pow = cut_pow
            self.augs = nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
                # K.RandomSolarize(0.01, 0.01, p=0.7),
                K.RandomSharpness(0.3, p=0.4),
                K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"),
                K.RandomPerspective(0.2, p=0.4),
            )
            self.noise_fac = 0.1

        def forward(self, input):
            with autocast():
                sideY, sideX = input.shape[2:4]
                max_size = min(sideX, sideY)
                min_size = min(sideX, sideY, self.cut_size)
                cutouts = []
                for _ in range(self.cutn):
                    size = int(
                        torch.rand([]) ** self.cut_pow * (max_size - min_size)
                        + min_size
                    )
                    offsetx = torch.randint(0, sideX - size + 1, ())
                    offsety = torch.randint(0, sideY - size + 1, ())
                    cutout = input[
                        :, :, offsety : offsety + size, offsetx : offsetx + size
                    ]
                    cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
                batch = self.augs(torch.cat(cutouts, dim=0))
                if self.noise_fac:
                    facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(
                        0, self.noise_fac
                    )
                    batch = batch + facs * torch.randn_like(batch)
                return batch

    # This is the original version (No pooling)
    class MakeCutoutsOrig(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow=1.0):
            super().__init__()
            self.cut_size = cut_size
            self.cutn = cutn
            self.cut_pow = cut_pow

        def forward(self, input):
            with autocast():
                sideY, sideX = input.shape[2:4]
                max_size = min(sideX, sideY)
                min_size = min(sideX, sideY, self.cut_size)
                cutouts = []
                for _ in range(self.cutn):
                    size = int(
                        torch.rand([]) ** self.cut_pow * (max_size - min_size)
                        + min_size
                    )
                    offsetx = torch.randint(0, sideX - size + 1, ())
                    offsety = torch.randint(0, sideY - size + 1, ())
                    cutout = input[
                        :, :, offsety : offsety + size, offsetx : offsetx + size
                    ]
                    cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
                return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)

    def load_vqgan_model(config_path, checkpoint_path):
        config = OmegaConf.load(config_path)
        if config.model.target == "taming.models.vqgan.VQModel":
            model = vqgan.VQModel(**config.model.params)
            model.eval().requires_grad_(False)
            model.init_from_ckpt(checkpoint_path)
        elif config.model.target == "taming.models.vqgan.GumbelVQ":
            ValueError(f"Gumble model not supported now: {config.model.target}")
        elif config.model.target == "taming.models.cond_transformer.Net2NetTransformer":
            parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
            parent_model.eval().requires_grad_(False)
            parent_model.init_from_ckpt(checkpoint_path)
            model = parent_model.first_stage_model
        else:
            raise ValueError(f"unknown model type: {config.model.target}")
        del model.loss
        return model

    def resize_image(image, out_size):
        ratio = image.size[0] / image.size[1]
        area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
        size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
        return image.resize(size, Image.LANCZOS)

    # Do it
    model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(calc_device)
    jit = True if "1.7.1" in torch.__version__ else False
    perceptor = (
        clip.load(clip_model, calc_device, jit=jit)[0].eval().requires_grad_(False)
    )

    # clock=deepcopy(perceptor.visual.positional_embedding.data)
    # perceptor.visual.positional_embedding.data = clock/clock.max()
    # perceptor.visual.positional_embedding.data=clamp_with_grad(clock,0,1)

    cut_size = perceptor.visual.input_resolution
    f = 2 ** (model.decoder.num_resolutions - 1)

    # Cutout class options:
    # 'latest','original','updated' or 'updatedpooling'
    if cut_method == "latest":
        make_cutouts = MakeCutouts(cut_size, num_cuts, cut_pow=cut_power)
    elif cut_method == "original":
        make_cutouts = MakeCutoutsOrig(cut_size, num_cuts, cut_pow=cut_power)
    elif cut_method == "updated":
        make_cutouts = MakeCutoutsUpdate(cut_size, num_cuts, cut_pow=cut_power)
    elif cut_method == "nrupdated":
        make_cutouts = MakeCutoutsNRUpdate(cut_size, num_cuts, cut_pow=cut_power)
    else:
        make_cutouts = MakeCutoutsPoolingUpdate(cut_size, num_cuts, cut_pow=cut_power)

    toksX, toksY = size[0] // f, size[1] // f
    sideX, sideY = toksX * f, toksY * f

    # Gumbel or not?
    e_dim = model.quantize.e_dim
    n_toks = model.quantize.n_e
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    if init_image:
        if "http" in init_image:
            img = Image.open(urlopen(init_image))
        else:
            img = Image.open(init_image)
        pil_image = img.convert("RGB")
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = model.encode(pil_tensor.to(calc_device).unsqueeze(0) * 2 - 1)
    elif init_noise == "pixels":
        img = random_noise_image(size[0], size[1])
        pil_image = img.convert("RGB")
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = model.encode(pil_tensor.to(calc_device).unsqueeze(0) * 2 - 1)
    elif init_noise == "gradient":
        img = random_gradient_image(size[0], size[1])
        pil_image = img.convert("RGB")
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = model.encode(pil_tensor.to(calc_device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(
            torch.randint(n_toks, [toksY * toksX], device=calc_device), n_toks
        ).float()

        z = one_hot @ model.quantize.embedding.weight

        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        # z = torch.rand_like(z)*2						# NR: check

    z_orig = z.clone()
    z.requires_grad_(True)

    pMs = []
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    # From imagenet - Which is better?
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # Set the optimiser
    def get_opt(opt_name, opt_lr):
        if opt_name == "Adam":
            opt = optim.Adam([z], lr=opt_lr)  # LR=0.1 (Default)
        elif opt_name == "AdamW":
            opt = optim.AdamW([z], lr=opt_lr)
        elif opt_name == "Adagrad":
            opt = optim.Adagrad([z], lr=opt_lr)
        elif opt_name == "Adamax":
            opt = optim.Adamax([z], lr=opt_lr)
        elif opt_name == "DiffGrad":
            opt = DiffGrad(
                [z], lr=opt_lr, eps=1e-9, weight_decay=1e-9
            )  # NR: Playing for reasons
        elif opt_name == "AdamP":
            opt = AdamP([z], lr=opt_lr)
        elif opt_name == "RAdam":
            opt = RAdam([z], lr=opt_lr)
        elif opt_name == "RMSprop":
            opt = optim.RMSprop([z], lr=opt_lr)
        else:
            print("Unknown optimiser. Are choices broken?")
            opt = optim.Adam([z], lr=opt_lr)
        return opt

    opt = get_opt(optimiser, learningrate)

    # Output for the user
    print("Using device:", calc_device)
    print("Optimising using:", optimiser)

    # if args.prompts:
    #     print("Using text prompts:", args.prompts)
    if filemusic:
        print("Using audio prompts:", filemusic)
    if init_image:
        print("Using initial image:", init_image)

    # Vector quantize
    def synth(z):
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(
            3, 1
        )
        return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

    # @torch.no_grad()
    @torch.inference_mode()
    def checkin(i, losses):
        losses_str = ", ".join(f"{loss.item():g}" for loss in losses)
        tqdm.write(f"i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}")
        out = synth(z)
        info = PngImagePlugin.PngInfo()
        # info.add_text("comment", f"{args.prompts}")
        TF.to_pil_image(out[0].cpu()).save(
            workplace + "/" + display_picname, pnginfo=info
        )

    def clac_loss(i):
        out = synth(z)
        iii = (
            perceptor.encode_image(normalize(make_cutouts(out))).to(calc_device).float()
        )

        result = []

        for prompt in pMs:
            result.append(prompt(iii))

        if transimg != None:
            img = torch.squeeze(out, 0).transpose(0, 1).transpose(1, 2)
            minus = (img - User_img / 255).reshape(-1)
            loss_trans = torch.dot(minus, minus) / minus.shape[0] * 4 * transrate
            result.append(loss_trans)

        img = np.array(
            out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8)
        )[:, :, :]
        img = np.transpose(img, (1, 2, 0))
        imageio.imwrite(f"{steps_folder}/" + str(i) + ".png", np.array(img))
        return result  # return loss

    def train(i):
        opt.zero_grad(set_to_none=True)
        lossAll = clac_loss(i)

        if i % display_freq == 0:
            checkin(i, lossAll)

        loss = sum(lossAll)
        loss.backward()
        opt.step()

        # with torch.no_grad():
        with torch.inference_mode():
            z.copy_(z.maximum(z_min).minimum(z_max))
        return f"{steps_folder}/" + str(i) + ".png"

    # Loading the models & Getting total video length
    wav2clip_model = wav2clip.get_model()
    audio_lst = []
    fps_lst = []
    audiotimestamp_lst = []
    audio_length = []
    iterations_num = []
    audio, sr = librosa.load(filemusic, sr=audio_sampling_freq)
    total_seconds = int(len(audio) // sr)

    tempo, beats_raw = librosa.beat.beat_track(y=audio, sr=sr)
    spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, fmax=8000, hop_length=512
    )
    # get mean power at each time point
    specm = np.mean(spec, axis=0)
    # normalize mean power between 0-1
    specm = (specm - np.min(specm)) / np.ptp(specm)

    beats_raw = beats_raw * 512  # indexed audio before sampling

    beats = [0]

    for i in range(len(beats_raw)):  # 每两拍记一次
        if i % 2 == 0:
            beats.append(beats_raw[i])
    beats.append(len(audio) - 1)

    def interpolate_array(arr, interval):
        i = 0
        while i < len(arr) - 1:
            diff = abs(arr[i + 1] - arr[i])
            if diff > interval:
                new_value = (arr[i + 1] + arr[i]) / 2
                arr.insert(i + 1, int(new_value + 0.5))
            else:
                i += 1
        return arr

    beats = interpolate_array(beats, 4 * sr)

    length_cum = 0

    # input()

    for i in range(len(beats) - 1):
        audio_seg = audio[beats[i] : beats[i + 1]]
        length = (beats[i + 1] - beats[i]) / sr

        audio_lst.append(audio_seg)
        audio_length.append(length)
        # fps_lst.append()

        length_cum += length
        iterations_num.append(int(length * iterations_per_second + 0.5))
        print(
            "Clip: " + str(i) + "/" + str(len(beats) - 1),
            ": length",
            time.ctime(16 * 3600 + length_cum - length)[-13:-5],
            "~",
            time.ctime(16 * 3600 + length_cum)[-13:-5],
            length,
            length * iterations_per_second,
            int(length * iterations_per_second + 0.5),
        )
        audiotimestamp_lst.append(
            "Clip: "
            + str(i)
            + "/"
            + str(len(beats) - 1)
            + "  "
            + ": length"
            + "  "
            + time.ctime(16 * 3600 + length_cum - length)[-13:-5]
            + "~"
            + time.ctime(16 * 3600 + length_cum)[-13:-5]
            + "  "
            + str(length)
            + "  "
            + str(length * iterations_per_second)
            + "  "
            + str(int(length * iterations_per_second + 0.5))
        )
    print("Iterations_num:", np.sum(iterations_num))

    # input("Ready to start?")

    # lyrics_index = 0

    # Looping to create segments of mp4 files
    for a in range(len(audio_length)):
        print("\n" + audiotimestamp_lst[a])
        # text_turn = False
        # Randomly initializing seed in each video clip
        if seed == None:
            seed = torch.seed()
        torch.manual_seed(seed)
        pMs = []

        # WavCLIP embedding
        audio = audio_lst[a]
        # print(audio.shape)
        # input("Are you OK?")
        pMs.append(
            Prompt(
                torch.from_numpy(wav2clip.embed_audio(audio, wav2clip_model)),
                float(1.0),
                float("-inf"),
            ).to(calc_device)
        )
        # input("Go Now?")

        if a != 0:
            # Initial Image embedding
            prompt = workplace + "/" + display_picname
            path, weight, stop = split_prompt(prompt)
            img = Image.open(path)
            pil_image = img.convert("RGB")
            img = resize_image(pil_image, (sideX, sideY))
            batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(calc_device))
            pMs.append(
                Prompt(
                    perceptor.encode_image(normalize(batch)).to(calc_device).float(),
                    weight,
                    stop,
                ).to(calc_device)
            )
            # print(batch)

        i = 0  # Iteration counter
        p = 1  # Phrase counter
        smoother = 0  # Smoother counter
        this_video_frame = 0  # for video styling

        # Do it
        with tqdm() as pbar:
            while True:
                # Training time
                if i == 1 and a > 0:
                    pMs.pop()  # Getting rid of initial image prompt after first iteration
                train(i)
                # input("Trained...")

                # Ready to stop yet?
                if i == iterations_num[a]:
                    break
                i += 1
                pbar.update()

        # All done :)

        # Video generation
        init_frame = 0  # Initial video frame
        last_frame = (
            i  # This will raise an error if that number of frames does not exist.
        )

        length = audio_length[a]
        # length = args.video_length # Desired time of the video in seconds

        min_fps = 10
        max_fps = 120

        total_frames = last_frame - init_frame

        frames = []
        tqdm.write("Generating video...")
        for i in range(init_frame, last_frame):
            temp = Image.open(f"{steps_folder}/" + str(i) + ".png")
            keep = temp.copy()
            frames.append(keep)
            temp.close()

        if output_video_fps > 9:
            # Hardware encoding and video frame interpolation
            print("Hardware encoding and video frame interpolation")
            print("Creating interpolated frames...")
            ffmpeg_filter = f"minterpolate='mi_mode=mci:me=hexbs:me_mode=bidir:mc_mode=aobmc:vsbmc=1:mb_size=8:search_param=32:fps={output_video_fps}'"
            output_videocut = re.compile("\.png$").sub(
                f"{a}.mp4", workplace + "/" + display_picname
            )
            try:
                p = Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "png",
                        "-r",
                        str(iterations_per_second),
                        "-i",
                        "-",
                        "-b:v",
                        "10M",
                        "-vcodec",
                        "h264_nvenc",
                        "-pix_fmt",
                        "yuv420p",
                        "-strict",
                        "-2",
                        "-filter:v",
                        f"{ffmpeg_filter}",
                        # "-metadata",
                        # f"comment={args.prompts}",
                        output_videocut,
                    ],
                    stdin=PIPE,
                )
            except FileNotFoundError:
                print("ffmpeg command failed - check your installation")
            for im in tqdm(frames):
                im.save(p.stdin, "PNG")
            p.stdin.close()
            p.wait()
        else:
            # CPU
            fps = np.clip(total_frames / length, min_fps, max_fps)
            output_videocut = re.compile("\.png$").sub(
                f"{a:03d}.mp4", workplace + "/" + display_picname
            )
            print(">>> Creating Video With CPU: FPS=", fps)
            try:
                p = Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "png",
                        "-r",
                        str(fps),
                        "-i",
                        "-",
                        "-vcodec",
                        "libx264",
                        "-r",
                        str(fps),
                        "-pix_fmt",
                        "yuv420p",
                        "-crf",
                        "17",
                        "-preset",
                        "veryslow",
                        # "-metadata",
                        # f"comment={args.prompts}",
                        output_videocut,
                    ],
                    stdin=PIPE,
                )
            except FileNotFoundError:
                print("ffmpeg command failed - check your installation")
            for im in tqdm(frames):
                im.save(p.stdin, "PNG")
            p.stdin.close()
            p.wait()

    mp4_files = [file for file in os.listdir(workplace) if file.endswith(".mp4")]
    mp4_files.sort()
    with open("list-of-files.txt", "w") as f:
        for file in mp4_files:
            file = workplace + "/" + file
            f.write(f"file {file[:-4]}.ts\n")
            os.system(
                f"ffmpeg -i {file} -vcodec copy -acodec copy -vbsf h264_mp4toannexb {file[:-4]}.ts"
            )

    tmp_ts_combine = workplace + "/" + "output.ts"
    os.system(f"ffmpeg -f concat -safe 0 -i list-of-files.txt -c copy {tmp_ts_combine}")

    video = ffmpeg.input(tmp_ts_combine)
    audio = ffmpeg.input(filemusic)
    ffmpeg.concat(video, audio, v=1, a=1).output(video_output, strict="-2").run()
    pMs = []
    torch.cuda.empty_cache()
    return video_output
