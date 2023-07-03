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
import argparse
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
import pandas as pd

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.cuda import get_device_properties
from torch.cuda.amp import autocast

torch.backends.cudnn.benchmark = (
    True  # NR: True is a bit faster, but can lead to OOM. False is more deterministic.
)
# torch.use_deterministic_algorithms(True)  # NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation


# Create the parser
vq_parser = argparse.ArgumentParser(description="Image generation using VQGAN+CLIP")

# Check for GPU and reduce the default image size if low VRAM
default_image_size = 512  # >8GB VRAM
if not torch.cuda.is_available():
    print("Warning: No GPU Found.")
    default_image_size = 512  # no GPU found
elif (
    get_device_properties(0).total_memory <= 2**33
):  # 2 ** 33 = 8,589,934,592 bytes = 8 GB
    print("Warning: GPU VRAM is less than 8GB.")
    default_image_size = 368  # <8GB VRAM

def clean_file_name(filename:str):
    invalid_chars='[\\\/:*?"<>|]'
    replace_char='-'
    return re.sub(invalid_chars,replace_char,filename)

# Add the arguments
# vq_parser.add_argument(
#     "-p", "--prompts", type=str, help="Text prompts", default=None, dest="prompts"
# )
# vq_parser.add_argument(
#     "-ip",
#     "--image_prompts",
#     type=str,
#     help="Image prompts / target image",
#     default=[],
#     dest="image_prompts",
# )
# vq_parser.add_argument(
#     "-ips",
#     "--iterations_per_second",
#     type=int,
#     help="Number of iterations per second",
#     default=30,
#     dest="iterations_per_second",
# )
# vq_parser.add_argument(
#     "-se",
#     "--save_every",
#     type=int,
#     help="Save image iterations",
#     default=50,
#     dest="display_freq",
# )
# vq_parser.add_argument(
#     "-s",
#     "--size",
#     nargs=2,
#     type=int,
#     help="Image size (width height) (default: %(default)s)",
#     default=[default_image_size, default_image_size],
#     dest="size",
# )
# vq_parser.add_argument(
#     "-ii", "--f", type=str, help="Initial image", default=None, dest="init_image"
# )
# vq_parser.add_argument(
#     "-in",
#     "--init_noise",
#     type=str,
#     help="Initial noise image (pixels or gradient)",
#     default=None,
#     dest="init_noise",
# )
# vq_parser.add_argument(
#     "-iw",
#     "--init_noise",
#     type=float,
#     help="Initial weight",
#     default=0.0,
#     dest="init_noise",
# )
# vq_parser.add_argument(
#     "-m",
#     "--clip_model",
#     type=str,
#     help="CLIP model (e.g. ViT-B/32, ViT-B/16)",
#     default="ViT-B/32",
#     dest="clip_model",
# )
# vq_parser.add_argument(
#     "-conf",
#     "--vqgan_config",
#     type=str,
#     help="VQGAN config",
#     default=f"checkpoints/vqgan_imagenet_f16_16384.yaml",
#     dest="vqgan_config",
# )
# vq_parser.add_argument(
#     "-ckpt",
#     "--vqgan_checkpoint",
#     type=str,
#     help="VQGAN checkpoint",
#     default=f"checkpoints/vqgan_imagenet_f16_16384.ckpt",
#     dest="vqgan_checkpoint",
# )
# vq_parser.add_argument(
#     "-nps",
#     "--noise_prompt_seeds",
#     nargs="*",
#     type=int,
#     help="Noise prompt seeds",
#     default=[],
#     dest="noise_prompt_seeds",
# )
# vq_parser.add_argument(
#     "-npw",
#     "--noise_prompt_weights",
#     nargs="*",
#     type=float,
#     help="Noise prompt weights",
#     default=[],
#     dest="noise_prompt_weights",
# )
# vq_parser.add_argument(
#     "-lr",
#     "--learning_rate",
#     type=float,
#     help="Learning rate",
#     default=0.1,
#     dest="step_size",
# )
# vq_parser.add_argument(
#     "-cutm",
#     "--cut_method",
#     type=str,
#     help="Cut method",
#     choices=["original", "updated", "nrupdated", "updatedpooling", "latest"],
#     default="latest",
#     dest="cut_method",
# )
# vq_parser.add_argument(
#     "-cuts", "--num_cuts", type=int, help="Number of cuts", default=32, dest="cutn"
# )
# vq_parser.add_argument(
#     "-cutp", "--cut_power", type=float, help="Cut power", default=1.0, dest="cut_pow"
# )
# vq_parser.add_argument(
#     "-sd", "--seed", type=int, help="Seed", default=None, dest="seed"
# )
# vq_parser.add_argument(
#     "-opt",
#     "--optimiser",
#     type=str,
#     help="Optimiser",
#     choices=[
#         "Adam",
#         "AdamW",
#         "Adagrad",
#         "Adamax",
#         "DiffGrad",
#         "AdamP",
#         "RAdam",
#         "RMSprop",
#     ],
#     default="Adam",
#     dest="optimiser",
# )
# vq_parser.add_argument(
#     "-pic",
#     "--picture",
#     type=str,
#     help="Picture filename",
#     default="output.png",
#     dest="picturename",
# )
# vq_parser.add_argument(
#     "-wp",
#     "--workplace",
#     type=str,
#     help="Workplace foldername",
#     default="workplace",
#     dest="workplace",
# )
# vq_parser.add_argument(
#     "-vo",
#     "--video_output",
#     type=str,
#     help="video output filename",
#     default="output.mp4",
#     dest="video_output",
# )
# vq_parser.add_argument(
#     "-cl",
#     "--clean",
#     action="store_true",
#     help="Reset workplace folder",
#     dest="ifclean",
# )
# vq_parser.add_argument(
#     "-cpe",
#     "--change_prompt_every",
#     type=int,
#     help="Prompt change frequency",
#     default=0,
#     dest="prompt_frequency",
# )
# vq_parser.add_argument(
#     "-vl",
#     "--video_length",
#     type=float,
#     help="Video length in seconds (not interpolated)",
#     default=10,
#     dest="video_length",
# )
# vq_parser.add_argument(
#     "-ofps",
#     "--output_video_fps",
#     type=float,
#     help="Create an interpolated video (Nvidia GPU only) with this fps (min 10. best set to 30 or 60)",
#     default=0,
#     dest="output_video_fps",
# )
# vq_parser.add_argument(
#     "-d",
#     "--deterministic",
#     action="store_true",
#     help="Enable cudnn.deterministic?",
#     dest="cudnn_determinism",
# )
# vq_parser.add_argument(
#     "-aug",
#     "--augments",
#     nargs="+",
#     action="append",
#     type=str,
#     choices=["Ji", "Sh", "Gn", "Pe", "Ro", "Af", "Et", "Ts", "Cr", "Er", "Re"],
#     help="Enabled augments (latest vut method only)",
#     default=[],
#     dest="augments",
# )
# vq_parser.add_argument(
#     "-cd",
#     "--cuda_device",
#     type=str,
#     help="Cuda device to use",
#     default="cuda:0",
#     dest="cuda_device",
# )
# vq_parser.add_argument(
#     "-ap", "--audio_prompt", type=str, default=None, dest="audio_prompt"
# )
# vq_parser.add_argument("-lyr", "--lyrics", type=str, default=None, dest="lyrics")
# vq_parser.add_argument(
#     "-sf", "--audio_sampling_freq", type=int, default=16000, dest="audio_sampling_freq"
# )
# vq_parser.add_argument("-gid", "--gpu_id", type=str, default=0, dest="gpu_id")

# vq_parser.add_argument(
#     "-tr",
#     "--trans_rate",
#     type=float,
#     help="Transform rate",
#     default=1,
#     dest="transrate",
# )

# vq_parser.add_argument(
#     "-ti",
#     "--trans_image",
#     type=str,
#     help="Transform image input",
#     default="None",
#     dest="usr_img_path",
# )


# Execute the parse_args() method
# args = vq_parser.parse_args()


def generate(
    filemusic: str,
    video_output: str="output.mp4",
    iterations_per_second: int = 30,
    output_video_fps: float = 0,
    audio_sampling_freq: int = 16000,
    display_freq: int = 10,
    size: Union[int, int] = [default_image_size, default_image_size],
    calc_device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    init_image: str = None,
    init_noise: str = None,
    clip_model: str = "ViT-B/32",
    vqgan_config: str = f"checkpoints/vqgan_imagenet_f16_16384.yaml",
    vqgan_checkpoint: str = f"checkpoints/vqgan_imagenet_f16_16384.ckpt",
    learningrate: float = 0.2,
    cut_method: str = "latest",
    num_cuts: int = 32,
    cut_power: float = 1.0,
    seed: int = None,
    optimiser: str = "Adam",
    display_picname: str = "output.jpg",
    workplace: str = "workplace",
    toclean: bool = True,
    augments: list = [],
    transrate: float = 0,
    transimg: str = None,
):
    # Setting gpu id to use

    # pip install taming-transformers doesn't work with Gumbel, but does not yet work with coco etc
    # appending the path does work with Gumbel, but gives ModuleNotFoundError: No module named 'transformers' for coco etc
    sys.path.append("taming-transformers")

    # import taming.modules

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Supress warnings

    warnings.filterwarnings("ignore")

    # if not args.prompts and not args.image_prompts:
    #    args.prompts = "A cute, smiling, Nerdy Rodent"

    if not augments:
        augments = [["Af", "Pe", "Ji", "Er"]]

    # Clean up
    # input(">> Cleanup the output folder?(Press ENTER to continue)")
    os.system("mkdir " + workplace)
    if toclean:
        for file in os.listdir(workplace):
            os.remove(workplace + "/" + file)

    # Make steps directory
    steps_folder=f"./{clean_file_name(calc_device)}"
    if not os.path.exists(steps_folder):
        os.mkdir(steps_folder)

    # Split text prompts using the pipe character (weights are split later)
    # if args.prompts:
    #     # For stories, there will be many phrases
    #     story_phrases = [phrase.strip() for phrase in args.prompts.split("^")]

    #     # Make a list of all phrases
    #     all_phrases = []
    #     for phrase in story_phrases:
    #         all_phrases.append(phrase.split("|"))

    #     # First phrase
    #     args.prompts = all_phrases[0]

    # Split target images using the pipe character (weights are split later)
    # if args.image_prompts:
    #     args.image_prompts = args.image_prompts.split("|")
    #     args.image_prompts = [image.strip() for image in args.image_prompts]

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
            ctx.shape = x_backward.shape
            return x_forward

        @staticmethod
        def backward(ctx, grad_in):
            return None, grad_in.sum_to_size(ctx.shape)

    replace_grad = ReplaceGrad.apply

    class ClampWithGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, min, max):
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
            input_normed = F.normalize(input.unsqueeze(1), dim=2)
            embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
            dists = (
                input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
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
            cutouts = []

            for _ in range(self.cutn):
                # Use Pooling
                cutout = (self.av_pool(input) + self.max_pool(input)) / 2
                cutouts.append(cutout)

            batch = self.augs(torch.cat(cutouts, dim=0))

            if self.noise_fac:
                facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
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
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []

            for _ in range(self.cutn):
                cutout = (self.av_pool(input) + self.max_pool(input)) / 2
                cutouts.append(cutout)

            batch = self.augs(torch.cat(cutouts, dim=0))

            if self.noise_fac:
                facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
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
                            brightness=0.1,
                            contrast=0.1,
                            saturation=0.1,
                            hue=0.1,
                            p=0.7
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

        @autocast
        def forward(self, input):
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            for _ in range(self.cutn):
                size = int(
                    torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
                )
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            batch = self.augs(torch.cat(cutouts, dim=0))
            if self.noise_fac:
                facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
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
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            for _ in range(self.cutn):
                size = int(
                    torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
                )
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            batch = self.augs(torch.cat(cutouts, dim=0))
            if self.noise_fac:
                facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
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
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            for _ in range(self.cutn):
                size = int(
                    torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
                )
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
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
    # if args.image_prompts:
    #     print("Using image prompts:", args.image_prompts)
    if init_image:
        print("Using initial image:", init_image)
    # if args.noise_prompt_weights:
    #     print("Noise prompt weights:", args.noise_prompt_weights)

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

        # if args.init_noise:
        #     # result.append(F.mse_loss(z, z_orig) * args.init_noise / 2)
        #     result.append(
        #         F.mse_loss(z, torch.zeros_like(z_orig))
        #         * ((1 / torch.tensor(i * 2 + 1)) * args.init_noise)
        #         / 2
        #     )

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
    # Loading the lyrics
    # if args.lyrics:
    #     df = pd.read_csv(args.lyrics)
    #     lyrics_cp = []
    #     lyrics = []

    #     for index, row in df.iterrows():
    #         ts = row["timestamp"]
    #         txt = str(row["text"])
    #         ts = ts.split(":")
    #         minute = int(ts[0])
    #         sec = int(ts[1])
    #         ts = sec + (60 * minute)
    #         print(ts, txt)
    #         lyrics_cp.append(ts)
    #         lyrics.append(txt)

    #     lyrics_num = len(lyrics)
    #     print(f"number of lyrics blocks : {lyrics_num}")

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

    current_accum = 0  # Accumulating length of audios
    prev_accum = 0
    # Looping to create segments of mp4 files
    for a in range(len(audio_length)):
        print("\n" + audiotimestamp_lst[a])
        # text_turn = False
        # Randomly initializing seed in each video clip
        if seed == None:
            seed = torch.seed()
        torch.manual_seed(seed)
        pMs = []
        # if args.lyrics:
        #     if lyrics_index < lyrics_num:
        #         lyric_timestamp = lyrics_cp[lyrics_index]
        #         lyrics_text = lyrics[lyrics_index]
        #         len = audio_length[a]
        #         current_accum += len
        #         if (lyric_timestamp >= prev_accum) and (lyric_timestamp <= current_accum):
        #             text_turn = True
        #             lyrics_index += 1
        #         prev_accum = current_accum

        # if text_turn:
        #     # CLIP tokenize/encode
        #     prompt = lyrics_text
        #     txt, weight, stop = split_prompt(prompt)
        #     embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        #     pMs.append(Prompt(embed, weight, stop).to(device))
        # else:
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
                # Change text prompt
                # if args.prompt_frequency > 0:
                #     if i % args.prompt_frequency == 0 and i > 0:
                #         # In case there aren't enough phrases, just loop
                #         if p >= len(all_phrases):
                #             p = 0

                #         pMs = []
                #         args.prompts = all_phrases[p]

                #         # Show user we're changing prompt
                #         print(args.prompts)

                #         for prompt in args.prompts:
                #             txt, weight, stop = split_prompt(prompt)
                #             embed = perceptor.encode_text(
                #                 clip.tokenize(txt).to(device)
                #             ).float()
                #             pMs.append(Prompt(embed, weight, stop).to(device))

                #         p += 1

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
