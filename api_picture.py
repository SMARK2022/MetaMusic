# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun

import cv2
import warnings
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
import time
from typing import Any, Union, List

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


def generate(
    filemusic: str,
    n_iteration: int = 100,
    audio_sampling_freq: int = 16000,
    display_freq: int = 10,
    size: Union[int, int] = [default_image_size, default_image_size],
    calc_device: str = "cuda" if torch.cuda.is_available() else "cpu",
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
    """
    ## Function Summary

    This function generates an image using the WAV2CLIP+VQGAN algorithm. It takes an audio prompt as input and iteratively updates an initial image to match the prompt using gradient descent. The generated image is saved as output.jpg.

    ## Arguments

    - `filemusic` (str): Audio prompt.
    - `n_iteration` (int, optional): Number of iterations. Defaults to 100.
    - `audio_sampling_freq` (int, optional): Audio sampling frequency. Defaults to 16000.
    - `display_freq` (int, optional): Number of iterations between displaying progress. Defaults to 10.
    - `size` (Union[int, int], optional): Output size of the image. Defaults to [default_image_size, default_image_size].
    - `calc_device` (str, optional): Calculation device to use, "cuda:0" or "cpu". Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    - `init_image` (str, optional): Initial image. Defaults to None.
    - `init_noise` (str, optional): Initial noise image (pixels or gradient). Defaults to None.
    - `clip_model` (str, optional): CLIP model to use (e.g. "ViT-B/32", "ViT-B/16"). Defaults to "ViT-B/32".
    - `vqgan_config` (str, optional): VQGAN config file path. Defaults to "checkpoints/vqgan_imagenet_f16_16384.yaml".
    - `vqgan_checkpoint` (str, optional): VQGAN checkpoint file path. Defaults to "checkpoints/vqgan_imagenet_f16_16384.ckpt".
    - `learningrate` (float, optional): Learning rate for gradient descent. Defaults to 0.2.
    - `cut_method` (str, optional): Cut method to use, ["original", "updated", "nrupdated", "updatedpooling", "latest"]. Defaults to "latest".
    - `num_cuts` (int, optional): Number of cuts. Defaults to 32.
    - `cut_power` (float, optional): Cut power. Defaults to 1.0.
    - `seed` (int, optional): Seed for random number generation. Defaults to None.
    - `optimiser` (str, optional): Optimiser to use, ["Adam", "AdamW", "Adagrad", "Adamax", "DiffGrad", "AdamP", "RAdam", "RMSprop"]. Defaults to "Adam".
    - `display_picname` (str, optional): Filename for the output picture. Defaults to "output.jpg".
    - `workplace` (str, optional): Folder name for the workspace. Defaults to "workplace".
    - `toclean` (bool, optional): Whether to reset the workspace folder. Defaults to True.
    - `augments` (list, optional): Enabled augmentations (for latest cut method), ["Ji", "Sh", "Gn", "Pe", "Ro", "Af", "Et", "Ts", "Cr", "Er", "Re"]. Defaults to [].
    - `transrate` (float, optional): Image transform rate. Defaults to 0.
    - `transimg` (str, optional): Image input for transformation. Defaults to None.

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
    if not os.path.exists(f"./{calc_device}"):
        os.mkdir(f"./{calc_device}")

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

    if calc_device != "cpu" and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True

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
                            p=0.7,
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

    def clac_loss():
        global i
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

        return result  # return loss

    def train(i):
        opt.zero_grad(set_to_none=True)
        lossAll = clac_loss()

        if i % display_freq == 0:
            checkin(i, lossAll)

        loss = sum(lossAll)
        loss.backward()
        opt.step()

        # with torch.no_grad():
        with torch.inference_mode():
            z.copy_(z.maximum(z_min).minimum(z_max))
            out = synth(z)
            img = np.array(
                out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8)
            )[:, :, :]
            img = np.transpose(img, (1, 2, 0))
            imageio.imwrite(f"./{calc_device}/" + str(i) + ".png", np.array(img))
            return f"./{calc_device}/" + str(i) + ".png"

    # Loading the models & Getting total video length
    wav2clip_model = wav2clip.get_model()
    audio, sr = librosa.load(filemusic, sr=audio_sampling_freq)
    
    # Randomly initializing seed in each video clip
    if seed == None:
        seed = torch.seed()
    torch.manual_seed(seed)
    pMs = []

    # WavCLIP embedding
    pMs.append(
        Prompt(
            torch.from_numpy(wav2clip.embed_audio(audio, wav2clip_model)),
            float(1.0),
            float("-inf"),
        ).to(calc_device)
    )
    # input("Go Now?")

    i = 0  # Iteration counter
    pic_name = ""

    # Do it
    with tqdm() as pbar:
        while True:
            # Training time
            pic_name = train(i)

            # Ready to stop yet?
            if i == n_iteration:
                return pic_name
            i += 1
            pbar.update()

    # All done :)