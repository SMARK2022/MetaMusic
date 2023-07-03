# MetaMusic 概述

这是一个使用Wav2CLIP和VQGAN-CLIP从任何歌曲中生成人工智能音乐视频和图片的项目。

基本代码使用了[VQGAN-CLIP](https://github.com/nerdyrodent/VQGAN-CLIP)，音频的CLIP嵌入则使用了[Wav2CLIP](https://github.com/descriptinc/lyrebird-wav2clip)。

有关此机制的技术论文，请参阅以下链接：[Music2Video: Automatic Generation of Music Video with fusion of audio and text](https://arxiv.org/abs/2201.03809v2)


## 使用教程

由于此模型代码可能需要一些时间，并且可能存在兼容性问题，配置可能不会很顺利，请谨慎操作，祝你成功！

此示例使用[Anaconda](https://www.anaconda.com/products/individual#Downloads)管理虚拟Python环境。

为VQGAN-CLIP创建一个新的虚拟Python环境：

```sh
conda create --name metamusic python=3.9
conda activate metamusic
```

在新环境中安装Pytorch：

注意：这将安装Pytorch的CUDA版本，如果您要使用AMD显卡，请阅读下面的[AMD部分](#using-an-amd-graphics-card)。

```sh
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

安装其他所需的Python包：

```sh
pip install ftfy regex tqdm omegaconf pytorch-lightning IPython kornia imageio imageio-ffmpeg einops torch_optimizer wav2clip
```

或者使用包含版本号的```requirements.txt```文件进行安装。

克隆所需的存储库：

```sh
git clone 'https://github.com/nerdyrodent/VQGAN-CLIP'
cd VQGAN-CLIP
git clone 'https://github.com/openai/CLIP'
git clone 'https://github.com/CompVis/taming-transformers'
```

注意：在我的开发环境中，CLIP和taming-transformers都存在于本地目录中，因此在`requirements.txt`或`vqgan.yml`文件中不包含它们。

作为替代，您还可以使用pip安装taming-transformers和CLIP。

您还需要至少一个预训练的VQGAN模型。例如：

```sh
mkdir checkpoints

curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
```
请注意，使用Microsoft Windows上的```curl```命令时，应使用双引号。

`download_models.sh`脚本是下载多个模型的可选方式。默认情况下，它只会下载一个模型。

有关VQGAN预训练模型的更多信息，包括下载链接，请参阅<https://github.com/CompVis/taming-transformers#overview-of-pretrained-models>。

默认情况下，期望将模型.yaml和.ckpt文件放在`checkpoints`目录中。
有关数据集和模型的更多信息，请参阅<https://github.com/CompVis/taming-transformers>。

## 制作音乐视频

要从音乐生成视频，请指定您的音乐，并根据需要使用以下代码示例。我们提供了来自Yannic Kilcher的[repo](https://github.com/yk/clip_music_video)中的示例音乐文件和歌词文件。

如果您有一个带有时间戳信息的歌词文件，例如'lyrics/imagenet_song_lyrics.csv'中的示例，您可以使用以下命令创建一个歌词音频引导的音乐视频：

```sh
python generate.py -vid -o outputs/output.png -ap "imagenet_song.mp3" -lyr "lyrics/imagenet_song_lyrics.csv" -gid 2 -ips 100
```

要在音频表示和文本表示之间插值，使用以下代码（更具有“音乐视频”感觉）：

```sh
python generate_interpolate.py -vid -ips 100 -o outputs/output.png -ap "imagenet_song.mp3" -lyr "lyrics/imagenet_song_lyrics.csv" -gid 0
```

如果您没有歌词信息，可以使用仅音频提示运行以下命令：

```sh
python generate.py -vid -o outputs/output.png -ap "imagenet_song.mp3" -gid 2 -ips 100
```

如果在合并视频片段的过程中出现任何错误，请使用combine_mp4.py从输出目录单独连接视频片段，或者从输出目录下载视频片段，并使用视频编辑软件手动合并。

## 引用

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