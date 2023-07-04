import gradio as gr
import api_picture
import api_video
import torch
from torch.cuda import get_device_properties

theme = gr.themes.Soft()
default_lr = 0.2
default_tr = 1
default_it = 60
default_fps = 25
available_devices = ["cpu"] + [
    f"cuda:{str(i)}" for i in range(torch.cuda.device_count())
]
torch.backends.cudnn.benchmark = (
    True  # NR: True is a bit faster, but can lead to OOM. False is more deterministic.
)
default_image_size = 512  # >8GB VRAM
if not torch.cuda.is_available():
    print("Warning: No GPU Found.")
    default_image_size = 512  # no GPU found
elif (
    get_device_properties(0).total_memory <= 2**33
):  # 2 ** 33 = 8,589,934,592 bytes = 8 GB
    print("Warning: GPU VRAM is less than 8GB.")
    default_image_size = 368  # <8GB VRAM


def Generate_img(music_file, image_path, X, Y, devices, Tr, Lr, It):
    output = api_picture.generate(
        filemusic=music_file.name,
        transimg=image_path,
        size=(X, Y),
        calc_device=devices,
        learningrate=Lr,
        transrate=Tr,
        n_iteration=It,
    )
    return output


def Generate_video(music_file, image_path, X, Y, devices, Tr, Lr, Fps):
    output = api_video.generate(
        filemusic=music_file.name,
        transimg=image_path,
        size=(X, Y),
        calc_device=devices,
        learningrate=Lr,
        transrate=Tr,
        iterations_per_second=Fps,
    )
    return output


theme = gr.themes.Soft()
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# MetaMusic —— 一款基于Wav2clip指导与VQ-GAN生成模型架构的音乐印象图生成系统")
    with gr.Tab("音乐印象图生成"):
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    music2pic_music_input = gr.File(
                        label="Music",
                    )
                    with gr.Accordion("Image Transfer (Optional)"):  # 可折叠的组件
                        music2pic_image_input_path = gr.Image(type="filepath")
                        Tr_bar_for_image = gr.Slider(
                            0,
                            8,
                            value=default_tr,
                            label="Image Transfer Rate",
                            info="Choose between 0 and 8",
                            interactive=True,
                        )
                    music2pic_devices = gr.Dropdown(
                        available_devices,
                        value=available_devices[-1],
                        label="Devices",
                    )
                    It_bar_for_image = gr.Slider(
                        1,
                        200,
                        value=default_it,
                        label="Number of iterations",
                        info="Choose between 1 and 200",
                        step=1,
                        interactive=True,
                    )
                    Lr_bar_for_image = gr.Slider(
                        0.001,
                        1,
                        value=default_lr,
                        label="Generate Learning Rate",
                        info="Choose between 0.001 and 1",
                        interactive=True,
                    )
                    X_size_bar_for_image = gr.Slider(
                        0,
                        2048,
                        value=default_image_size,
                        label="Image Output X-size",
                        info="Choose between 0 and 2048",
                        step=1,
                        interactive=True,
                    )
                    Y_size_bar_for_image = gr.Slider(
                        0,
                        2048,
                        value=default_image_size,
                        label="Image Output Y-size",
                        info="Choose between 0 and 2048",
                        step=1,
                        interactive=True,
                    )
                with gr.Column():
                    music2pic_output = gr.Image()
            with gr.Row():
                music2pic_button = gr.Button("Generate")
    with gr.Tab("音乐视频生成"):
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    music2video_music_input = gr.File(
                        label="Music",
                    )
                    with gr.Accordion("Image Transfer (Optional)"):  # 可折叠的组件
                        music2video_image_input_path = gr.Image(type="filepath")
                        Tr_bar_for_video = gr.Slider(
                            0,
                            8,
                            value=default_tr,
                            label="Image Transfer Rate",
                            info="Choose between 0 and 8",
                            interactive=True,
                        )
                    music2video_devices = gr.Dropdown(
                        available_devices,
                        value=available_devices[-1],
                        label="Devices",
                    )
                    Fps_bar_for_video = gr.Slider(
                        1,
                        120,
                        value=default_fps,
                        label="Fps of Video",
                        info="Choose between 1 and 120",
                        step=1,
                        interactive=True,
                    )
                    Lr_bar_for_video = gr.Slider(
                        0.001,
                        1,
                        value=default_lr,
                        label="Generate Learning Rate",
                        info="Choose between 0.001 and 1",
                        interactive=True,
                    )
                    X_size_bar_for_video = gr.Slider(
                        0,
                        2048,
                        value=default_image_size,
                        label="Video Output X-size",
                        info="Choose between 0 and 2048",
                        step=1,
                        interactive=True,
                    )
                    Y_size_bar_for_video = gr.Slider(
                        0,
                        2048,
                        value=default_image_size,
                        label="Video Output Y-size",
                        info="Choose between 0 and 2048",
                        step=1,
                        interactive=True,
                    )
                with gr.Column():
                    music2video_output = gr.Video()
            with gr.Row():
                music2video_button = gr.Button("Generate")

    gr.Markdown("## Made by MetaMusic")

    music2pic_button.click(
        Generate_img,
        inputs=[
            music2pic_music_input,
            music2pic_image_input_path,
            X_size_bar_for_image,
            Y_size_bar_for_image,
            music2pic_devices,
            Tr_bar_for_image,
            Lr_bar_for_image,
            It_bar_for_image,
        ],
        outputs=music2pic_output,
    )
    music2video_button.click(
        Generate_video,
        inputs=[
            music2video_music_input,
            music2video_image_input_path,
            X_size_bar_for_image,
            Y_size_bar_for_image,
            music2video_devices,
            Tr_bar_for_video,
            Lr_bar_for_video,
            Fps_bar_for_video,
        ],
        outputs=music2video_output,
    )

if __name__ == "__main__":
    demo.launch()
