import gradio as gr
import api_picture
import api_video

theme = gr.themes.Soft()


def Generate_img(music_file, image_path):
    output = api_picture.generate(filemusic=music_file.name, init_image=image_path)
    return output

def Generate_video(music_file, image_path):
    output = api_video.generate(filemusic=music_file.name, init_image=image_path)
    return output

def Get_init_value():
    return 512

theme = gr.themes.Soft()
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("MetaMusic —— 一款基于Wav2clip架构与VQ-GAN生成模型的音乐印象图生成系统")
    with gr.Tab("音乐印象图生成"):
        music2pic_music_input_for_image = gr.File()
        music2pic_image_input_path_for_image = gr.Image(type="filepath")
        X_size_bar_for_image = gr.Slider(0, 2048, value=Get_init_value(), label="Input wanted X-size", info="Choose between 0 and 2048", step=1, interactive=True)
        Y_size_bar_for_image = gr.Slider(0, 2048, value=Get_init_value(), label="Input wanted Y-size", info="Choose between 0 and 2048", step=1, interactive=True)
        music2pic_output = gr.Image()
        music2pic_button = gr.Button("Flip")
    with gr.Tab("音乐视频生成"):
        music2pic_music_input_for_video = gr.File()
        music2pic_image_input_path_for_video = gr.Image(type="filepath")
        X_size_bar_for_video = gr.Slider(0, 2048, value=Get_init_value(), label="Input wanted X-size", info="Choose between 0 and 2048", step=1, interactive=True)
        Y_size_bar_for_video = gr.Slider(0, 2048, value=Get_init_value(), label="Input wanted Y-size", info="Choose between 0 and 2048", step=1, interactive=True)
        music2video_output = gr.Video()
        music2video_button = gr.Button("Flip")

    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at me...")

    music2pic_button.click(Generate_img, inputs=[music2pic_music_input_for_image, music2pic_image_input_path_for_image], outputs=music2pic_output)
    music2video_button.click(Generate_img, inputs=[music2pic_music_input_for_video, music2pic_image_input_path_for_video], outputs=music2video_output)

if __name__ == "__main__":
    demo.launch()
