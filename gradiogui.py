import gradio as gr
import api_picture


theme=gr.themes.Soft()

def Generate_img(file_obj):
    output=api_picture.generate(filemusic=file_obj.name)
    return output
    

theme=gr.themes.Soft()
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("MetaMusic —— 一款基于Wav2clip架构与VQ-GAN生成模型的音乐印象图生成系统")
    with gr.Tab("音乐印象图生成"):
        music2pic_input = gr.File()
        music2pic_output = gr.Image()
        music2pic_button = gr.Button("Flip")
    with gr.Tab("音乐视频生成"):
        with gr.Row():
            music2video_input = gr.Image()
            music2video_output = gr.Video()
        music2video_button = gr.Button("Flip")

    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at me...")

    music2pic_button.click(Generate_img, inputs=music2pic_input, outputs=music2pic_output)
    music2video_button.click(Generate_img, inputs=music2video_input, outputs=music2video_output)

if __name__ == "__main__":
    demo.launch()
