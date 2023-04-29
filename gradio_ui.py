import gradio as gr

from predict_caption import predict_step

with gr.Blocks() as demo:
    image = gr.Image(type='pil', label='Image')
    label = gr.Text(label='Generated Caption')
    image.upload(
        predict_step,
        [image],
        [label]
    )

if __name__ == '__main__':
    demo.launch()
