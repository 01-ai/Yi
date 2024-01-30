"""A simple web interactive chat demo based on gradio."""

import os
from argparse import ArgumentParser
from threading import Thread

import gradio as gr
import torch
from llava.conversation import conv_templates, default_conversation
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
)
from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
from transformers import TextIteratorStreamer

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)


def load_model_tokenizer_processor(args):
    model_path = os.path.expanduser(args.model_path)
    key_info["model_path"] = model_path
    get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path)

    return model, tokenizer, image_processor


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


title_markdown = """
# <p align="center" ><img src="https://raw.githubusercontent.com/01-ai/Yi/main/assets/img/Yi_logo_icon_light.svg" style="height: 50px"/> </p> &nbsp; Yi Vision Language Model
[[GitHub](https://github.com/01-ai/Yi/tree/main/VL)] | [[🤗](https://huggingface.co/01-ai/Yi-VL-34B)] [[🤖](https://www.modelscope.cn/models/01ai/Yi-VL-34B/summary)] | 📚 [[MMMU](https://arxiv.org/abs/2311.16502)] [[CMMMU](https://arxiv.org/abs/2401.11944)]
"""

learn_more_markdown = """
### License
Please refer to the acknowledgments and attributions as well as individual components, for the license of source code. The Yi series models are fully open for academic research and free for commercial use, permissions of which are automatically granted upon application. All usage must adhere to the [Yi Series Models Community License Agreement 2.1](https://huggingface.co/01-ai/Yi-VL-34B/blob/main/LICENSE). For free commercial use, you only need to send an email to get official commercial permission.
"""

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""


def launch_demo(args, yi_model, tokenizer, image_processor):
    textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", container=False
    )

    def predict(state, temperature, top_p, max_new_tokens):
        model = yi_model

        if len(state.messages) == state.offset + 2:
            new_state = conv_templates["mm_default"].copy()
            new_state.append_message(new_state.roles[0], state.messages[-2][1])
            new_state.append_message(new_state.roles[1], None)
            state = new_state

        prompt = state.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        images = state.get_images(return_pil=True)
        image = images[0]

        image_tensor = image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        stop_str = state.sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(
            tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True
        )
        generate_kwargs = {
            "input_ids": input_ids,
            "images": image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
            "streamer": streamer,
            "do_sample": True,
            "top_p": float(top_p),
            "temperature": float(temperature),
            "stopping_criteria": [stopping_criteria],
            "use_cache": True,
            "max_new_tokens": min(int(max_new_tokens), 1536),
        }
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[: -len(stop_str)]
            state.messages[-1][-1] = _parse_text(generated_text)

            yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2

    def regenerate(state, image_process_mode):
        state.messages[-1][-1] = None
        prev_human_msg = state.messages[-2]
        if type(prev_human_msg[1]) in (tuple, list):
            prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
        state.skip_next = False
        return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 2

    def clear_history():
        state = default_conversation.copy()
        return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 2

    def add_text(state, text, image, image_process_mode):
        text = text[:1536]  # Hard cut-off
        if image is not None:
            text = text[:1200]  # Hard cut-off for images
            if DEFAULT_IMAGE_TOKEN not in text:
                text = DEFAULT_IMAGE_TOKEN + "\n" + text
            text = (text, image, image_process_mode)
            state = conv_templates["mm_default"].copy()

        state.append_message(state.roles[0], text)
        state.append_message(state.roles[1], None)
        state.skip_next = False
        return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 2

    with gr.Blocks(title="Yi-VL", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Pad",
                    label="Preprocess for non-square image",
                    visible=False,
                )

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(
                    examples=[
                        [
                            f"{cur_dir}/images/cats.jpg",
                            "Describe the cats and what they are doing in detail.",
                        ],
                        [
                            f"{cur_dir}/images/extreme_ironing.jpg",
                            "What is unusual about this image?",
                        ],
                    ],
                    inputs=[imagebox, textbox],
                )

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=1024,
                        value=512,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(elem_id="chatbot", label="Yi-VL-Chat", height=550)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(
                        value="🔄  Regenerate", interactive=False
                    )
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        gr.Markdown(learn_more_markdown)
        # Register listeners
        btn_list = [regenerate_btn, clear_btn]

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False,
        )

        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False,
        ).then(
            predict,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False,
        ).then(
            predict,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            show_progress=True,
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False,
        ).then(
            predict,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            show_progress=True,
        )

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main(args):
    model, tokenizer, image_processor = load_model_tokenizer_processor(args)

    launch_demo(args, model, tokenizer, image_processor)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="01-ai/Yi-VL-6B",
        help="model-path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=True,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=8111, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )

    args = parser.parse_args()

    main(args)
