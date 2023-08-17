from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer
import gradio as gr
import mdtex2html
import argparse
import time

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xpu", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="chatglm2-6b")
    args = parser.parse_args()
    return args

def postprocess(self, y):
    #print("postprocess")
    #print(y)
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
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
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(model, input, chatbot, max_length, top_p, temperature, history, past_key_values, chat_mode):
    if model == "model_fp16":
        model = model_fp16
    else:
        model = model_int4

    if chat_mode == True:
        chatbot.append((parse_text(input), ""))
        timeStart = time.time()
        timeFirst = 0
        timeFirstRecord = False
        for response, history, past_key_values in model.stream_chat(tokenizer, input, history, past_key_values=past_key_values,
                                                                    return_past_key_values=True,
                                                                    max_length=max_length, top_p=top_p,
                                                                    temperature=temperature):
            chatbot[-1] = (parse_text(input), parse_text(response))
            if timeFirstRecord == False:
                timeFirst = time.time() - timeStart
                timeFirstRecord = True
            yield chatbot, history, past_key_values, "", ""
        timeCost = time.time() - timeStart
        token_count_input = len(tokenizer.tokenize(input))
        token_count_output = len(tokenizer.tokenize(response))
        ms_first_token = timeFirst*1000
        ms_after_token = (timeCost-timeFirst)/(token_count_output-1)*1000
        print("input: ", input)
        print("output: ", parse_text(response))
        print("token count input: ", token_count_input)
        print("token count output: ", token_count_output)
        print("time cost(s): ", timeCost)
        print("1st token latency(ms): ", ms_first_token)
        print("After token latency(ms)", ms_after_token)
        yield chatbot, history, past_key_values, str(round(ms_first_token, 2)) + " ms", str(round(ms_after_token, 2)) + " ms"
    else:
        chatbot.append((parse_text(input), ""))
        # you could tune the prompt based on your own model,
        # here the prompt tuning refers to https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py#L1007
        CHATGLM_V2_PROMPT_FORMAT = "问：{prompt}\n\n答："
        prompt = CHATGLM_V2_PROMPT_FORMAT.format(prompt=input)
        #prompt = input
        timeStart = time.time()
        output = model.generate(tokenizer.encode(prompt, return_tensors="pt"), max_length=max_length, top_p=top_p, temperature=temperature)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        timeCost = time.time() - timeStart
        chatbot[-1] = (parse_text(input), response)
        token_count_input = len(tokenizer.tokenize(input))
        token_count_output = len(tokenizer.tokenize(response))
        ms_average_token = timeCost/token_count_output*1000
        print("input: ", input)
        print("output: ", response)
        print("token count input: ", token_count_input)
        print("token count output: ", token_count_output)
        print("time cost(s): ", timeCost)
        print("Overall(1st & after) token latency(ms)", ms_average_token)
        yield chatbot, [], None, "", str(round(ms_average_token, 2)) + " ms"

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], [], None, "", "", [], [], None, "", ""

def change_latency_ui(chat_mode):
    if chat_mode == True:
        return {
            f_latency_1: gr.update(visible=True, value=""),
            a_latency_1: gr.update(label="After Latency", value=""),
            f_latency_2: gr.update(visible=True, value=""),
            a_latency_2: gr.update(label="After Latency", value="")
        }
    else:
        return {
            f_latency_1: gr.update(visible=False, value=""),
            a_latency_1: gr.update(label="Average Latency", value=""),
            f_latency_2: gr.update(visible=False, value=""),
            a_latency_2: gr.update(label="Average Latency", value="")
        }

if __name__ == '__main__':
    global f_latency_1, a_latency_1, f_latency_2, a_latency_2, model_fp16, model_int4

    args = getArgs()
    xpu = args.xpu
    model_name_remote = "THUDM/" + args.model
    model_name_local = ".\\" + args.model

    tokenizer = AutoTokenizer.from_pretrained(model_name_local, trust_remote_code=True)
    model_fp16 = AutoModel.from_pretrained(model_name_local, trust_remote_code=True).float()
    model_fp16 = model_fp16.eval()
    model_int4 = AutoModel.from_pretrained(model_name_local, trust_remote_code=True, load_in_4bit=True)
    model_int4 = model_int4.eval()

    timeStart = 0
    """Override Chatbot.postprocess"""

    gr.Chatbot.postprocess = postprocess

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.HTML("""<h1 align="center">ChatGLM2-6B optimization for Intel Platform</h1>""")
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=1):
                    chatbot_1 = gr.Chatbot(label="CHATGLM2-6B fp16 (Original)")
                    with gr.Row(scale=1):
                        f_latency_1 = gr.Textbox(label="First Latency", visible=False)
                        a_latency_1 = gr.Textbox(label="Average Latency", visible=True)
                        submitBtn_1 = gr.Button("SUBMIT", variant="primary")
                with gr.Column(scale=1):
                    chatbot_2 = gr.Chatbot(label="ChatGLM2-6B int4 (opt for Intel)")
                    with gr.Row(scale=1):
                        f_latency_2 = gr.Textbox(label="First Latency", visible=False)
                        a_latency_2 = gr.Textbox(label="Average Latency", visible=True)
                        submitBtn_2 = gr.Button("SUBMIT", variant="primary")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Column(scale=12):
                        user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=9, container=False)
                with gr.Column(scale=1):
                    chat_mode = gr.Checkbox(label="Chat Mode",info="Real-time echo(additional latency per token)")
                    with gr.Row(scale=1):
                        max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="MaxLength", interactive=True)
                        temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
                        top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
                    with gr.Row(scale=1):
                        clearBtn = gr.Button("CLEAR")
                        resetBtn = gr.Button("RESET")

        #-- 1
        history_1 = gr.State([])
        past_key_values_1 = gr.State(None)
        model_1 = gr.State("model_fp16")

        submitBtn_1.click(predict, [model_1, user_input, chatbot_1, max_length, top_p, temperature, history_1, past_key_values_1, chat_mode],
                        [chatbot_1, history_1, past_key_values_1, f_latency_1, a_latency_1], show_progress=True)

        # -- 2
        history_2 = gr.State([])
        past_key_values_2 = gr.State(None)
        model_2 = gr.State("model_int4")

        submitBtn_2.click(predict, [model_2, user_input, chatbot_2, max_length, top_p, temperature, history_2, past_key_values_2, chat_mode],
                        [chatbot_2, history_2, past_key_values_2, f_latency_2, a_latency_2], show_progress=True)

        # -- common
        clearBtn.click(reset_user_input, [], [user_input])
        resetBtn.click(reset_state, outputs=[chatbot_1, history_1, past_key_values_1, f_latency_1, a_latency_1, chatbot_2, history_2, past_key_values_2, f_latency_2, a_latency_2], show_progress=True)
        chat_mode.select(change_latency_ui, [chat_mode], [f_latency_1, a_latency_1, f_latency_2, a_latency_2])


    demo.queue().launch(share=False, inbrowser=True)
