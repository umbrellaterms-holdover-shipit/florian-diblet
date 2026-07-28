import os
import json
import yaml
import gradio as gr
from llama_cpp import Llama

def main():
    if not os.path.exists("config.yaml"):
        raise FileNotFoundError("config.yaml not found.")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    gguf_file = config["gguf_output"]
    dpo_log_file = config["dpo_dataset"]

    print(f"=== Loading GGUF Model: {gguf_file} ===")
    llm = Llama(model_path=gguf_file, n_ctx=2048, verbose=False)

    def generate_options(prompt):
        res_a = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=256
        )["choices"][0]["message"]["content"]

        res_b = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85,
            max_tokens=256
        )["choices"][0]["message"]["content"]

        return res_a, res_b

    def log_preference(prompt, chosen, rejected):
        entry = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        with open(dpo_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return "Logged preference entry!"

    with gr.Blocks(title="Preference Grading UI") as demo:
        gr.Markdown("## Dual-Option Preference Collector")
        
        prompt_input = gr.Textbox(label="User Prompt", lines=2)
        gen_btn = gr.Button("Generate Candidates", variant="primary")

        with gr.Row():
            with gr.Column():
                draft_a = gr.Textbox(label="Option A", lines=6)
                btn_a = gr.Button("Pick Option A")
            
            with gr.Column():
                draft_b = gr.Textbox(label="Option B (Editable)", lines=6, interactive=True)
                btn_b = gr.Button("Pick Option B (or Edits)")

        status_output = gr.Textbox(label="Status", interactive=False)

        gen_btn.click(
            generate_options, 
            inputs=[prompt_input], 
            outputs=[draft_a, draft_b]
        )

        btn_a.click(
            lambda p, a, b: log_preference(p, a, b),
            inputs=[prompt_input, draft_a, draft_b],
            outputs=[status_output]
        )

        btn_b.click(
            lambda p, a, b: log_preference(p, b, a),
            inputs=[prompt_input, draft_b, draft_a],
            outputs=[status_output]
        )

    demo.launch(server_name="127.0.0.1", port=7860)

if __name__ == "__main__":
    main()