import os
import subprocess
import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig

def main():
    if not os.path.exists("config.yaml"):
        raise FileNotFoundError("config.yaml not found.")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    preset_name = config["active_preset"]
    preset = config["presets"][preset_name]
    stage = config["stage"]
    hf_id = preset["hf_id"]
    output_dir = config["output_dir"]

    print(f"=== Executing Stage: [{stage.upper()}] | Model Preset: [{preset_name}] ({hf_id}) ===")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    model.gradient_checkpointing_enable()

    if stage == "sft":
        dataset = load_dataset("json", data_files=config["sft_dataset"])
        args = SFTConfig(
            output_dir=output_dir,
            dataset_text_field="text",
            max_seq_length=preset["max_seq_len"],
            packing=True,
            per_device_train_batch_size=preset["batch_size"],
            gradient_accumulation_steps=preset["grad_accum"],
            learning_rate=preset["learning_rate"],
            num_train_epochs=3,
            use_cpu=True,
            bf16=True,
            optim="adafactor",
            save_strategy="no"
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset["train"],
            processing_class=tokenizer,
        )
        trainer.train()

    elif stage == "dpo":
        dataset = load_dataset("json", data_files=config["dpo_dataset"])
        args = DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=preset["batch_size"],
            gradient_accumulation_steps=preset["grad_accum"],
            learning_rate=5e-7,
            beta=0.1,
            max_steps=100,
            use_cpu=True,
            bf16=True,
            optimizer="adafactor",
            save_strategy="no"
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=args,
            train_dataset=dataset["train"],
            processing_class=tokenizer,
        )
        trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("=== Training Step Finished. HF Checkpoint Saved. ===")

    # Quantization Pipeline
    gguf_out = config["gguf_output"]
    print("=== Compiling llama.cpp & Quantizing to GGUF ===")

    if not os.path.exists("llama.cpp"):
        subprocess.run(["git", "clone", "https://github.com/ggml-org/llama.cpp"], check=True)
        subprocess.run(["cmake", "-B", "llama.cpp/build"], check=True)
        subprocess.run(["cmake", "--build", "llama.cpp/build", "--config", "Release", "-j"], check=True)

    temp_bf16 = "temp_bf16.gguf"
    subprocess.run(["python3", "llama.cpp/convert_hf_to_gguf.py", output_dir, "--outfile", temp_bf16, "--outtype", "bf16"], check=True)
    subprocess.run(["./llama.cpp/build/bin/llama-quantize", temp_bf16, gguf_out, "Q4_K_M"], check=True)

    if os.path.exists(temp_bf16):
        os.remove(temp_bf16)

    print(f"=== SUCCESS: Download {gguf_out} to laptop ===")

if __name__ == "__main__":
    main()