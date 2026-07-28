# Nano LLM Knowledge Engine

A zero-compiler, CPU-optimized pipeline for full-parameter factual knowledge injection and low-friction DPO preference alignment.

## System Workflow

[ Laptop ]      1. python -m preprocess   (HTML Export  -->  my_knowledge.jsonl)
│
[ Codespaces ]  2. python -m train        (JSONL Data   -->  my_brain.gguf)
│
[ Laptop ]      3. python -m infer        (.gguf Model  -->  Dual-Option Chat & DPO Logger)


---

## Quick Start (Validation Run)

### Step 1: Preprocess Data (Laptop)

Place your extracted HTML messages file (`extracted_messages.html`) in the project root directory.

```bash
# Activate venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Run preprocessing
python -m preprocess

Output: Creates my_knowledge.jsonl. Commit and push to GitHub.
Step 2: Knowledge Injection & Quantization (Codespaces)

    Open your repository in GitHub Codespaces.

    Ensure config.yaml is set to active_preset: "nano" and stage: "sft".

    Run training:

Bash

# Activate venv
source venv/bin/activate

# Run training
python -m train
```

Output: Performs full SFT in BF16, compiles llama.cpp, quantizes the checkpoint to Q4_K_M, and outputs my_brain.gguf. Download my_brain.gguf to your laptop project root.
Step 3: Local Chat & Preference Logging (Laptop)

Ensure my_brain.gguf is in your laptop's project root directory.

Run the local inference server:


`python -m infer`

Open http://127.0.0.1:7860 in your browser.

Input prompts to generate dual candidate completions. Select or edit an option to automatically append preference tuples to dpo_pairs.jsonl.

Weekly DPO Alignment Run

When you have built up entries in dpo_pairs.jsonl:

Upload dpo_pairs.jsonl to your GitHub repository or Codespaces instance.

In config.yaml, update the stage:


`stage: "dpo"`

Run training on Codespaces:


`python -m train`

Download the updated my_brain.gguf back to your laptop and replace the old file.

Scaling to Production

To switch model parameter scale (e.g., from nano to large), update active_preset in config.yaml:
YAML

active_preset: "large" # Target: Qwen2.5-7B on 64 GB RAM