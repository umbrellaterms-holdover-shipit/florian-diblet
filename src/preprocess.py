import re
import html
import json
import yaml
from pathlib import Path

def clean_element_text(raw_html):
    # Convert HTML line breaks to newline characters
    text = re.sub(r'<br\s*/?>', '\n', raw_html, flags=re.IGNORECASE)
    # Strip away all remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Unescape HTML entities (e.g., &#x27; -> ', &quot; -> ")
    text = html.unescape(text)
    # Normalize excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_html_path = Path(config["raw_html_file"])
    output_jsonl_path = Path(config["sft_dataset"])

    if not raw_html_path.exists():
        raise FileNotFoundError(f"Raw HTML file not found: {raw_html_path}")

    print(f"=== Preprocessing {raw_html_path} -> {output_jsonl_path} ===")

    with open(raw_html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract conversation blocks by <h2> headers
    conversations = re.split(r'<h2[^>]*>', content)[1:]
    
    total_samples = 0
    with open(output_jsonl_path, "w", encoding="utf-8") as out_f:
        for conv in conversations:
            # Find all <p> text blocks inside the conversation
            p_blocks = re.findall(r'<p[^>]*>(.*?)</p>', conv, flags=re.DOTALL)
            
            cleaned_messages = []
            for p in p_blocks:
                cleaned = clean_element_text(p)
                if cleaned:
                    cleaned_messages.append(cleaned)

            if cleaned_messages:
                # Group conversation turns into a unified context chunk
                full_conversation_text = "\n\n".join(cleaned_messages)
                out_f.write(json.dumps({"text": full_conversation_text}, ensure_ascii=False) + "\n")
                total_samples += 1

    print(f"=== Complete: Extracted {total_samples} conversation documents into {output_jsonl_path} ===")

if __name__ == "__main__":
    main()