import json
from transformers import AutoModelForCausalLM, AutoTokenizer


''' Bigcode starcoder '''
checkpoint = "bigcode/starcoder2-3b"
device = "mps"

# Načítanie modelu a tokenizeru s podporou FIM
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
})

model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

''' Load dataset from json file '''
def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

''' Use mode to complete code in between of prefix and suffix '''


def generate_code(prefix, suffix, model, tokenizer):
    prompt = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=50,
    )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    middle_only = extract_middle(generated, prefix, suffix)
    return middle_only

# def extract_middle(generated: str, prefix: str, suffix: str):
#     # Nájdi začiatok <fim_middle>
#     if "<fim_middle>" in generated:
#         middle_part = generated.split("<fim_middle>")[1]
#     else:
#         middle_part = generated
#
#     # Odstráň suffix, ak ho skopíroval
#     if suffix in middle_part:
#         middle_part = middle_part.split(suffix)[0]
#
#     # Odstráň prefix, ak tam omylom ostal
#     if prefix in middle_part:
#         middle_part = middle_part.replace(prefix, "")
#
#     # Odstráň zvyšné špeciálne tokeny a whitespace
#     return middle_part.replace("<|endoftext|>", "").strip()

def extract_middle(generated: str, prefix: str, suffix: str):
    # 1. Nájdeš <fim_middle> a vezmeš to, čo nasleduje
    if "<fim_middle>" in generated:
        middle_part = generated.split("<fim_middle>", 1)[1]
    else:
        middle_part = generated

    # 2. Ak generovanie pokračuje ďalším promptom (čo je bug), odseknúť to
    for stop_token in ["<fim_prefix>", "<fim_suffix>", "<file_sep>", "<|endoftext|>"]:
        if stop_token in middle_part:
            middle_part = middle_part.split(stop_token, 1)[0]

    return middle_part.strip()


if __name__ == "__main__":
    dataset_path = "dataset.json" # path to dataset file
    dataset = load_dataset(dataset_path)

    for sample in dataset[:5]: # test on first sample
        prefix = sample["prefix"]
        suffix = sample["suffix"]
        real_middle = sample["middle"]

        generated_middle = generate_code(prefix, suffix, model, tokenizer)

        print("-" * 50)
        print("🔹 Ground truth Middle:\n", real_middle)
        print("🔹 Model Generated Middle:\n", generated_middle)
