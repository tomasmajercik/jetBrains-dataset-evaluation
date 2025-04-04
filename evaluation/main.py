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
        # pad_token_id=tokenizer.eos_token_id, #TODO: skus
        max_new_tokens=50,
    )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # print("🔍 Full generated sequence:\n", generated)

    # Odstrihni prefix a suffix
    start = generated.find(suffix)
    if start != -1:
        middle_part = generated[:start]
    else:
        middle_part = generated

    # Odstráň prefix ak ho skopíroval
    if prefix in middle_part:
        middle_part = middle_part.replace(prefix, "")

    return middle_part.strip()


if __name__ == "__main__":
    dataset_path = "dataset.json" # path to dataset file
    dataset = load_dataset(dataset_path)

    ''' Hugging Face Tokenizer '''
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    # tokenizer.add_special_tokens({
    #     "additional_special_tokens": ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
    # })
    #
    # model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
    # model.resize_token_embeddings(len(tokenizer))
    # model.to(device)

    for sample in dataset[:5]: # test on first sample
        prefix = sample["prefix"]
        suffix = sample["suffix"]
        real_middle = sample["middle"]

        generated_middle = generate_code(prefix, suffix, model, tokenizer)

        print("-" * 50)
        print("🔹 Ground truth Middle:\n", real_middle)
        print("🔹 Model Generated Middle:\n", generated_middle)
