import json
from transformers import AutoModelForCausalLM, AutoTokenizer


''' Tiny starcoder '''
checkpoint = "bigcode/tiny_starcoder_py"
device = "cpu"

''' Load dataset from json file '''
def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

''' Use mode to complete code in between of prefix and suffix '''
def generate_code(prefix, suffix, model, tokenizer):
    prompt = prefix
    tokenizer.pad_token = tokenizer.eos_token  # Pad token setting

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=100)

    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Addition of attention mask
        max_new_tokens=50,
        pad_token_id=tokenizer.pad_token_id  # usage of pad token
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    ''' Extract only newly generated text '''
    middle_generated = generated_text[len(prompt):]
    return middle_generated.strip()


if __name__ == "__main__":
    dataset_path = "dataset.json" # path to dataset file
    dataset = load_dataset(dataset_path)

    ''' Hugging Face Tokenizer '''
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    for sample in dataset[:3]: # test on first 3 samples
        prefix = sample["prefix"]
        suffix = sample["suffix"]
        real_middle = sample["middle"]

        generated_middle = generate_code(prefix, suffix, model, tokenizer)

        print("=" * 50)
        print("🔹 Prefix:\n", prefix)
        print("🔹 Ground truth Middle:\n", real_middle)
        print("🔹 Model Generated Middle:\n", generated_middle)