import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
import Levenshtein

## dependencies
# pip install transformers
# pip install evaluate
# pip install python-Levenshtein
# pip install sacrebleu

''' Bigcode starcoder '''
checkpoint = "bigcode/starcoder2-3b"
device = "mps"

''' Load dataset and tokenizer (capable of Fill In the Middle) '''
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
        eos_token_id=tokenizer.convert_tokens_to_ids("<fim_suffix>"),
    )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    middle_only = extract_middle(generated)
    return middle_only

''' Extract only generated middle from generated dump '''
def extract_middle(generated: str):
    # Find <fim_middle> a take what is next
    if "<fim_middle>" in generated:
        middle_part = generated.split("<fim_middle>", 1)[1]
    else:
        middle_part = generated

    # If generating continues with another prompt (which is a bug), cut it out
    for stop_token in ["<fim_prefix>", "<fim_suffix>", "<file_sep>", "<|endoftext|>"]:
        if stop_token in middle_part:
            middle_part = middle_part.split(stop_token, 1)[0]

    return middle_part.strip()

''' Evaluation metrics '''
chrf_metric = evaluate.load("chrf")
def compute_metrics(real_output, generated_output):
    exact_match = real_output.strip() == generated_output.strip()
    chrf_score = chrf_metric.compute(references=[real_output], predictions=[generated_output])["score"]
    levenshtein_distance = Levenshtein.distance(real_output.strip(), generated_output.strip())
    return exact_match, chrf_score, levenshtein_distance


if __name__ == "__main__":
    dataset_path = "dataset.json" # path to dataset file
    output_path = "predictions.json"
    dataset = load_dataset(dataset_path)
    results = []

    for sample in dataset: # test on first sample
        prefix = sample["prefix"]
        suffix = sample["suffix"]
        real_middle = sample["middle"]

        generated_middle = generate_code(prefix, suffix, model, tokenizer)
        exact, chrf, lev = compute_metrics(real_middle, generated_middle)

        results.append({
            "prefix": prefix,
            "suffix": suffix,
            "real_middle": real_middle,
            "generated_middle": generated_middle,
            "exact_match": exact,
            "chrf": chrf,
            "levenshtein_distance": lev,
        })

        print("-" * 50)
        print("🔹 Real Middle:\n", real_middle)
        print("🔹 Model Generated Middle:\n", generated_middle)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete! predictions saved to '{output_path}'")
