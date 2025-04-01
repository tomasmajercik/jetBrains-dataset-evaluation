import os
import random
import json
import sys

def split_code(file_path, num_samples=5):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    samples = []
    code_length = len(code)
    
    for _ in range(num_samples):
        if code_length < 20:  # Preskočí príliš krátke súbory
            continue
        
        cursor_pos = random.randint(10, code_length - 10)  # Vyber miesto kurzora
        split_range = random.randint(15, 30)  # Koľko znakov „vynechať“
        
        prefix = code[:cursor_pos]
        middle = code[cursor_pos:cursor_pos + split_range]
        suffix = code[cursor_pos + split_range:]

        samples.append({"prefix": prefix, "middle": middle, "suffix": suffix})
    
    return samples

def process_files(folder, output_file="dataset.json"):
    dataset = []
    for filename in os.listdir(folder):
        if filename.endswith((".js", ".ts", ".jsx", ".tsx", ".py")):  # Použiteľné typy súborov
            file_path = os.path.join(folder, filename)
            dataset.extend(split_code(file_path))
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"\nDataset saved as {output_file}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("error: use python3 generate_dataset.py </path/to/folder>")
        sys.exit(1)
    
    
    path = sys.argv[1]

    if os.path.exists(path):
        print("\nPath is valid. \nCreating dataset...")
    else:
        sys.exit(0)

    process_files(path)