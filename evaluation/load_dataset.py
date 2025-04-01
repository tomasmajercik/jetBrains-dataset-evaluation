import json

dataset_file = "dataset_vI.json"

with open(dataset_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

for i, sample in enumerate(dataset[:3]):
    print(f"Sample {i+1}:")
    print(f"  Prefix: {sample['prefix'][:50]}...")  # Shortened print (only to test)
    print(f"  Middle: {sample['middle'][:50]}...")
    print(f"  Suffix: {sample['suffix'][:50]}...")
    print("-" * 50)

print(f"\nNumber of samples: {len(dataset)}")