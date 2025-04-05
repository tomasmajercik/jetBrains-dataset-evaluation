### This is an jetbrains internship application task, for code competion for my own projects. 

# Walkthrough:
- `/data_files_vI/II/III` is a folder of files from three of my projects marked as data_files and verion from 1 to 3.
- `/datasets` is a folder containing datasets created by `generate_dataset.py` where it randomly selects part of a code that will be completed by AI model. There are `.json` files containing from 20 to 50 records with `"prefix"` the code before hidden part, `"middle"` the hidden part and `"suffix"` the part after middle part
- `/evaluation` is a folder containing python project with `main.py` in it. This is a script that implements the AI model `bigcode/starcoder2-3b` from hugging face that completes the hidden part and as a result create `.json` file containg `prefix`, `suffix` and `original middle`, `generated middle` then `exact_match`, `chrf`, `levenshtein_distance` which are computed metrics. 
- `generate_dataset.py` is script that generates the datasets
- `load_dataset.py` is script that loads dataset (only for testing purposes)
- `report.pdf` is a report describing my thought process, findings and learnings
- `resultingDataset-wAnotations.json` is the resulting dataset with annotations and computed metrics. The `"label"` says if the generated code is exact match and therefor is correct, whether the code is different but will work or if the generated code is wrong and will lead to an error 

# Dependences
- `pip install transformers`
- `pip install evaluate`
- `pip install python-Levenshtein`
- `pip install sacrebleu`