from datasets import load_dataset
import json

ds = load_dataset("nkazi/SciEntsBank")
train_ds = ds["train"]

labels = set()
examples = {}

for sample in train_ds:
    lbl = sample["label"]
    if lbl not in labels:
        labels.add(lbl)
        examples[lbl] = sample
        
print("Unique labels:", labels)
for lbl, ex in examples.items():
    print(f"Label {lbl}:")
    print(f"  Ref: {ex['reference_answer']}")
    print(f"  Stu: {ex['student_answer']}")
    print()
