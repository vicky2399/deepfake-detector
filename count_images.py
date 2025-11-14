# count_images.py
import os

root = r"D:\DeepFakeProjectNsic\dataset"

def count(folder):
    counts = {}
    for sub in ('Real','real','Fake','fake'):
        p = os.path.join(folder, sub)
        if os.path.exists(p):
            files = [f for f in os.listdir(p) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            counts[sub] = len(files)
        else:
            counts[sub] = 0
    return counts

parts = ['Train','Validation','Test']
print("Counts per split (Train / Validation / Test):")
total = 0
for part in parts:
    folder = os.path.join(root, part)
    if not os.path.exists(folder):
        print(f"{part}: folder not found")
        continue
    c = count(folder)
    print(f"\n{part}:")
    for k,v in c.items():
        print(f"  {k}: {v}")
        total += v
print(f"\nTotal images counted: {total}")
