import json

with open('/home/pawel/Documents/RISA/sem1/SW/Licence-Plate-Recognition/results.json') as f:
    data = json.load(f)

total_same_chars = 0
total_chars = 0

for key, value in data.items():
    key_without_ext = key.split(".")[0]  # Remove .jpg extension from the key
    same_chars = sum(c1 == c2 for c1, c2 in zip(key_without_ext, value))
    total_same_chars += same_chars
    total_chars += len(key_without_ext)

result = total_same_chars / total_chars
print(result)