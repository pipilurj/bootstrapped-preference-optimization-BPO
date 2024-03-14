import sys
import json

def merge_json(files):
    merged_data = {}
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            merged_data.update(data)
    return merged_data

if __name__ == "__main__":
    files = sys.argv[1:]
    merged_data = merge_json(files)
    with open('merge.json', 'w') as f:
        json.dump(merged_data, f, indent=4)
