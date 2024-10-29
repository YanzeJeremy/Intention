import json

# Define paths for the input files and the output merged file
train_file_path = 'data/split/casino_train.json'
test_file_path = 'data/split/casino_test.json'
valid_file_path = 'data/split/casino_valid.json'
output_file_path = 'merged_casino_dataset.json'

# Load all datasets
with open(train_file_path, 'r') as f:
    train_data = json.load(f)

with open(test_file_path, 'r') as f:
    test_data = json.load(f)

with open(valid_file_path, 'r') as f:
    valid_data = json.load(f)

# Prepare the data with the prompt in the desired format
merged_data = {"train": [], "valid": [], "test": []}
prompt_template = "Classify the following message: '{}'."


def process_data(data, split_name):
    id_counter = 0
    for dialogue in data:
        if "annotations" in dialogue:
            for annotation in dialogue["annotations"]:
                text = annotation[0]
                label = annotation[1]
                prompt = prompt_template.format(text)
                merged_entry = {
                    "id": id_counter,
                    "text": f"<s>[INST] <<SYS>>You are an excellent assistant capable of classifying different user intentions. You will be given an utterance of dialogue between two individuals, classify the utterance into one or more strategies<</SYS>> {prompt} [/INST] {label} </s>"
                }
                merged_data[split_name].append(merged_entry)
                id_counter += 1

# Process each dataset split and add to the merged_data
process_data(train_data, "train")
process_data(valid_data, "valid")
process_data(test_data, "test")

# Save merged data to output file
with open(output_file_path, 'w') as f:
    json.dump(merged_data, f, indent=2)

print(f"Merged dataset saved to {output_file_path}")