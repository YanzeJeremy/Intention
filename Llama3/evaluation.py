import pandas as pd
from ast import literal_eval
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the CSV file
file_path = 'filtered_results.csv'
df = pd.read_csv(file_path)

# Clean the `desired_output` and `filtered_output` columns with error handling
def clean_labels(label):
    try:
        # Handle cases where the label is a single string (not in list format)
        if isinstance(label, str):
            # Attempt to parse as a list using literal_eval if it starts with '['
            if label.startswith('[') and label.endswith(']'):
                label = literal_eval(label)
            # Otherwise, treat as a comma-separated string
            else:
                label = [l.strip().capitalize() for l in label.split(',')]
        # Ensure all items in list are capitalized and whitespace-free
        return [item.strip().capitalize() for item in label]
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing label: {label}, Error: {e}")
        return []

df['desired_output'] = df['desired_output'].apply(clean_labels)
df['filtered_output'] = df['filtered_output'].apply(clean_labels)

# Flatten labels for multi-label binary evaluation
all_labels = sorted(set(label for labels in df['desired_output'] for label in labels).union(
                    set(label for labels in df['filtered_output'] for label in labels)))

# Create binary matrices for true and predicted labels
y_true = [[1 if label in labels else 0 for label in all_labels] for labels in df['desired_output']]
y_pred = [[1 if label in labels else 0 for label in all_labels] for labels in df['filtered_output']]

# Calculate precision, recall, and F1-score
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
