# Load the CSV data from the provided file path
import pandas as pd
import re
file_path = 'llama3_test.csv'
data = pd.read_csv(file_path)

# Define classification categories
categories = [
    "Small-Talk", "Empathy", "Coordination", "No-Need", "Elicit-Pref", 
    "Undervalue-Partner", "Vouch-Fairness", "Self-Need", "Other-Need", "Non-strategic"
]

# Function to extract the classifications
def extract_classifications(text):
    # Using regex to extract the desired classifications based on predefined categories
    extracted = set(re.findall(r'\b(?:' + '|'.join(categories) + r')\b', text, flags=re.IGNORECASE))
    # Normalizing extracted labels to match the case in categories
    return [category for category in categories if category.lower() in extracted]

# Apply the extraction function to model_output
data['filtered_output'] = data['model_output'].apply(extract_classifications)

# Compare filtered output with desired output
data['comparison'] = data.apply(lambda row: set(row['filtered_output']) == set(row['desired_output'].split()), axis=1)

# Save the filtered results into a new CSV file with the specified columns
filtered_data = data[['id', 'prompt', 'desired_output', 'filtered_output']]
filtered_data.to_csv('filtered_results.csv', index=False)
