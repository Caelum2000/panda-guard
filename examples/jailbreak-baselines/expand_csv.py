import pandas as pd
import json

# Specify the input CSV file and output CSV file
input_csv_file = 'data/sos_lite_expanded_test.csv'
output_csv_file = 'data/sos_lite_expanded_test.csv'
attacker_json_file = "benchmarks/sos_lite/S1-Base-Lite/ReNeLLMAttacker_ReNeLLM/NoneDefender/results.json"

# new column name, usually attacker name
new_column_name = "ReNeLLM"

# extract attack prompts from attacker json
with open(attacker_json_file, 'r') as file:
    attacker_json = json.load(file)

new_column_data = []

for item in attacker_json['results']:
    new_column_data.append(item['data'][0]['messages'][0]['content'])


# write
try:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv_file)

    # Check if the DataFrames have the same number of rows
    if len(df) != len(new_column_data):
        raise ValueError("The number of rows in the CSV and new column data must match.")

    # Append the new column to the DataFrame
    df[new_column_name] = new_column_data

    # Write the result to a new CSV file
    df.to_csv(output_csv_file, index=False, encoding='utf-8')

    print(f"Successfully appended new column to {input_csv_file} and saved to {output_csv_file}")

except FileNotFoundError:
    print(f"Error: The file {input_csv_file} was not found.")
except ValueError as e:
    print(f"Error: {str(e)}")
except Exception as e:
    print(f"An error occurred: {str(e)}")