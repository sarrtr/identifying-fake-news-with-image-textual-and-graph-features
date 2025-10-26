import pandas as pd
import random

def generate_random_labels_dataset(input_file, output_file, n):
    """
    Generate a TSV file with n random rows containing IDs and random labels (0 or 1)
    
    Args:
        input_file (str): Path to the input TSV file
        output_file (str): Path to the output TSV file
        n (int): Number of rows to generate
    """
    # Read the dataset
    df = pd.read_csv(input_file, sep='\t')
    
    # Check if n is larger than dataset size
    if n > len(df):
        print(f"Warning: Requested {n} rows but dataset only has {len(df)} rows.")
        n = len(df)
    
    # Randomly sample n rows
    sampled_rows = df.sample(n)  # random_state for reproducibility
    
    # Create new dataframe with id and random label
    result_df = pd.DataFrame({
        'id': sampled_rows['id'],
        'label': [random.uniform(0, 1) for _ in range(n)]
    })
    
    # Save to TSV file
    result_df.to_csv(output_file, sep='\t', index=False)
    print(f"Generated {output_file} with {n} rows")

# Example usage
if __name__ == "__main__":
    input_filename = "multimodal_train.tsv"
    output_filename = "mock_predictions.tsv"
    
    # Get n from user input
    try:
        n = int(input("Enter the number of rows to generate (n): "))
        generate_random_labels_dataset(input_filename, output_filename, n)
    except ValueError:
        print("Please enter a valid integer for n.")
    except FileNotFoundError:
        print(f"Error: Could not find the input file '{input_filename}'")