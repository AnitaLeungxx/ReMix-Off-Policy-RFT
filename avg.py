import os
import json
import numpy as np  # Use numpy for more stable mean calculation

# ==============================================================================
# 1. Configure the path to your root log directory here (the only place you need to change)
# ==============================================================================
LOGGER_DIRECTORY = './tmp/logs'  # Replace with the actual path of your "logger" folder

# The name of the JSON field from which you want to extract scores
bench_name = os.getenv("DATA", "math")
TARGET_KEY = f"val/{bench_name}/is_correct_finalanswer_score"

# ==============================================================================
# 2. Core processing logic (no need to edit the code below)
# ==============================================================================

def calculate_average_from_logs(log_dir: str, key: str):
    """
    Traverse all sub-folders under the log directory, read logs.json files,
    extract the specified key's value, and compute the average score.
    """
    if not os.path.isdir(log_dir):
        print(f"Error: Directory '{log_dir}' not found. Please check the path.")
        return

    scores = []
    print(f"--- Scanning directory: {os.path.abspath(log_dir)} ---\n")

    # Iterate over all entries in the logger directory
    for experiment_folder_name in os.listdir(log_dir):
        experiment_path = os.path.join(log_dir, experiment_folder_name)

        # Ensure it is a directory
        if os.path.isdir(experiment_path):
            log_file_path = os.path.join(experiment_path, 'logs.json')

            # Check if logs.json exists
            if not os.path.isfile(log_file_path):
                print(f"Warning: 'logs.json' not found in folder '{experiment_folder_name}', skipped.")
                continue

            # Read and parse the JSON file
            try:
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract the score
                # The target field is a list; we need its first element
                if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                    score = data[key][0]
                    scores.append(score)
                    print(f"Successfully extracted score from '{experiment_folder_name}': {score:.4f}")
                else:
                    print(f"Warning: Field '{key}' not found or invalid format in '{log_file_path}', skipped.")

            except json.JSONDecodeError:
                print(f"Error: File '{log_file_path}' is not valid JSON, skipped.")
            except Exception as e:
                print(f"Unknown error occurred while processing file '{log_file_path}': {e}")

    # --- Compute and print the final result ---
    print("\n--- Scan complete ---")

    if not scores:
        print("No valid scores were extracted from any file.")
        return

    num_files_processed = len(scores)
    average_score = np.mean(scores)
    std_dev = np.std(scores)

    print("\n=================================================")
    print(f"  Final Result (based on {num_files_processed} experiments)")
    print("=================================================")
    print(f"  Average Score: {average_score:.6f}")
    print(f"  Standard Deviation: {std_dev:.6f}")
    print("=================================================")


if __name__ == '__main__':
    calculate_average_from_logs(LOGGER_DIRECTORY, TARGET_KEY)