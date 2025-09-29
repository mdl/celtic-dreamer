# trim_dataset.py
import numpy as np
import argparse
import os


def trim_dataset(filepath, num_transitions_to_keep):
    """
    Loads a dataset, keeps only the specified number of initial transitions,
    and overwrites the original file with the cleaned data.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    print(f"Loading dataset from: {filepath}...")
    try:
        with np.load(filepath) as data:
            # Create a dictionary from the loaded data to make it mutable
            data_dict = {key: data[key] for key in data.keys()}
    except Exception as e:
        print(f"Error loading .npz file: {e}")
        return

    original_size = len(data_dict['observations'])
    print(f"Original dataset size: {original_size} transitions.")

    # if num_transitions_to_keep >= original_size:
    #     print("Number to keep is greater than or equal to original size. No changes made.")
    #     return

    print(f"Trimming dataset to the first {num_transitions_to_keep} transitions...")

    # Trim all the main data arrays
    data_dict['observations'] = data_dict['observations'][:num_transitions_to_keep]
    data_dict['actions'] = data_dict['actions'][:num_transitions_to_keep]
    data_dict['rewards'] = data_dict['rewards'][:num_transitions_to_keep]
    data_dict['dones'] = data_dict['dones'][:num_transitions_to_keep]

    # Update the buffer index and full status
    # The new index is simply the number of transitions we kept.
    # The buffer is no longer considered 'full'.
    data_dict['bufferIndex'] = np.array(num_transitions_to_keep, dtype=np.int32)
    data_dict['full'] = np.array(False, dtype=np.bool_)

    print(f"Saving trimmed dataset back to: {filepath}...")
    try:
        # Use savez_compressed to save the modified dictionary
        np.savez_compressed(filepath, **data_dict)
        print("Dataset trimmed and saved successfully!")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim a Dreamer replay buffer dataset.")
    parser.add_argument(
        "--file",
        type=str,
        default="celtic_heroes_dataset.npz",
        help="Path to the .npz dataset file."
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=30000,
        help="Number of initial transitions to keep."
    )
    args = parser.parse_args()

    trim_dataset(args.file, args.keep)
