import numpy as np
import pandas as pd

def normalized_cosine_similarity(e1, e2):
    """
    Compute the normalized cosine similarity between two emotion vectors.

    :param e1: First emotion vector (pleasure, arousal)
    :param e2: Second emotion vector (pleasure, arousal)
    :return: Normalized cosine similarity in range [0, 1]
    """
    e1, e2 = np.array(e1), np.array(e2)
    dot_product = np.dot(e1, e2)
    norm_e1 = np.linalg.norm(e1)
    norm_e2 = np.linalg.norm(e2)

    if norm_e1 == 0 or norm_e2 == 0:
        return 0  # Prevent division by zero; assume dissimilarity

    cosine_similarity = dot_product / (norm_e1 * norm_e2)

    # Normalize to [0, 1]
    return (1 + cosine_similarity) / 2


def read_valence_arousal_csv(file_path):
    # Read the CSV into a DataFrame
    df = pd.read_csv(file_path)

    # Initialize an empty list to store the data
    valence_arousal_data = []

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Construct a dictionary for each row with the necessary information
        data = {
            'valence_arousal': (row['Valence'], row['Arousal']),  # Store valence and arousal as a tuple
            'affect_label': row['Affect Label'],  # Assuming the column is named 'Affect Label'
            'affect_label_index': row['Affect Label index']  # Assuming the column is named 'Affect Label Index'
        }
        # Append the dictionary to the list
        valence_arousal_data.append(data)

    return valence_arousal_data


def generate_affect_similarity_matrix(data):
    """
    Generate the affect similarity matrix for valence-arousal pairs.

    :param data: List of dictionaries containing valence-arousal pairs and affect labels
    :return: A similarity matrix (2D numpy array)
    """
    num_pairs = len(data)
    similarity_matrix = np.zeros((num_pairs, num_pairs))  # Initialize an empty similarity matrix

    # Extract valence-arousal pairs from the data
    valence_arousal_pairs = [entry['valence_arousal'] for entry in data]

    # Compute similarity for each pair of valence-arousal vectors
    for i in range(num_pairs):
        for j in range(num_pairs):
            e1 = valence_arousal_pairs[i]
            e2 = valence_arousal_pairs[j]
            similarity_matrix[i][j] = normalized_cosine_similarity(e1, e2)

    # Normalize each row
    row_sums = np.sum(similarity_matrix, axis=1, keepdims=True)  # Sum of each row
    normalized_matrix = similarity_matrix / row_sums  # Divide each element by the row sum

    # Handle potential division by zero in case a row sum is zero
    normalized_matrix[np.isnan(normalized_matrix)] = 0  # Replace NaN (from 0/0) with 0

    return normalized_matrix

def affect_sim_mat():
    file_path = 'emotion labels fuzzy relationship matrix-20250129.csv'  # Path to your CSV file
    valence_arousal_data = read_valence_arousal_csv(file_path)
    # Generate the similarity matrix
    similarity_matrix = generate_affect_similarity_matrix(valence_arousal_data)
    return similarity_matrix


