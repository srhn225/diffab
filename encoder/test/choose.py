import os
import argparse
import torch
import torch.nn.functional as F

def load_tensors_from_directory(directory_path):
    """Load all .pt files (tensors) from the given directory."""
    tensors = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.pt'):
            tensor = torch.load(file_path)
            tensors[filename] = tensor
    return tensors

def compute_cosine_similarity(data, target_tensor):
    """Compute cosine similarity between target_tensor and all tensors in data."""
    similarities = {}
    for filename, interface in data.items():
        antigen_feature = interface['antigen_feature']
        similarities[filename] = F.cosine_similarity(target_tensor, antigen_feature[0], dim=0).sum()
    return similarities

def find_top_similar(similarity_dict, top_n=10, id=None):
    """Find the top N files with the highest cosine similarity, excluding entries with the given id."""
    
    # If an id is provided, filter out entries where id appears in the key (file name)
    if id is not None:
        filtered_dict = {k: v for k, v in similarity_dict.items() if id not in k}
    else:
        filtered_dict = similarity_dict


    # Sort the remaining similarity dictionary by values (similarity scores) in descending order
    sorted_similarities = sorted(filtered_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Return the top N results
    return sorted_similarities[:top_n]


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Find top similar tensors based on cosine similarity.')
    parser.add_argument('--directory', type=str, help='Path to the directory containing .pt files')
    args = parser.parse_args()

    # Load tensors from the directory
    data = load_tensors_from_directory(args.directory)

    # Define a target tensor (For example purposes, defined here)
    target_tensor = data["['1fj1_B_A_F']_features.pt"]['light_feature'][0]

    # Compute cosine similarity
    similarities = compute_cosine_similarity(data, target_tensor)

    # Find the top 10 most similar files
    top_similar_files = find_top_similar(similarities, top_n=20)
    
    # Print the results
    print("Top 10 most similar files:")
    for filename, score in top_similar_files:
        print(f"File: {filename}, Similarity score: {score}")
