# in load_embeddings.py
import os, pickle
from collections import defaultdict

def load_char_embeddings(input_dir="char_embeddings_by_group_all"):
    char_embeddings_by_group = defaultdict(lambda: defaultdict(list))
    for filename in os.listdir(input_dir):
        if filename.endswith(".pkl"):
            char, group = filename.replace(".pkl", "").split("_")
            with open(os.path.join(input_dir, filename), "rb") as f:
                vectors = pickle.load(f)
                char_embeddings_by_group[char][group] = vectors
    return char_embeddings_by_group
