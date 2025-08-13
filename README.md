# Download data 
You will need the Traditional Chinese folder from https://talkbank.org/childes/access/Chinese/ (linked as "this complete set") in the description. 
Unzip it into the same directory as the contents of this repository and make sure it is named "Mandarin". 

# Preprocessing / getting data from the folder 
Use process_childes.py. Each function comes with a docstring. 
The file starts with a list of participant IDs that divides them into children, adults and "other", which contains any ID strings where the categorization depends on the file. For these cases, create_participant_dict(cha_path) sorts them correctly for each specific .cha file. 
To see how each preprocessing fuction is used, refer to test_processing.ipynb. 
In general, if you want to use the full corpus, use: 

data = clean_chinese_utterances(get_all_utterance_gra_by_group_folder("Mandarin"))

This returns cleaned up utterances (only Chinese characters remaining) along with the GRA tier info: 

    {
      "child": { age: [(utterance, gra), ...], ... },
      "adult": [(utterance, gra), ...]
    }

CAUTION: I am not sure yet if there are any instances where cleaning up non-Chinese characters causes problems (such as if someone uses a foreign word?) 

# Dependency Embeddings 
I tried to implement something close to what this paper does based on the dependency tier: https://aclanthology.org/P14-2050/ 
Check dependency2vec-full-clean.ipynb for results. I could not see any pattern in the resulting plots, so I abandoned the idea for now and tried contextual embeddings instead. 

# Pre-trained contextual embeddings 
Use this snippet to import the pickled embeddings into your notebook: 

import os
import pickle
from collections import defaultdict
from load_embeddings import load_char_embeddings

input_dir = "char_embeddings_by_group_all_with_utts"
char_embeddings_by_group = defaultdict(lambda: defaultdict(list))

for filename in os.listdir(input_dir):
    if filename.endswith(".pkl"):
        char, group = filename.replace(".pkl", "").split("_")
        with open(os.path.join(input_dir, filename), "rb") as f:
            vectors = pickle.load(f)
            char_embeddings_by_group[char][group] = vectors
char_embeddings_by_group = load_char_embeddings()

char_embeddings is a dictionary with the following structure: 
{
    "child": { age: character: [(embedding_vector, utterance), ...], ... },
    "adult": character: [(embedding_vector, utterance), ...]
}
where "character" refers to one of "把", "被", "給".

Check context_embedding_full.ipynb to see how the embeddings were created and the first few PCA plots. 
I used model = SentenceTransformer("distiluse-base-multilingual-cased-v2") which is a popular Huggingface model to try it out, but we can try other models later. 
