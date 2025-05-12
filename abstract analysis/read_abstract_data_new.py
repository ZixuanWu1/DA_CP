import numpy as np
import pandas as pd
import gc
import os
import json
import nltk
nltk.download('punkt')  # Download the sentence tokenizer
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

import json
f= open("/abstract_data/filtered_metadata.json")
filtered_data = json.load(f)

import pandas as pd

# Create initial DataFrame
df = pd.DataFrame(filtered_data)
df["doc_id"] = df["id"]

# Step 1: Extract (doc_id, stat_category) pairs
rows = []
for entry in filtered_data:
    doc_id = entry['id']
    abstract = entry['abstract']
    categories = [cat for cat in entry['categories'].split() if cat.startswith("stat.")]
    if len(categories) == 1:  # Keep only if there's exactly one stat category
        rows.append({
            "doc_id": doc_id,
            "abstract": abstract,
            "stat_category": categories[0]
        })

# Step 2: Create DataFrame with unique (abstract, label) pairs
df_unique = pd.DataFrame(rows).drop_duplicates(subset=["doc_id", "stat_category"])

# Step 3: Output lists
abstracts = df_unique["abstract"].tolist()
labels = df_unique["stat_category"].tolist()

filtered_articles = []
filtered_topics = []

for article, topic in zip(abstracts, labels):
    sentences = sent_tokenize(article)  # Split into sentences
    if len(article) > 200 and article[:5].isascii() and len(sentences) >= 6 and topic != "stat.OT":
        filtered_articles.append(article)
        filtered_topics.append(topic)

np.random.seed(123)
indices = np.random.permutation(len(filtered_articles))

articles_selected = [filtered_articles[indices[i]] for i in range(0, 9000)]
articles_additional= [filtered_articles[indices[i]] for i in range(9000, len(filtered_articles))]


topics_selected = [filtered_topics[indices[i]] for i in range(0, 9000)]
topics_additional= [filtered_topics[indices[i]] for i in range(9000, len(filtered_articles))]

articles = articles_selected
input_texts = []
response_texts = []
for article in articles:
    sentences = sent_tokenize(article)  # Split into sentences
    input_texts.append([sentences[i] for i in [0, 2, 4]])
    response_texts.append([sentences[i] for i in [1, 3, 5]])

len(full_texts)

import pickle

with open("abstract_data/abstract_input_10000.pickle", "wb") as file:
    pickle.dump(input_texts, file)

with open("abstract_data/abstract_response_10000.pickle", "wb") as file:
    pickle.dump(response_texts, file)

import pickle

with open("abstract_data/data_abstract_10000.pickle", "wb") as file:
    pickle.dump( [articles_selected, topics_selected, articles_additional, topics_additional], file)