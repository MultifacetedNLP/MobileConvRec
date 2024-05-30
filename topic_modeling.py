
import sklearn
from copy import deepcopy
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import collections
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import umap
import hdbscan
from bertopic import BERTopic
import torch

torch.cuda.set_device(0)
print(torch.cuda.current_device())

#load dataset
def load_dataset():
    file_path = 'dataset/amazon/all_beauty_review.json'
    
    # Read the specified number of lines from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    
    texts =[]
    for item in data[0]:
        try:
            texts.append(item['review_text'])
        except KeyError:
            pass 
    return texts

def save_embeddings(docs):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(docs, show_progress_bar=True)

    with open('embeddings/amazon/embeddings.npy', 'wb') as f:
        np.save(f, embeddings)
    return

def load_embeddings():
    embeddings = np.load('embeddings/amazon/embeddings.npy')
    return embeddings

def prepare_vocab(docs):
    vocab_counter = collections.Counter()
    tokenizer = CountVectorizer().build_tokenizer()
    for doc in tqdm(docs):
        vocab_counter.update(tokenizer(doc))
    
    vocab = [word for word, frequency in vocab_counter.items() if frequency >= 15]; 
    return vocab

def get_aspects(docs, embeddings, vocab):
    
    # Prepare sub-models
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    umap_model = umap.UMAP(n_components=5, n_neighbors=50, random_state=42, metric="cosine", low_memory=True)
    hdbscan_model = hdbscan.HDBSCAN(min_samples=20, gen_min_span_tree=True, prediction_data=False, min_cluster_size=20)
    vectorizer_model = CountVectorizer(vocabulary=vocab, stop_words="english", ngram_range=(1,3), min_df=10)

    # Fit BERTopic without actually performing any clustering
    topic_model= BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics = "auto"
    ).fit(docs, embeddings=embeddings)

    #Fine Tune topic representations after training Bertopic
    
    topic_info = topic_model.get_topic_info()

    topic_model.save(
        path='model_dir/amazon',
        serialization="safetensors",
        save_embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )


    print("Topic Model Save:")


if __name__ == '__main__':
    sentences = load_dataset()
    save_embeddings(sentences)
    embeddings = load_embeddings()
    vocab = prepare_vocab(sentences)
    get_aspects(sentences, embeddings, vocab)