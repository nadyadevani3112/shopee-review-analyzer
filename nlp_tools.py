import re
import emoji
import string
import numpy as np
import pandas as pd
import sklearn.cluster as cluster
from numpy import linalg as LA
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer


def get_key_phrases(reviews, num_clusters=5):
    reviews['reviews'] = reviews['Reviews'].apply(lambda x: re.sub(emoji.get_emoji_regexp(), '.', x)
                                                              .replace(';', '.')
                                                              .replace('!', '.')
                                                              .replace(',', '.')
                                                              .replace('?', '.')
                                                              .replace('\n', '.')
                                                              .split('.'))
    
    clauses = reviews.explode('reviews')
    clauses['reviews'] = clauses['reviews'].apply(lambda x: x.strip(string.punctuation + ' ’'))
    
    # Remove clauses with length 1
    clauses = clauses[clauses['reviews'].apply(lambda x: len(list(x.split())) > 2)]
    
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    ratings_summary = {}
    for rating in range(1, 6):
        print('Finding key phrases for reviews with rating ' + str(rating) + '...')
        sentences = clauses[clauses['Ratings'] == rating][['reviews']]
        
        if sentences.empty:
            ratings_summary[rating] = ['***There are no reviews with this rating***']
        
        else:
            sentences['reviews'] = sentences['reviews'].apply(lambda x: strip_conj(x)) # Remove conjugations
            sentences['reviews'] = sentences['reviews'].apply(lambda x: x.strip())
            sentences = sentences[sentences['reviews'].apply(lambda x: x.count(' ') >= 1)]
            sentences_no_dup = sentences.drop_duplicates()

            if sentences_no_dup.shape[0] <= 5:
                ratings_summary[rating] = list(sentences_no_dup['reviews'].values)
                continue
            
            sentences_no_dup = sentences_no_dup.copy()
            sentences_no_dup['encoding'] = sentences_no_dup['reviews'].apply(lambda x: model.encode(x))
            sentences = pd.merge(sentences, sentences_no_dup, on='reviews')
            enc = np.array(sentences['encoding'].to_list())
            reviews_sent = list(sentences['reviews'].values)

            num_clusters = num_clusters
            kmeans_labels = cluster.KMeans(n_clusters=num_clusters, random_state=2021).fit_predict(enc)
            clusters = {}
            for i in range(num_clusters):
                clusters[i] = np.where(kmeans_labels == i)
            
            cluster_summary = {}
            for k in range(num_clusters):
                clust = sentences.reset_index(drop=True).reset_index()[sentences.reset_index(drop=True)
                                                                                .reset_index()['index']
                                                                                .isin(clusters[k][0])]
                enc_clust = np.array(clust['encoding'].to_list())
                enc_clust_normalized = np.array(list(map(lambda x: np.array(x) / LA.norm(x, 2) 
                                                         if LA.norm(x, 2) != 0 else np.array(x), enc_clust)))
                similarities_clust = enc_clust_normalized.dot(enc_clust_normalized.T)
                sim_total_scores = np.sum(similarities_clust, axis=1)
                scores_clust = dict(list(enumerate(sim_total_scores)))
                reviews_sent_clust = list(clust['reviews'].values)
                sentences_clust = [(i,scores_clust[i], s) for i, s in enumerate(reviews_sent_clust)]
                ranked_sentences_clust = sorted(((i,scores_clust[i], s) for i, s in enumerate(reviews_sent_clust)), 
                                                reverse=True, key=lambda x: x[1])
                cluster_summary[k] = list(map(lambda x: x[2], ranked_sentences_clust[:5])) # Top 5
                shortest_length = min(list(map(lambda x: len(x), cluster_summary[k])))
                cluster_summary[k] = list(filter(lambda x: len(x) == shortest_length, cluster_summary[k]))[0]
        
            ratings_summary[rating] = list(cluster_summary.values())
    
    print('Done!')
    return ratings_summary


def strip_conj(x):
    # Remove conjugation(s) at the start and end of text
    conj = ['for', 'and', 'nor', 'but', 'or', 'yet', 'so', 'accordingly', 'furthermore', 
            'moreover', 'similarly', 'also', 'hence', 'namely', 'still', 'anyway', 'however', 
            'nevertheless', 'then', 'besides', 'incidentally', 'next', 'thereafter', 'certainly', 
            'indeed', 'nonetheless', 'therefore', 'consequently', 'instead', 'now', 'thus', 
            'finally', 'likewise', 'otherwise', 'undoubtedly', 'further', 'meanwhile']
    
    new_x = x
    for c in conj:
        if x.lower().startswith(c + ' '):
            l = len(c) + 1
            new_x = x[l:]
            new_x = new_x.strip()
    while new_x != x:
        x = new_x
        for c in conj:
            if x.lower().startswith(c + ' '):
                l = len(c) + 1
                new_x = x[l:]
                new_x = new_x.strip()
    for c in conj:
        if x.endswith(' ' + c):
            l = len(c) + 1
            new_x = x[:-l]
            new_x = new_x.strip()
    while new_x != x:
        x = new_x
        for c in conj:
            if x.endswith(' ' + c):
                l = len(c) + 1
                new_x = x[:-l]
                new_x = new_x.strip()
    return x


def make_lowercase(text_list):
    result = [s.lower() for s in text_list]
    return result


def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


def clean_text(text):
    # Remove punctuations, numbers, and special characters
    text = re.sub(r"[^a-zA-Z]", ' ', text)
    return text


def tokenize_reviews(reviews):
    sentences = []
    for s in reviews:
        sentences.append(sent_tokenize(s))
    
    sentences = [y for x in sentences for y in x] # Flatten list
    return sentences


def get_top_reviews(sentences, top_n=10):
    print('Getting top reviews...')
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    
    # Remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).apply(lambda x: clean_text(x))

    # Make alphabets lowercase
    clean_sentences = make_lowercase(clean_sentences)

    # Remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    
    # Vectorize sentences based on word embeddings
    sentence_vectors = model.encode(clean_sentences)
    
    # Create similarity matrix
    enc_clust_normalized = np.array(list(map(lambda x: np.array(x) / LA.norm(x, 2) 
                                             if LA.norm(x, 2) != 0 else np.array(x), sentence_vectors)))
    similarities_clust = enc_clust_normalized.dot(enc_clust_normalized.T)
    sim_total_scores = np.sum(similarities_clust, axis=1)
    scores = dict(list(enumerate(sim_total_scores)))
    
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), 
                              reverse=True)
    
    top_reviews = []
    for i in range(top_n):
        top_reviews.append(''.join(ranked_sentences[i][1]))
    
    print('Done!')
    return top_reviews


def get_ratings_proportion(reviews):
    counts = reviews['Ratings'].value_counts()
    total = len(reviews)
    
    results = {}
    for i in range(1, 6):
        try:
            count = counts[i]
            results[i] = (count, (count/total)*100)
        except:
            results[i] = (0, 0)
    
    return results