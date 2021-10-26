import re
import emoji
import string
import numpy as np
import pandas as pd
import networkx as nx
import sklearn.cluster as cluster
from numpy import linalg as LA
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.cluster.util import cosine_distance
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
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
                                                         if LA.norm(x, 2) !=0 else np.array(x), enc_clust)))
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


def tokenize_reviews(reviews):
    sentences = []
    for s in reviews:
        sentences.append(sent_tokenize(s))
    
    sentences = [y for x in sentences for y in x] # Flatten list
    return sentences


def process_glove_vect(glove_vect):
    word_embeddings = {}
    for line in glove_vect:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
        
    return word_embeddings


def make_lowercase(text_list):
    result = [s.lower() for s in text_list]
    return result


def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    
    return sen_new


def create_sentence_vectors(sentences, word_embeddings):
    sentence_vectors = []
    for i in sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
        
    return sentence_vectors


def create_similarity_matrix(sentences, sentence_vectors):
    sim_mat = np.zeros([len(sentences), len(sentences)])
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,768), 
                                                  sentence_vectors[j].reshape(1,768))[0,0]
    
    return sim_mat


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
                                             if LA.norm(x, 2) !=0 else np.array(x), sentence_vectors)))
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


def get_top_words(model, feature_names, top_n_words):
    top_words = []
    for topic_idx, topic in enumerate(model.components_):
        message = 'Topic #%d: ' % topic_idx
        message += ', '.join([feature_names[i]
                             for i in topic.argsort()[:-top_n_words-1:-1]])
        top_words.append(message)

    return top_words


def topic_extraction_nmf(df, rating, num_topics=3, top_n_words=5):
    stop_words = stopwords.words('english')
    
    # Vectorize bigrams and trigrams
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(2,3))

    # NMF
    nmf = NMF(n_components=num_topics)
    pipe = make_pipeline(tfidf_vectorizer, nmf)
    pipe.fit(df[df['Ratings']==rating]['Reviews'])
    
    # Print top words representing each topic
    top_words = get_top_words(nmf, tfidf_vectorizer.get_feature_names(), 
                              top_n_words=top_n_words)
    return top_words


def topic_extraction_lda(df, rating, num_topics=3, top_n_words=5):
    stop_words = stopwords.words('english')
    
    # Vectorize bigrams and trigrams
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(2,3))

    # LDA
    lda = LatentDirichletAllocation(n_components=num_topics)
    pipe = make_pipeline(tfidf_vectorizer, lda)
    pipe.fit(df[df['Ratings']==rating]['Reviews'])
    
    # Print top words representing each topic
    top_words = get_top_words(lda, tfidf_vectorizer.get_feature_names(), 
                              top_n_words=top_n_words)
    return top_words


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1+sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # Build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # Build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: # Ignore if both are the same sentence
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], 
                                                                sentences[idx2], 
                                                                stop_words)

    return similarity_matrix


def summarize_text(sentences, top_n=5):
    stop_words = stopwords.words('english')
    summary_text = []
    
    # Step 1 - Generate similary matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    
    # Step 2 - Rank sentences in similarity matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank_numpy(sentence_similarity_graph)
    
    # Step 3 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), 
                             reverse=True)
    
    for i in range(top_n):
        summary_text.append(''.join(ranked_sentence[i][1]))

    return summary_text


def clean_text(text):
    # Remove punctuations, numbers, and special characters
    text = re.sub(r"[^a-zA-Z]", ' ', text)
    return text


def remove_punctuation(text):
    for p in string.punctuation:
        text = text.replace(p, '')
    return text


def decontract_text(text):
    # Replace contractions with the full word  
    text = re.sub(r"’", "'", text)
    text = re.sub(r"won\'t", 'will not', text)
    text = re.sub(r"can\'t", 'can not', text)
    text = re.sub(r"it\'s", 'it is', text)
    text = re.sub(r"don\'t", 'do not', text)
    text = re.sub(r"'t", ' not', text)
    text = re.sub(r"'re", ' are', text)
    text = re.sub(r"'s", ' is', text)
    text = re.sub(r"'d", ' would', text)
    text = re.sub(r"'ll", ' will', text)
    text = re.sub(r"'t", ' not', text)
    text = re.sub(r"'ve", ' have', text)
    text = re.sub(r"'m", ' am', text)
    text = re.sub(r"'", '', text)
    
    return text


def remove_emoji(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        u"\U00002500-\U00002BEF"  # Chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"                 # Dingbats
        u"\u3030""]+", re.UNICODE)
    
    return emoj.sub(r'', text)


def pos_tagging(text):
    text = word_tokenize(text)
    tag_list = []
    for i in text:
        tag_list.append(pos_tag([i]))
    
    return tag_list


def wordnet_tags(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    if tag.startswith('V'):
        return wordnet.VERB
    if tag.startswith('N'):
        return wordnet.NOUN
    if tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def change_tag(text):
    # Map tags from pos_tag to WordNet pos categories
    new = []
    for i in text:
        new.append([i[0][0], wordnet_tags(i[0][1])])
    
    return new


def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    new = []
    for i in text:
        new.append(lemmatizer.lemmatize(i[0]))
    
    new = ' '.join(new)
    return new