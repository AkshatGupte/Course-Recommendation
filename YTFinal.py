import pandas as pd
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = pickle.load(open('model.pkl','rb'))

df = pickle.load(open('youtube_courses.pkl','rb'))

vectors = pickle.load(open('youtube_vector.pkl','rb'))


def faiss_load(vectors1):
    d = vectors1.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors1)
    return index


index = faiss_load(vectors)


def faiss_recommend(word, index, df, top=5):
    embed = model.encode([word]).astype('float32')
    distances, indexes = index.search(embed, top)

    return df.iloc[indexes[0]][['title', 'avg_rating', 'course_url', 'platform']]


def get_recommendations(query):
    if not query:
        return None, None

    udemy_results = faiss_recommend(query, index, df, top=10)
    udemy_results = udemy_results.sort_values(by="avg_rating", ascending=False)
    coursera_results = faiss_recommend(query, index2, df2, top=5)
    coursera_results = coursera_results.sort_values(by="avg_rating", ascending=False)
    return udemy_results, coursera_results

