import pandas as pd
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = pickle.load(open('model.pkl','rb'))

df1 = pickle.load(open('courses.pkl','rb'))
df2 = pickle.load(open('coursera.pkl','rb'))
df = pickle.load(open('youtube_courses.pkl','rb'))

vectors3 = pickle.load(open('youtube_vector.pkl','rb'))
vectors = pickle.load(open('vectors.pkl', 'rb'))
vectors2 = pickle.load(open('vectors2.pkl', 'rb'))


def faiss_load(vectors1):
    d = vectors1.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors1)
    return index


index = faiss_load(vectors)
index2 = faiss_load(vectors2)
index3 = faiss_load(vectors3)

def faiss_recommend(word, index, df, top=5):
    embed = model.encode([word]).astype('float32')
    distances, indexes = index.search(embed, top)

    return df.iloc[indexes[0]][['title', 'avg_rating', 'course_url', 'platform']]

def faiss_recommend1(word,index,df,top=10):
    embed = model.encode([word]).astype('float32')
    distances, indexes = index.search(embed, top + 1)

    input_index = df[df['title'] == word].index

    filtered_indexes = [idx for idx in indexes[0] if idx not in input_index]

    filtered_indexes = filtered_indexes[:top]

    return df.iloc[filtered_indexes][['title','url','platform']]


def get_recommendations(query):
    if not query:
        return None, None

    udemy_results = faiss_recommend(query, index, df1, top=10)
    udemy_results = udemy_results.sort_values(by="avg_rating", ascending=False)
    coursera_results = faiss_recommend(query, index2, df2, top=5)
    coursera_results = coursera_results.sort_values(by="avg_rating", ascending=False)
    youtube_results = faiss_recommend1(query, index3,df,top = 10)
    return udemy_results, coursera_results, youtube_results
