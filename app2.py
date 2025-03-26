import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
import faiss
from sentence_transformers import SentenceTransformer

st.title('Course Recommendation System')
df = pickle.load(open('courses.pkl','rb'))
model = pickle.load(open('model.pkl', 'rb'))
vectors = pickle.load(open('vectors.pkl', 'rb'))


def faiss_load():
    d = vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    return index


index = faiss_load()


def faiss_recommend(word, top=5):
    embed = model.encode([word]).astype('float32')
    distances, indexes = index.search(embed, top + 1)

    input_index = df[df['title'] == word].index

    filtered_indexes = [idx for idx in indexes[0] if idx not in input_index]

    filtered_indexes = filtered_indexes[:top]

    return df.iloc[filtered_indexes][['title', 'avg_rating', 'course_url']]

selected = st.selectbox(
    "Select course ",
    (df['title'].values),
)


if st.button("Recommend"):
    recommended_indexes = faiss_recommend(selected)
    st.write("### Recommended Courses:")

    for i, row in recommended_indexes.iterrows():
        url = str(row['course_url']).strip("[]'")
        st.write(f"**{row['title']}** (Rating: {row['avg_rating']})")
        st.markdown(f"[Course Link](https://www.udemy.com{url})", unsafe_allow_html=True)