import streamlit as st
import pandas as pd
from Final import get_recommendations
st.title('Course Recommendation System')

query = st.text_input("Enter your search query:")

if st.button("Recommend"):
    if query:
        udemy_courses, coursera_courses = get_recommendations(query)

        st.markdown("<h2 style='font-size:24px;'>Top Udemy Courses</h2>", unsafe_allow_html=True)
        st.markdown("<style>table {font-size: 18px;}</style>", unsafe_allow_html=True)
        st.table(udemy_courses)

        st.markdown("<h2 style='font-size:28px;'>Top Coursera Courses</h2>", unsafe_allow_html=True)
        st.markdown("<style>table {font-size: 16px;}</style>", unsafe_allow_html=True)
        st.table(coursera_courses)
    else:
        st.warning("Please enter a search query before clicking Recommend.")