import cohere
from dotenv import load_dotenv
import os
from annoy import AnnoyIndex
import numpy as np
import streamlit as st

load_dotenv()

co = cohere.Client(os.getenv('COHERE_API_KEY'))

def generate_book_list(topic):
    prompt = f"Write a list of the 5 most popular books about {topic} with a short description for each one."
    response = co.generate(prompt=prompt)
    with open('output.txt', 'w') as f:
        f.write(response.generations[0].text)
    return response.generations[0].text

def load_book_list():
    with open('output.txt') as f:
        lines = list(filter(None, f.read().splitlines()))[1:]
    return lines

def create_search_index(lines, model='small'):
    embeds = co.embed(texts=lines, model=model).embeddings
    search_index = AnnoyIndex(np.array(embeds).shape[1], 'angular')
    for i, embed in enumerate(embeds):
        search_index.add_item(i, embed)
    search_index.build(10)
    search_index.save('embeds.ann')
    return search_index

def search_books(query, search_index, model='small'):
    query_embed = co.embed(texts=[query], model=model).embeddings
    similar_item_id = search_index.get_nns_by_vector(query_embed[0], 1, include_distances=False)
    return similar_item_id

st.subheader('Book List Generator and Ranker:')
topic = st.text_input("Topic", placeholder="Topic", label_visibility="collapsed")
if st.button("Generate"):
    with st.spinner("Please wait..."):
        try:
            book_list = generate_book_list(topic)
            st.success(book_list)
        except Exception as e:
            st.exception(f"Exception: {e}")

if os.path.isfile('output.txt'):
    st.subheader('Semantic Search:')
    query = st.text_input("Query", placeholder="Query", label_visibility="collapsed")
    if st.button("Query"):
        with st.spinner("Please wait..."):
            try:
                lines = load_book_list()
                search_index = create_search_index(lines)
                similar_item_id = search_books(query, search_index)
                st.success(lines[similar_item_id[0]])
            except Exception as e:
                st.exception(f"Exception: {e}")
