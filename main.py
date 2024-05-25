import cohere
from dotenv import load_dotenv
import os
from annoy import AnnoyIndex
import numpy as np
import streamlit as st

# Loading environment variables
load_dotenv()

# Initialising Cohere client
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

st.markdown("""
    <style>
        .main { background-color: #D7BDE2; }
        .stButton>button { background-color: #AF7AC5; color: white; border-radius: 5px; }
        .stTextInput>div>div>input { border: 2px solid #ccc; border-radius: 5px; padding: 10px; background-color:#9B59B6 }
        .stMarkdown { color: #333; }
        .stTextInput label { font-size: 40px; } /* Adjust font size of the text input label */
        
        st.markdown("<h1 style='text-align: center; font-size: 36px;'>Book List Generator and Semantics</h1>", unsafe_allow_html=True)


        footer {
            position: absolute;
            left: 0;
            bottom: 0;
            width: 100%;
            color: bule;
            background-color: #D7BDE2;
            color: white;
            text-align: center;
            padding: 7px;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.title('Book List Generator and Semantics')

# Section: Generate Book List
st.header('Generate Book List')
st.write("Enter a topic to generate a list of the 5 most popular books with descriptions.")

# Topic input
topic = st.text_input("Your Topic :", placeholder="e.g., Artificial Intelligence")

if st.button("Generate"):
    if not topic.strip():
        st.error("Please enter a topic.")
    else:
        with st.spinner("Generating book list..."):
            try:
                book_list = generate_book_list(topic)
                st.success("Book list generated successfully!")
                st.markdown(book_list.replace("\n", "\n\n"))
            except Exception as e:
                st.exception(f"Exception: {e}")

# Section: Semantic Search
if os.path.isfile('output.txt'):
    st.header('Semantic Search')
    st.write("Enter a query to search within the generated book list.")

    query = st.text_input("Your Query : ", placeholder="e.g., machine learning")

    if st.button("Query"):
        if not query.strip():
            st.error("Please enter a query.")
        else:
            with st.spinner("Searching..."):
                try:
                    lines = load_book_list()
                    search_index = create_search_index(lines)
                    similar_item_id = search_books(query, search_index)
                    st.success("Search result:")
                    st.write(lines[similar_item_id[0]])
                except Exception as e:
                    st.exception(f"Exception: {e}")

#copyrights
st.markdown('<br> <br> <div style="text-align:center;font-size:15px; color: black;"> ©AnanditaSingh | 2024 </div>', unsafe_allow_html=True)
