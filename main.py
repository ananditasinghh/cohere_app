import cohere
from dotenv import load_dotenv
import os
from annoy import AnnoyIndex
import numpy as np
import streamlit as st

load_dotenv()

co = cohere.Client(os.getenv('COHERE_API_KEY'))

st.subheader('Book list Generator and Ranker:')

# Topic input
topic = st.text_input("Topic", placeholder="Topic",
                      label_visibility="collapsed")

if st.button("Generate"):
    try:
        with st.spinner("Please wait..."):
            # Load the Cohere Generate module
            response = co.generate(prompt="Write a list of the 5 most popular books about " +
                                   topic+" with a short description for each one.")

            # Write the results in a text file for future use
            with open('output.txt', 'w') as f:
                f.write(response.generations[0].text)

            # Show the results on the front end
            st.success(response.generations[0].text)
    except Exception as e:
        st.exception(f"Exception: {e}")

###########################################################################################################################

if os.path.isfile('output.txt'):
    st.subheader('Semantic search:')
    query = st.text_input("Query", placeholder="Query", label_visibility="collapsed")
    
    if st.button("Query"):
        with st.spinner("Please wait..."):

            # Load the list generated in the last step and save it in an array after removing irrelevant lines
            with open('output.txt') as f:
                lines = list(filter(None, f.read().splitlines()))[1:]

            # Load the Cohere Embed module
            embeds = co.embed(
                texts=lines,
                model='small',
            ).embeddings

            # Create the search index, pass the size of embedding
            search_index = AnnoyIndex(np.array(embeds).shape[1], 'angular')

            # Add all the vectors to the search index
            for i in range(len(embeds)):
                search_index.add_item(i, embeds[i])
            search_index.build(10)  # 10 trees
            search_index.save('embeds.ann')

            print(embeds)

            # Embed the query using the same 'small' model
            query_embed = co.embed(texts=[query], model='small').embeddings

            # Retrieve the nearest neighbor
            similar_item_id = search_index.get_nns_by_vector(query_embed[0], 1, include_distances=False)

            # Show the results on the front end
            st.success(lines[similar_item_id[0]])
            
####################################################################################################################
#
#if os.path.isfile('output.txt'):
#    st.subheader('Semantic similarity ranking:')
#    query = st.text_input("Rerank", placeholder="Query", label_visibility="collapsed")

 #   if st.button("Rerank"):
#        with st.spinner("Please wait..."):
#            try:
                # Load the list generated in last step and save it in an array after removing irrelevant lines
 #               with open('output.txt') as f:
  #                  lines = list(filter(None, f.read().splitlines()))[1:]

 #               # Load the Cohere Rerank module
#                results = co.rerank(query=query, documents=lines, top_n=5, model='rerank-english-v2.0')

                # Debug: Print out the structure of results
 #               st.write("Results structure:", results)

                # Parse the results
  #              results_string = ""
   #             for idx, r in enumerate(results):
                    # Debug: Print the type and value of r.index
 #                   st.write(f"Result {idx}: Type of r.index: {type(r.index)}, Value: {r.index}")

                    # Use the index attribute to fetch the corresponding document text
   #                 if isinstance(r.index, int):
   #                     document_text = lines[r.index]
    #                    results_string += f"{document_text}\n\n"
     #                   results_string += f"Relevance Score: {r.relevance_score:.2f}\n"
      #                  results_string += "\n"
       #             else:
        #                st.error(f"Unexpected type for index: {type(r.index)}")

                # Show the results on the frontend
         #       st.success(results_string)
          #  except Exception as e:
           #     st.error(f"An error occurred: {str(e)}")


