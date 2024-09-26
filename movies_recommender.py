import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import ast
import nltk
from nltk.corpus import stopwords
from pyvis.network import Network
import streamlit.components.v1 as components

# Download required NLTK data
nltk.download('stopwords', quiet=True)

@st.cache_data
def load_and_preprocess_data(movies_file, credits_file):
    movies_df = pd.read_csv(movies_file)
    credits_df = pd.read_csv(credits_file)

    # Merge datasets on 'id' (movies_df) and 'movie_id' (credits_df)
    df = movies_df.merge(credits_df, left_on='id', right_on='movie_id')
    
    # Use 'title_x' from the movies dataset
    df['title'] = df['title_x'].fillna('Unknown Title')

    def parse_features(x, key, max_items=None):
        try:
            elements = ast.literal_eval(x)
            if isinstance(elements, list):
                names = [element[key] for element in elements if isinstance(element, dict) and key in element]
                return ' '.join(names[:max_items]) if max_items else ' '.join(names)
        except (ValueError, SyntaxError):
            return ''
        return ''
    
    # Extract relevant features with safe parsing
    df['genres'] = df['genres'].apply(lambda x: parse_features(x, 'name'))
    df['cast'] = df['cast'].apply(lambda x: parse_features(x, 'name', max_items=3))
    df['crew'] = df['crew'].apply(lambda x: parse_features(x, 'name', max_items=None))
    
    # Combine features with weighted importance
    df['combined_features'] = (
        df['genres'] + ' ' + 
        df['cast'] + ' ' +
        df['crew'] + ' ' +
        (df['overview'].fillna('') * 3) + ' ' +
        (df['keywords'].fillna('') * 1)
    )
    
    return df

@st.cache_resource
def create_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)
    return tfidf.fit_transform(df['combined_features'])

@st.cache_resource
def build_graph(df, _tfidf_matrix, n_neighbors=10):
    G = nx.Graph()

    # Add nodes with movie ID and title
    for idx, row in df.iterrows():
        if 'title' in row and pd.notna(row['title']):
            G.add_node(row['id'], title=row['title'])
    
    # Use NearestNeighbors for optimized similarity search
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute').fit(_tfidf_matrix)
    
    # Get neighbors (top n_neighbors most similar movies)
    distances, indices = nbrs.kneighbors(_tfidf_matrix)
    
    # Add edges based on nearest neighbors
    for i in range(len(df)):
        for j in range(1, n_neighbors):  # start from 1 to avoid self-similarity
            if distances[i][j] < 1:  # similarity < 1 means they are not identical
                G.add_edge(df.iloc[i]['id'], df.iloc[indices[i][j]]['id'], weight=1 - distances[i][j])
    
    return G

def collaborative_filtering(G, movie_id, depth=2):
    similar_movies = set()
    visited = set()
    queue = [(movie_id, 0)]
    
    while queue:
        current_id, current_depth = queue.pop(0)
        
        if current_depth > depth:
            continue
        
        if current_id not in visited:
            visited.add(current_id)
            similar_movies.add(current_id)
            
            for neighbor in G.neighbors(current_id):
                if neighbor not in visited and current_depth + 1 <= depth:
                    queue.append((neighbor, current_depth + 1))
    
    return similar_movies

def get_recommendations(G, df, movie_title, top_n=10):
    try:
        movie_id = df[df['title'].str.lower() == movie_title.lower()]['id'].values[0]
        similar_movies = collaborative_filtering(G, movie_id)
        
        recommendations = []
        for movie in similar_movies:
            if movie != movie_id and G.has_node(movie):
                title = G.nodes[movie]['title']
                # Check if the edge exists to avoid KeyError
                if G.has_edge(movie_id, movie):
                    similarity = G[movie_id][movie]['weight']
                else:
                    similarity = 0  # Assign a default similarity if no edge exists
                recommendations.append((title, similarity))
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    
    except IndexError:
        return "Movie not found. Please check the title and try again."
    except Exception as e:
        return f"An error occurred: {e}"

def visualize_graph(G, movie_title, df, recommendations, max_nodes=250):
    # Filter the graph to include only a subset of movies
    filtered_nodes = list(G.nodes())
    if len(filtered_nodes) > max_nodes:
        filtered_nodes = np.random.choice(filtered_nodes, size=max_nodes, replace=False)

    # Create a subgraph with filtered nodes
    subgraph = G.subgraph(filtered_nodes).copy()

    # Add the selected movie node explicitly
    selected_movie_id = df[df['title'].str.lower() == movie_title.lower()]['id'].values[0]
    if selected_movie_id not in subgraph.nodes:
        subgraph.add_node(selected_movie_id, title=movie_title)

    # Add the recommended movies to ensure they're visible in the subgraph
    recommendation_ids = []
    rec_dict = {}
    for rec in recommendations:
        movie_id = df[df['title'] == rec[0]]['id'].values[0]
        recommendation_ids.append(movie_id)
        rec_dict[movie_id] = rec[1]  # Store similarity score
        if movie_id not in subgraph.nodes:
            subgraph.add_node(movie_id, title=rec[0])

    # Find secondary similar movies (but not direct recommendations)
    secondary_similar_ids = set(subgraph.nodes()) - set(recommendation_ids) - {selected_movie_id}

    # Initialize a Pyvis Network
    net = Network(height='600px', width='100%', notebook=True)

    # Add nodes and edges to the Pyvis network
    for node in subgraph.nodes:
        node_id = str(node)
        title = subgraph.nodes[node]['title']

        if node == selected_movie_id:
            # Highlight the selected movie node with a different color and label
            label = f"{title}\n(Selected)"
            net.add_node(node_id, label=label, title=label, color='red')
        elif node in recommendation_ids:
            # Highlight recommended movies with similarity score in the label
            similarity = rec_dict.get(node, 0)
            label = f"{title}\n(Similarity: {similarity:.2f})"
            net.add_node(node_id, label=label, title=label, color='green')
        elif node in secondary_similar_ids:
            # Highlight secondary similar movies with a different color
            label = f"{title}\n(Secondary Similar)"
            net.add_node(node_id, label=label, title=label, color='orange')
        else:
            # Use default color for other nodes
            label = title
            net.add_node(node_id, label=label, title=label)

    # Ensure edges between the selected movie and its recommendations
    for rec_id in recommendation_ids:
        if G.has_edge(selected_movie_id, rec_id):
            net.add_edge(str(selected_movie_id), str(rec_id))

    # Also add the edges of the recommendations within the subgraph
    for edge in subgraph.edges:
        net.add_edge(str(edge[0]), str(edge[1]))

    # Save and display the graph
    net.show('network.html')
    HtmlFile = open('network.html', 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    st.components.v1.html(source_code, height=600, width=700)

# Modify your main function to include visualization
def main():
    st.title("Movie Recommender System with Graph Visualization")

    # Load data
    with st.spinner("Loading and preprocessing data..."):
        df = load_and_preprocess_data('tmdb_5000_movies.csv', 'tmdb_5000_credits.csv')

    # Create TF-IDF matrix
    with st.spinner("Creating TF-IDF matrix..."):
        tfidf_matrix = create_tfidf_matrix(df)

    # Build graph
    with st.spinner("Building graph with optimized Nearest Neighbors..."):
        G = build_graph(df, tfidf_matrix)

    st.success("Ready to make recommendations and visualize the graph!")

    # Create a list of movie titles for the dropdown
    movie_titles = df['title'].dropna().unique().tolist()

    # Dropdown menu with filtering
    movie_title = st.selectbox("Select or type a movie title:", movie_titles)

    if st.button("Get Recommendations"):
        recommendations = get_recommendations(G, df, movie_title)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.write("Recommendations:")
            for title, sim in recommendations:
                st.write(f"{title} (Similarity: {sim:.2f})")
        
        # Visualize the graph and highlight recommendations
        visualize_graph(G, movie_title, df, recommendations)


if __name__ == "__main__":
    main()
