
# Movie Recommender System Documentation

## Overview

This document provides a comprehensive explanation of a movie recommender system implemented in Python. The system uses a content-based approach combined with collaborative filtering for suggesting movies similar to a user's input. It also includes a network visualization of the recommendations.

## Table of Contents

1. [Dependencies](#dependencies)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Engineering](#feature-engineering)
4. [Similarity Calculation](#similarity-calculation)
5. [Recommendation Generation](#recommendation-generation)
6. [Visualization](#visualization)
7. [Main Execution Flow](#main-execution-flow)

## 1. Dependencies

The system relies on several Python libraries:

- **pandas**: For data manipulation and analysis
- **numpy**: For numerical computations
- **scikit-learn**: For machine learning utilities (TF-IDF vectorization, Nearest Neighbors)
- **nltk**: For natural language processing tasks (handling stopwords)
- **networkx**: For graph construction and collaborative filtering
- **pyvis**: For visualizing the recommendation network graph
- **streamlit**: For creating the interactive user interface

## 2. Data Preprocessing

### 2.1 Data Loading and Merging

The `load_and_preprocess_data` function handles the initial data preparation:

- Loads the data from two CSV files: one containing movie metadata and another containing cast and crew information.
- Merges the datasets on the movie ID to combine relevant features (genres, cast, crew, etc.).
- Handles missing values by filling them with appropriate defaults.

### 2.2 Feature Parsing

The system extracts useful information from complex JSON-like columns:

- **Genres**: Extracts the genre names from the nested dictionaries.
- **Cast**: Extracts the names of the top three cast members.
- **Crew**: Extracts crew member names.

These features are combined into a single `combined_features` column used for similarity calculations.

## 3. Feature Engineering

### 3.1 Combined Features

The `combined_features` column is created by concatenating genres, cast, crew, movie overview, and keywords, with weighted importance given to the overview and keywords.

### 3.2 Text Vectorization

The `create_tfidf_matrix` function creates a TF-IDF (Term Frequency-Inverse Document Frequency) matrix from the `combined_features` column, allowing for efficient similarity calculations.

## 4. Similarity Calculation

### 4.1 Nearest Neighbors

The system uses the Nearest Neighbors algorithm to find similar movies based on the TF-IDF vectors. Cosine similarity is used to measure the similarity between movies, with the top similar movies being identified for each input movie.

## 5. Recommendation Generation

The `get_recommendations` function generates movie recommendations by:

1. Finding the ID of the movie selected by the user.
2. Performing collaborative filtering using the Nearest Neighbors graph.
3. Filtering and sorting the most similar movies based on similarity scores.
4. Returning the top N movie recommendations to the user.

## 6. Visualization

### 6.1 Graph Visualization

The `visualize_graph` function visualizes the recommendation graph:

- A network graph is built using **NetworkX** and visualized with **Pyvis**.
- The input movie is highlighted in red, and the recommended movies are shown in green.
- The graph is displayed within the Streamlit interface using an embedded HTML component.

## 7. Main Execution Flow

The `main` function controls the flow of the application:

1. Loads and preprocesses the data.
2. Creates a TF-IDF matrix and builds a similarity graph using Nearest Neighbors.
3. Provides an interactive interface using **Streamlit**, where users can select a movie and receive recommendations.
4. Displays the movie recommendations and visualizes the relationship between similar movies.

## Conclusion

This movie recommender system uses a hybrid approach of content-based filtering with collaborative signals from Nearest Neighbors. The system also offers a graph-based visualization to provide an interactive view of movie recommendations, enhancing the user experience by visually mapping out similar movies.
