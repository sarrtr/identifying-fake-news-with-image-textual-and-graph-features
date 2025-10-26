import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

class FakedditGraphBuilder:
    def __init__(self, similarity_threshold=0.3, max_features=5000):
        """
        Initialize the graph builder
        
        Args:
            similarity_threshold: Minimum similarity score for edges (0-1)
            max_features: Maximum number of features for TF-IDF vectorizer
        """
        self.similarity_threshold = similarity_threshold
        self.max_features = max_features
        self.graph = nx.Graph()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
    
    def load_dataset(self, file_path, labels_file_path=None):
        """
        Load the Fakeddit TSV dataset and filter based on labels file
        
        Args:
            file_path: Path to the main dataset TSV file
            labels_file_path: Path to the needed file containing IDs to include
        """
        print(f"Loading dataset from {file_path}")
        df = pd.read_csv(file_path, sep='\t')
        
        # If labels file is provided, filter dataset to only include those IDs
        if labels_file_path:
            print(f"Loading labels from {labels_file_path}")
            labels_df = pd.read_csv(labels_file_path, sep='\t')
            target_ids = labels_df['id'].tolist()
            
            # Filter main dataset to only include IDs from labels file
            initial_count = len(df)
            df = df[df['id'].isin(target_ids)]
            final_count = len(df)
            
            print(f"Filtered dataset: {initial_count} -> {final_count} rows")
            print(f"IDs found in main dataset: {final_count}/{len(target_ids)}")
            
            # Add the initial labels to the main dataframe
            labels_dict = dict(zip(labels_df['id'], labels_df['label']))
            df['initial_label'] = df['id'].map(labels_dict)
        
        return df
    
    def preprocess_text(self, df, text_columns=['title', 'clean_title']):
        """
        Combine and preprocess text columns
        """
        # Combine available text columns
        df['combined_text'] = ''
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
                df['combined_text'] += df[col] + ' '
        
        # Remove extra spaces and basic cleaning
        df['combined_text'] = df['combined_text'].str.strip()
        df['combined_text'] = df['combined_text'].str.lower()
        df['combined_text'] = df['combined_text'].str.replace(r'[^\w\s]', '', regex=True)
        
        # Remove very short texts
        df = df[df['combined_text'].str.len() > 10]
        
        return df
    
    def compute_similarity_matrix(self, texts):
        """
        Compute cosine similarity matrix between texts using TF-IDF
        """
        print("Vectorizing texts...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        print("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return similarity_matrix, tfidf_matrix
    
    def build_graph(self, df, text_column='combined_text', id_column='id'):
        """
        Build graph from dataframe with weighted edges based on similarity
        """
        if id_column not in df.columns:
            df = df.reset_index()
            id_column = 'index'
        
        print("Building graph nodes...")
        # Add nodes with all available attributes
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Adding nodes"):
            node_id = row[id_column]
            node_attributes = {
                'title': row.get('title', ''),
                'clean_title': row.get('clean_title', ''),
                'author': row.get('author', ''),
                'domain': row.get('domain', ''),
                'subreddit': row.get('subreddit', ''),
                'num_comments': row.get('num_comments', 0),
                'score': row.get('score', 0),
                'upvote_ratio': row.get('upvote_ratio', 0),
                'hasImage': row.get('hasImage', False),
                'label_2_way': row.get('2_way_label', ''),
                'label_3_way': row.get('3_way_label', ''),
                'label_6_way': row.get('6_way_label', ''),
                'combined_text': row.get(text_column, ''),
                'initial_label': row.get('initial_label', '')
            }
            self.graph.add_node(node_id, **node_attributes)
        
        print("Computing similarities and building weighted edges...")
        # Compute similarity matrix
        similarity_matrix, _ = self.compute_similarity_matrix(df[text_column].tolist())
        
        # Add edges based on similarity threshold
        n_nodes = len(df)
        edges_added = 0
        
        for i in tqdm(range(n_nodes), desc="Building edges"):
            for j in range(i + 1, n_nodes):
                similarity = similarity_matrix[i, j]
                if similarity > self.similarity_threshold:
                    node_i = df.iloc[i][id_column]
                    node_j = df.iloc[j][id_column]
                    
                    # Add edge with similarity as weight
                    self.graph.add_edge(
                        node_i, 
                        node_j, 
                        weight=float(similarity),
                        similarity=float(similarity)
                    )
                    edges_added += 1
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {edges_added} edges")
        return self.graph
    
    def analyze_graph(self):
        """
        Analyze graph properties and statistics
        """
        if len(self.graph) == 0:
            print("Graph is empty. Build graph first.")
            return
        
        print("\n" + "="*50)
        print("GRAPH ANALYSIS")
        print("="*50)
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        print(f"Graph density: {nx.density(self.graph):.6f}")
        
        if self.graph.number_of_nodes() > 0:
            # Degree statistics
            degrees = [deg for _, deg in self.graph.degree()]
            print(f"Average degree: {np.mean(degrees):.2f}")
            print(f"Max degree: {max(degrees)}")
            print(f"Min degree: {min(degrees)}")
            
            # Weight statistics
            edge_weights = [data['weight'] for _, _, data in self.graph.edges(data=True)]
            print(f"Average edge weight: {np.mean(edge_weights):.3f}")
            print(f"Max edge weight: {max(edge_weights):.3f}")
            print(f"Min edge weight: {min(edge_weights):.3f}")
            
            # Connected components
            components = list(nx.connected_components(self.graph))
            print(f"Number of connected components: {len(components)}")
            if components:
                component_sizes = [len(comp) for comp in components]
                print(f"Largest component size: {max(component_sizes)}")
                print(f"Smallest component size: {min(component_sizes)}")
                
                # Analyze labels distribution
                self.analyze_labels_distribution()
    
    def analyze_labels_distribution(self):
        """
        Analyze the distribution of labels in the graph
        """
        labels_2_way = [data.get('label_2_way', '') for _, data in self.graph.nodes(data=True)]
        labels_3_way = [data.get('label_3_way', '') for _, data in self.graph.nodes(data=True)]
        labels_6_way = [data.get('label_6_way', '') for _, data in self.graph.nodes(data=True)]
        initial_labels = [data.get('initial_label', '') for _, data in self.graph.nodes(data=True)]
        
        print("\nLabel Distribution:")
        print(f"2-way labels: {pd.Series(labels_2_way).value_counts().to_dict()}")
        print(f"3-way labels: {pd.Series(labels_3_way).value_counts().to_dict()}")
        print(f"6-way labels: {pd.Series(labels_6_way).value_counts().to_dict()}")
        
        # Analyze initial labels distribution
        if any(initial_labels):
            print(f"initial labels stats - Min: {min([x for x in initial_labels if x != '']):.3f}, "
                  f"Max: {max([x for x in initial_labels if x != '']):.3f}, "
                  f"Mean: {np.mean([x for x in initial_labels if x != '']):.3f}")
    
    def visualize_graph(self, max_nodes_to_plot=200, figsize=(15, 12)):
        """
        Visualize a subset of the graph with node colors based on labels
        """
        if len(self.graph) == 0:
            print("Graph is empty. Build graph first.")
            return
        
        # Create subgraph if graph is too large
        if len(self.graph) > max_nodes_to_plot:
            nodes_to_keep = list(self.graph.nodes())[:max_nodes_to_plot]
            subgraph = self.graph.subgraph(nodes_to_keep)
            print(f"Visualizing subgraph with {len(subgraph)} nodes")
        else:
            subgraph = self.graph
        
        plt.figure(figsize=figsize)
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(
            subgraph, 
            k=2,
            iterations=1000,
            threshold=1e-4,
            weight='weight',
            scale=2
        )
    
        # Get node colors based on 2-way labels
        node_colors = []
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            label = node_data.get('label_2_way', '')
            if label == 0:
                node_colors.append('green')  # Real news
            elif label == 1:
                node_colors.append('red')    # Fake news
            else:
                node_colors.append('blue')   # Unknown
        
        # Get edge weights for visualization
        edge_weights = [subgraph[u][v]['weight'] * 2 for u, v in subgraph.edges()]
        
        # Draw the graph
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                              node_size=80, alpha=0.8, edgecolors='black', linewidths=0.5)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.3, 
                              edge_color='gray', width=edge_weights)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Real News (0)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Fake News (1)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Unknown')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f"Fakeddit News Similarity Graph\n"
                 f"(Nodes: {len(subgraph)}, Edges: {subgraph.number_of_edges()}, "
                 f"Similarity Threshold: {self.similarity_threshold})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Also plot weight distribution
        self.plot_weight_distribution()
    
    def plot_weight_distribution(self):
        """
        Plot distribution of edge weights
        """
        edge_weights = [data['weight'] for _, _, data in self.graph.edges(data=True)]
        
        plt.figure(figsize=(10, 6))
        plt.hist(edge_weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.similarity_threshold, color='red', linestyle='--', 
                   label=f'Threshold: {self.similarity_threshold}')
        plt.xlabel('Edge Weight (Similarity)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Edge Weights')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def save_graph(self, filename='fakeddit_similarity_graph.graphml'):
        """
        Save graph to GraphML file
        """
        nx.write_graphml(self.graph, filename)
        print(f"Graph saved to {filename}")
    
    def get_similar_news(self, node_id, top_k=5):
        """
        Get most similar news articles to a given node
        """
        if node_id not in self.graph:
            print(f"Node {node_id} not found in graph")
            return []
        
        neighbors = list(self.graph.neighbors(node_id))
        similarities = []
        
        for neighbor in neighbors:
            similarity = self.graph[node_id][neighbor]['similarity']
            neighbor_title = self.graph.nodes[neighbor].get('title', 'N/A')
            similarities.append((neighbor, similarity, neighbor_title))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def find_communities(self):
        """
        Find communities in the graph using Louvain method
        """
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph)
            
            # Add community information to nodes
            for node, community_id in partition.items():
                self.graph.nodes[node]['community'] = community_id
            
            print(f"Found {max(partition.values()) + 1} communities")
            return partition
            
        except ImportError:
            print("python-louvain package not installed. Install with: pip install python-louvain")
            return None

def main():
    """
    Main function to demonstrate the graph building process
    """
    # Initialize graph builder with adjustable parameters
    similarity_threshold = float(input("Enter similarity threshold: "))
    graph_builder = FakedditGraphBuilder(
        similarity_threshold,  # Adjust based on your needs (0.3-0.6 works well)
        max_features=300000
    )
    
    # Load and preprocess the dataset with filtering based on file
    df = graph_builder.load_dataset(
        'multimodal_train.tsv', 
        labels_file_path='mock_predictions.tsv'  # This will filter the dataset
    )
    
    print(f"Dataset loaded with {len(df)} rows")
    print("Columns:", df.columns.tolist())
    
    # Check if we have data to process
    if len(df) == 0:
        print("No data found after filtering. Please check if the IDs in file exist in database")
        return
    
    # Preprocess text
    df = graph_builder.preprocess_text(df)
    print(f"After text preprocessing: {len(df)} rows")
    
    # Build graph
    graph = graph_builder.build_graph(df)
    
    # Analyze graph
    graph_builder.analyze_graph()
    
    # Find communities
    graph_builder.find_communities()
    
    # Visualize graph
    graph_builder.visualize_graph(max_nodes_to_plot=100000)
    
    # Save graph
    graph_builder.save_graph('fakeddit_multimodal_graph.graphml')
    
    # Demonstrate finding similar news
    if graph.number_of_nodes() > 0:
        sample_node = list(graph.nodes())[0]
        similar_news = graph_builder.get_similar_news(sample_node, top_k=3)
        
        print(f"\nMost similar news to node '{graph.nodes[sample_node].get('title', 'N/A')}':")
        for node_id, similarity, title in similar_news:
            print(f"  Similarity: {similarity:.3f} - Title: {title}")

if __name__ == "__main__":
    main()