import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageFile
import torch
from torchvision import transforms
import warnings
import random
warnings.filterwarnings('ignore')

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageTitleGraphGenerator:
    def __init__(self, text_file_path, images_folder_path, similarity_threshold=0.5):
        self.text_file_path = text_file_path
        self.images_folder_path = images_folder_path
        self.similarity_threshold = similarity_threshold
        self.data = None
        self.graph = nx.Graph()
        
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Define embedding dimensions
        self.text_embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        self.image_embedding_dim = 512  # CLIP image dimension
        
        # Try to load CLIP model with proper error handling
        self.use_clip = False
        try:
            import clip
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.use_clip = True
            print("Successfully loaded OpenAI CLIP model")
        except (ImportError, AttributeError, Exception) as e:
            print(f"CLIP not available: {e}. Using text-only embeddings.")
            self.clip_model = None
            self.preprocess = None
        
        # Initialize text model (always available)
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Store embeddings separately for analysis
        self.text_embeddings = {}
        self.image_embeddings = {}
        self.similarity_matrix = None
        
    def load_data(self):
        """Load and parse the text file data"""
        self.data = pd.read_csv(self.text_file_path, sep='\t')
        print(f"Loaded {len(self.data)} entries")
        return self.data
        
    def find_image_path(self, image_id):
        """Find the actual image file path for a given ID"""
        if not os.path.exists(self.images_folder_path):
            print(f"Images folder not found: {self.images_folder_path}")
            return None
            
        possible_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        
        for ext in possible_extensions:
            image_path = os.path.join(self.images_folder_path, f"{image_id}{ext}")
            if os.path.exists(image_path):
                return image_path
        
        # Try without extension (in case IDs are the full filenames)
        image_path = os.path.join(self.images_folder_path, image_id)
        if os.path.exists(image_path):
            return image_path
            
        return None
    
    def get_zero_image_embedding(self):
        """Return zero embedding for unprocessable images"""
        return np.zeros(self.image_embedding_dim)
    
    def get_image_embedding(self, image_path):
        """Extract embedding from image using available models"""
        if not self.use_clip:
            return self.get_zero_image_embedding()
            
        try:
            # Check if file exists and is readable
            if not os.path.exists(image_path):
                return self.get_zero_image_embedding()
                
            # Try to open and verify the image
            try:
                with Image.open(image_path) as img:
                    img.verify()  # Verify it's a valid image
            except Exception:
                return self.get_zero_image_embedding()
            
            # Now process the image for embedding
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_embedding = image_features.cpu().numpy().flatten()
                
            # Ensure the embedding has the right dimension
            if len(image_embedding) != self.image_embedding_dim:
                return self.get_zero_image_embedding()
                
            return image_embedding / np.linalg.norm(image_embedding)  # Normalize
            
        except Exception:
            return self.get_zero_image_embedding()
    
    def get_text_embedding(self, text):
        """Extract embedding from text using only sentence transformer"""
        try:
            embedding = self.text_model.encode([text])[0]
            return embedding / np.linalg.norm(embedding)  # Normalize
        except Exception as e:
            print(f"Text encoding failed for '{text}': {e}")
            # Return zero embedding for text
            zero_embedding = np.zeros(self.text_embedding_dim)
            return zero_embedding
    
    def add_input_image(self, input_image_path, input_text_path):
        """Add the input image and its title to the graph"""
        if not os.path.exists(input_image_path):
            print(f"Input image not found: {input_image_path}")
            return None
        if not os.path.exists(input_text_path):
            print(f"Input text file not found: {input_text_path}")
            return None
            
        # Read the input title
        try:
            with open(input_text_path, 'r', encoding='utf-8') as f:
                input_title = f.read().strip()
        except Exception as e:
            print(f"Error reading input text file: {e}")
            return None
        
        # Get embeddings for input image and title
        input_image_embedding = self.get_image_embedding(input_image_path)
        input_text_embedding = self.get_text_embedding(input_title)
        
        # Create input node ID
        input_node_id = "input_image"
        
        # Add input node to graph with "unknown" label
        self.graph.add_node(input_node_id,
                          image_id=input_node_id,
                          title=input_title,
                          label="unknown",
                          image_path=input_image_path,
                          has_image=True,
                          is_input=True)
        
        # Store embeddings for the input node
        self.text_embeddings[input_node_id] = input_text_embedding
        self.image_embeddings[input_node_id] = input_image_embedding
        
        print(f"Added input image: {input_title}")
        return input_node_id
    
    def build_graph(self, sample_size=None, include_input=True, input_image_path="input_image.jpg", input_text_path="input_text.txt"):
        """Build the graph with optional sampling and input image"""
        if self.data is None:
            self.load_data()
        
        # Sample the data if sample_size is specified
        if sample_size is not None and sample_size < len(self.data):
            self.sampled_data = self.data.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} random entries from dataset")
        else:
            self.sampled_data = self.data
            print(f"Using all {len(self.data)} entries from dataset")
        
        print("Processing images and generating embeddings...")
        
        # Store valid nodes and their embeddings
        nodes = []
        text_embeddings_list = []
        image_embeddings_list = []
        processed_count = 0
        image_success_count = 0
        
        # Process sampled dataset entries
        for idx, row in self.sampled_data.iterrows():
            image_id = row['id']
            title = row['clean_title']
            label = row['2_way_label']
            
            image_path = self.find_image_path(image_id)
            
            if image_path is None:
                image_path = f"not_found_{image_id}"
            
            # Get embeddings separately
            text_embedding = self.get_text_embedding(title)
            image_embedding = self.get_image_embedding(image_path)
            
            # Store embeddings for analysis
            self.text_embeddings[image_id] = text_embedding
            self.image_embeddings[image_id] = image_embedding
            
            # Verify embeddings have correct dimensions
            if (text_embedding is not None and len(text_embedding) == self.text_embedding_dim and
                image_embedding is not None and len(image_embedding) == self.image_embedding_dim):
                
                node_id = f"{image_id}"
                nodes.append(node_id)
                text_embeddings_list.append(text_embedding)
                image_embeddings_list.append(image_embedding)
                
                # Track image processing success
                has_image = image_path is not None and "not_found" not in image_path and not np.all(image_embedding == 0)
                if has_image:
                    image_success_count += 1
                
                # Add node to graph
                self.graph.add_node(node_id, 
                                  image_id=image_id,
                                  title=title, 
                                  label=label,
                                  image_path=image_path if "not_found" not in image_path else None,
                                  has_image=has_image,
                                  is_input=False)
                processed_count += 1
        
        print(f"Successfully processed {processed_count} nodes from dataset")
        print(f"Images successfully processed: {image_success_count}/{processed_count}")
        
        # Add input image if requested
        input_node_id = None
        if include_input:
            input_node_id = self.add_input_image(input_image_path, input_text_path)
            if input_node_id:
                nodes.append(input_node_id)
                text_embeddings_list.append(self.text_embeddings[input_node_id])
                image_embeddings_list.append(self.image_embeddings[input_node_id])
                processed_count += 1
        
        if not nodes:
            print("No valid nodes found!")
            return
        
        # Compute similarity matrices separately
        print("Computing similarity matrices...")
        try:
            text_embeddings_array = np.array(text_embeddings_list)
            image_embeddings_array = np.array(image_embeddings_list)
            
            # Compute text and image similarities
            text_similarity = cosine_similarity(text_embeddings_array)
            image_similarity = cosine_similarity(image_embeddings_array)
            
            # Combine similarities (you can adjust weights here)
            # For nodes with valid images, use average of text and image similarity
            # For nodes without valid images, use text similarity only
            combined_similarity = np.zeros_like(text_similarity)
            
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if i == j:
                        combined_similarity[i, j] = 1.0
                    else:
                        has_image_i = self.graph.nodes[nodes[i]]['has_image']
                        has_image_j = self.graph.nodes[nodes[j]]['has_image']
                        
                        if has_image_i and has_image_j:
                            # Both have images, average text and image similarity
                            combined_similarity[i, j] = (text_similarity[i, j] + image_similarity[i, j]) / 2
                        else:
                            # At least one doesn't have image, use text similarity only
                            combined_similarity[i, j] = text_similarity[i, j]
            
            self.similarity_matrix = combined_similarity
            
            # Add edges based on similarity threshold
            print("Building graph edges...")
            edge_count = 0
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    similarity = combined_similarity[i, j]
                    
                    if similarity > self.similarity_threshold:
                        self.graph.add_edge(nodes[i], nodes[j], weight=float(similarity))
                        edge_count += 1
            
            print(f"Graph built with {len(nodes)} nodes and {edge_count} edges")
            
        except Exception as e:
            print(f"Error building similarity matrix: {e}")
            print("Creating graph with no edges...")
    
    def visualize_graph(self):
        """Visualize the graph with simple node colors based on label only"""
        if len(self.graph.nodes()) == 0:
            print("No nodes to visualize!")
            return
            
        plt.figure(figsize=(12, 10))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Get node colors based on labels and input status
        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            label = self.graph.nodes[node].get('label', 0)
            is_input = self.graph.nodes[node].get('is_input', False)
            
            if is_input:
                # Red for input image
                node_colors.append('red')
                node_sizes.append(200)  # Larger size for input image
            elif label == "unknown":
                # Red for unknown (shouldn't happen except for input)
                node_colors.append('red')
                node_sizes.append(200)
            else:
                # Use green for label 0, blue for label 1
                node_colors.append('green' if label == 0 else 'blue')
                node_sizes.append(80)
        
        # Get edge weights for coloring
        if self.graph.edges():
            edge_weights = [self.graph[u][v]['weight'] for u, v in self.graph.edges()]
            
            # Draw the graph with edges
            nx.draw_networkx_edges(self.graph, pos, edge_color=edge_weights, 
                                  edge_cmap=plt.cm.Blues, width=1.5, alpha=0.6)
        else:
            edge_weights = []
        
        # Draw nodes without any labels
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8)
        
        # Remove axis
        plt.axis('off')
        
        # Create a simple legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Label 0'),
            Patch(facecolor='blue', label='Label 1'),
            Patch(facecolor='red', label='Input')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f"Image-Title Similarity Graph\n"
                 f"Nodes: {len(self.graph.nodes())}, Edges: {len(self.graph.edges())}\n"
                 f"Similarity Threshold: {self.similarity_threshold}")
        plt.tight_layout()
        plt.show()
    
    def plot_similarity_distribution(self):
        """Plot distribution of similarity scores"""
        if self.similarity_matrix is None:
            print("No similarity matrix available")
            return
        
        # Get all similarity values (excluding self-similarity)
        similarities = []
        n = len(self.similarity_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(self.similarity_matrix[i, j])
        
        if not similarities:
            print("No similarities to plot")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Similarity Distribution Analysis', fontsize=16)
        
        # Plot 1: Histogram of all similarities
        axes[0, 0].hist(similarities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.similarity_threshold, color='red', linestyle='--', 
                          label=f'Threshold: {self.similarity_threshold}')
        axes[0, 0].set_xlabel('Similarity Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of All Similarity Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cumulative distribution
        sorted_sims = np.sort(similarities)
        y_vals = np.arange(len(sorted_sims)) / float(len(sorted_sims))
        axes[0, 1].plot(sorted_sims, y_vals, linewidth=2)
        axes[0, 1].axvline(self.similarity_threshold, color='red', linestyle='--',
                          label=f'Threshold: {self.similarity_threshold}')
        axes[0, 1].set_xlabel('Similarity Score')
        axes[0, 1].set_ylabel('Cumulative Probability')
        axes[0, 1].set_title('Cumulative Distribution of Similarity Scores')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Box plot
        axes[1, 0].boxplot(similarities, vert=True)
        axes[1, 0].set_ylabel('Similarity Score')
        axes[1, 0].set_title('Box Plot of Similarity Scores')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Statistics
        axes[1, 1].axis('off')
        stats_text = f"""
        Similarity Statistics:
        
        Total pairs: {len(similarities)}
        Mean similarity: {np.mean(similarities):.4f}
        Median similarity: {np.median(similarities):.4f}
        Std deviation: {np.std(similarities):.4f}
        Min similarity: {np.min(similarities):.4f}
        Max similarity: {np.max(similarities):.4f}
        
        Current threshold: {self.similarity_threshold}
        Edges created: {len(self.graph.edges())}
        Percentage above threshold: {100 * np.mean(np.array(similarities) > self.similarity_threshold):.2f}%
        """
        axes[1, 1].text(0.1, 0.9, stats_text, fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def analyze_modality_similarities(self):
        """Compare text and image similarity distributions"""
        if not self.text_embeddings or not self.image_embeddings:
            print("No embeddings available for analysis")
            return
        
        # Get a sample of nodes that have both modalities
        valid_nodes = [node_id for node_id in self.graph.nodes() 
                      if self.graph.nodes[node_id].get('has_image', False) and node_id != "input_image"]
        
        if len(valid_nodes) < 2:
            print("Not enough nodes with both modalities for comparison")
            return
        
        # Compute text and image similarities for valid nodes
        text_sims = []
        image_sims = []
        
        for i, node1 in enumerate(valid_nodes):
            for j, node2 in enumerate(valid_nodes):
                if i < j:  # Avoid duplicates and self-comparison
                    idx1 = list(self.graph.nodes()).index(node1)
                    idx2 = list(self.graph.nodes()).index(node2)
                    text_sims.append(cosine_similarity(
                        [self.text_embeddings[node1]], 
                        [self.text_embeddings[node2]]
                    )[0][0])
                    image_sims.append(cosine_similarity(
                        [self.image_embeddings[node1]], 
                        [self.image_embeddings[node2]]
                    )[0][0])
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Text vs Image Similarity Comparison', fontsize=16)
        
        # Scatter plot
        axes[0].scatter(text_sims, image_sims, alpha=0.6)
        axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.8)  # Diagonal line
        axes[0].set_xlabel('Text Similarity')
        axes[0].set_ylabel('Image Similarity')
        axes[0].set_title('Text vs Image Similarity Scatter Plot')
        axes[0].grid(True, alpha=0.3)
        
        # Distribution comparison
        axes[1].hist(text_sims, bins=30, alpha=0.7, label='Text', color='blue')
        axes[1].hist(image_sims, bins=30, alpha=0.7, label='Image', color='red')
        axes[1].set_xlabel('Similarity Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print correlation
        correlation = np.corrcoef(text_sims, image_sims)[0, 1]
        print(f"Correlation between text and image similarities: {correlation:.4f}")
        
        return fig
    
    def get_graph_statistics(self):
        """Print graph statistics"""
        if len(self.graph.nodes()) == 0:
            print("Graph is empty!")
            return
        
        print("\n=== Graph Statistics ===")
        print(f"Number of nodes: {len(self.graph.nodes())}")
        print(f"Number of edges: {len(self.graph.edges())}")
        
        # Count nodes with/without images
        has_image_count = sum(1 for node in self.graph.nodes() 
                            if self.graph.nodes[node].get('has_image', False))
        no_image_count = len(self.graph.nodes()) - has_image_count
        print(f"Nodes with images: {has_image_count}")
        print(f"Nodes without images: {no_image_count}")
        
        # Count labels and input
        label_0_count = sum(1 for node in self.graph.nodes() 
                           if self.graph.nodes[node].get('label', 0) == 0 and not self.graph.nodes[node].get('is_input', False))
        label_1_count = sum(1 for node in self.graph.nodes() 
                           if self.graph.nodes[node].get('label', 0) == 1 and not self.graph.nodes[node].get('is_input', False))
        input_count = sum(1 for node in self.graph.nodes() 
                         if self.graph.nodes[node].get('is_input', False))
        
        print(f"Label 0 nodes: {label_0_count}")
        print(f"Label 1 nodes: {label_1_count}")
        print(f"Input nodes: {input_count}")
        
        if len(self.graph.nodes()) > 0:
            degrees = [deg for _, deg in self.graph.degree()]
            print(f"Average degree: {np.mean(degrees):.2f}")
            print(f"Maximum degree: {max(degrees)}")
            print(f"Minimum degree: {min(degrees)}")
        
        # Connected components
        components = list(nx.connected_components(self.graph))
        print(f"Number of connected components: {len(components)}")
        
        if len(components) > 0:
            largest_component = max(components, key=len)
            print(f"Largest component size: {len(largest_component)}")
    
    def save_graph(self, output_file="image_title_graph.graphml"):
        """Save graph to file"""
        nx.write_graphml(self.graph, output_file)
        print(f"Graph saved as {output_file}")
    
    def find_similar_nodes(self, node_id, top_k=5):
        """Find most similar nodes to a given node"""
        if node_id not in self.graph:
            print(f"Node {node_id} not found in graph")
            return
        
        similarities = []
        for neighbor in self.graph.neighbors(node_id):
            weight = self.graph[node_id][neighbor]['weight']
            similarities.append((neighbor, weight))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {top_k} most similar nodes to '{node_id}':")
        for i, (neighbor, similarity) in enumerate(similarities[:top_k]):
            node_data = self.graph.nodes[neighbor]
            has_image = node_data.get('has_image', False)
            image_status = "Has Image" if has_image else "No Image"
            print(f"{i+1}. {neighbor} (Similarity: {similarity:.3f}, {image_status})")
            print(f"   Title: {node_data['title']}")
            print(f"   Label: {node_data['label']}")

def main():
    # Configuration
    TEXT_FILE_PATH = "text/text.txt"
    IMAGES_FOLDER_PATH = "images"
    INPUT_IMAGE_PATH = "input_image.jpg"
    INPUT_TEXT_PATH = "input_text.txt"
    SIMILARITY_THRESHOLD = 0.4  # Adjust this based on your needs
    SAMPLE_SIZE = 1500  # Set to None to use all data, or specify a number for sampling
    
    # Initialize graph generator
    generator = ImageTitleGraphGenerator(
        text_file_path=TEXT_FILE_PATH,
        images_folder_path=IMAGES_FOLDER_PATH,
        similarity_threshold=SIMILARITY_THRESHOLD
    )
    
    # Build the graph with sampling and input image
    generator.build_graph(sample_size=SAMPLE_SIZE, include_input=True, 
                         input_image_path=INPUT_IMAGE_PATH, input_text_path=INPUT_TEXT_PATH)
    
    # Get statistics
    generator.get_graph_statistics()
    
    # Visualize the graph
    generator.visualize_graph()
    
    # Plot similarity distribution
    generator.plot_similarity_distribution()
    
    # Compare text vs image similarities
    generator.analyze_modality_similarities()
    
    # Save the graph
    generator.save_graph()

if __name__ == "__main__":
    main()