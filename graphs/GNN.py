import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import xml.etree.ElementTree as ET
import warnings
from collections import Counter

class FakedditGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, model_type='gcn'):
        super(FakedditGNN, self).__init__()
        self.model_type = model_type
        
        if model_type == 'gcn':
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'gat':
            self.conv1 = GATConv(num_features, hidden_channels, heads=4, concat=True)
            self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True)
            self.conv3 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False)
        elif model_type == 'sage':
            self.conv1 = SAGEConv(num_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_weight=None):
        if self.model_type == 'gat':
            x = F.elu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = F.elu(self.conv2(x, edge_index))
            x = self.dropout(x)
            x = F.elu(self.conv3(x, edge_index))
        else:
            # For GCN and GraphSAGE, use edge weights if available
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, edge_index, edge_weight))
            x = self.dropout(x)
            x = F.relu(self.conv3(x, edge_index, edge_weight))
        
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def parse_graphml_file(file_path):
    """Parse GraphML file and convert to PyG Data object"""
    ET.register_namespace('', "http://graphml.graphdrawing.org/xmlns")
    
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    ns = {'ns': 'http://graphml.graphdrawing.org/xmlns'}
    
    # Extract key definitions
    key_definitions = {}
    for key_elem in root.findall('.//ns:key', ns):
        key_id = key_elem.get('id')
        key_name = key_elem.get('attr.name')
        key_type = key_elem.get('attr.type')
        key_definitions[key_id] = (key_name, key_type)
    
    print(f"Found key definitions: {key_definitions}")
    
    nodes = []
    node_features = []
    node_labels = []
    node_mapping = {}
    edges = []
    
    # Find the graph element
    graph_elem = root.find('.//ns:graph', ns)
    if graph_elem is None:
        graph_elem = root.find('.//graph')
    
    if graph_elem is None:
        print("Error: Could not find graph element")
        return nodes, node_features, node_labels, edges, node_mapping
    
    # Process nodes
    node_elems = graph_elem.findall('.//ns:node', ns)
    if not node_elems:
        node_elems = graph_elem.findall('.//node')
    
    print(f"Found {len(node_elems)} node elements")
    
    for i, node_elem in enumerate(node_elems):
        node_id = node_elem.get('id')
        node_mapping[node_id] = i
        
        features = {}
        label = None
        
        data_elems = node_elem.findall('.//ns:data', ns)
        if not data_elems:
            data_elems = node_elem.findall('.//data')
        
        for data_elem in data_elems:
            key_id = data_elem.get('key')
            value = data_elem.text
            
            if key_id in key_definitions:
                key_name, key_type = key_definitions[key_id]
                
                if key_name == 'label_2_way':
                    if value and value != 'nan':
                        try:
                            label = int(value)
                        except ValueError:
                            label = None
                
                elif key_name == 'initial_label':
                    if value and value != 'nan':
                        try:
                            features['initial_label'] = float(value)
                        except ValueError:
                            features['initial_label'] = 0.0
                    else:
                        features['initial_label'] = 0.0
                
                elif key_name == 'score':
                    if value and value != 'nan':
                        try:
                            features['score'] = int(value)
                        except ValueError:
                            features['score'] = 0
                    else:
                        features['score'] = 0
                
                elif key_name == 'upvote_ratio':
                    if value and value != 'nan':
                        try:
                            features['upvote_ratio'] = float(value)
                        except ValueError:
                            features['upvote_ratio'] = 0.0
                    else:
                        features['upvote_ratio'] = 0.0
                
                elif key_name == 'num_comments':
                    if value and value != 'nan':
                        try:
                            features['num_comments'] = float(value)
                        except ValueError:
                            features['num_comments'] = 0.0
                    else:
                        features['num_comments'] = 0.0
                
                elif key_name == 'hasImage':
                    if value and value.lower() == 'true':
                        features['hasImage'] = 1.0
                    else:
                        features['hasImage'] = 0.0
                
                elif key_name == 'community':
                    if value and value != 'nan':
                        try:
                            features['community'] = int(value)
                        except ValueError:
                            features['community'] = 0
                    else:
                        features['community'] = 0
        
        # Ensure all expected features are present
        expected_features = ['initial_label', 'score', 'upvote_ratio', 'num_comments', 'hasImage', 'community']
        for feat in expected_features:
            if feat not in features:
                features[feat] = 0.0
        
        nodes.append(node_id)
        node_features.append(features)
        node_labels.append(label)
    
    # Process edges
    edge_elems = graph_elem.findall('.//ns:edge', ns)
    if not edge_elems:
        edge_elems = graph_elem.findall('.//edge')
    
    print(f"Found {len(edge_elems)} edge elements")
    
    for edge_elem in edge_elems:
        source = edge_elem.get('source')
        target = edge_elem.get('target')
        
        weight = 1.0
        data_elems = edge_elem.findall('.//ns:data', ns)
        if not data_elems:
            data_elems = edge_elem.findall('.//data')
        
        for data_elem in data_elems:
            key_id = data_elem.get('key')
            if key_id in key_definitions:
                key_name, _ = key_definitions[key_id]
                if key_name == 'weight' and data_elem.text:
                    try:
                        weight = float(data_elem.text)
                    except ValueError:
                        weight = 1.0
        
        if source in node_mapping and target in node_mapping:
            edges.append((node_mapping[source], node_mapping[target], weight))
    
    print(f"Successfully processed {len(nodes)} nodes and {len(edges)} edges")
    return nodes, node_features, node_labels, edges, node_mapping

def create_pyg_data(node_features, node_labels, edges):
    """Create PyG Data object from parsed data"""
    
    # Create feature matrix
    feature_keys = ['initial_label', 'score', 'upvote_ratio', 'num_comments', 'hasImage', 'community']
    x_list = []
    y_list = []
    valid_indices = []
    
    for i, (features, label) in enumerate(zip(node_features, node_labels)):
        if label is not None:
            feature_vector = [features.get(key, 0.0) for key in feature_keys]
            x_list.append(feature_vector)
            y_list.append(label)
            valid_indices.append(i)
    
    if len(x_list) == 0:
        print("Error: No valid nodes with labels found!")
        return None, []
    
    x = torch.tensor(x_list, dtype=torch.float)
    y = torch.tensor(y_list, dtype=torch.long)
    
    print(f"Created feature matrix with shape: {x.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {torch.bincount(y).tolist()}")
    
    # Create edge index and edge weights
    edge_sources = []
    edge_targets = []
    edge_weights = []
    
    valid_indices_set = set(valid_indices)
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
    
    for src, tgt, weight in edges:
        if src in valid_indices_set and tgt in valid_indices_set:
            edge_sources.append(index_mapping[src])
            edge_targets.append(index_mapping[tgt])
            edge_weights.append(weight)
    
    if len(edge_sources) == 0:
        print("Warning: No edges found between valid nodes. Creating self-loops...")
        for i in range(len(valid_indices)):
            edge_sources.append(i)
            edge_targets.append(i)
            edge_weights.append(1.0)
    
    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    data = Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight)
    
    print(f"Final graph: {data}")
    return data, valid_indices

def calculate_class_weights(labels):
    """Calculate class weights for imbalanced datasets"""
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    weights = total_samples / (num_classes * class_counts.float())
    return weights

def safe_classification_report(y_true, y_pred, target_names=None):
    """Safe classification report that handles undefined metrics"""
    with warnings.catch_warnings():
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

def train_model(model, data, train_mask, val_mask, test_mask, epochs=200, use_class_weights=True):
    """Train the GNN model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Handle class imbalance with weights
    if use_class_weights:
        class_weights = calculate_class_weights(data.y[train_mask])
        criterion = torch.nn.NLLLoss(weight=class_weights)
        print(f"Using class weights: {class_weights.tolist()}")
    else:
        criterion = torch.nn.NLLLoss()
    
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    patience = 50
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.edge_weight)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index, data.edge_weight)
                val_pred = val_out[val_mask].argmax(dim=1)
                val_acc = accuracy_score(data.y[val_mask].numpy(), val_pred.numpy())
                
            train_losses.append(loss.item())
            val_accuracies.append(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 50 == 0:
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                # Load best model
                model.load_state_dict(best_model_state)
                break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index, data.edge_weight)
        test_pred = test_out[test_mask].argmax(dim=1)
        test_acc = accuracy_score(data.y[test_mask].numpy(), test_pred.numpy())
        
        train_pred = test_out[train_mask].argmax(dim=1)
        val_pred = test_out[val_mask].argmax(dim=1)
        
        print(f"\nFinal Results:")
        print(f"Train Accuracy: {accuracy_score(data.y[train_mask].numpy(), train_pred.numpy()):.4f}")
        print(f"Val Accuracy: {accuracy_score(data.y[val_mask].numpy(), val_pred.numpy()):.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Safe classification report
        print(f"\nClassification Report (Test):")
        try:
            # Import here to avoid issues
            from sklearn.exceptions import UndefinedMetricWarning
            report = safe_classification_report(
                data.y[test_mask].numpy(), 
                test_pred.numpy(), 
                target_names=['Class 0', 'Class 1']
            )
            print(report)
        except ImportError:
            # Fallback if sklearn version doesn't have UndefinedMetricWarning
            print(classification_report(
                data.y[test_mask].numpy(), 
                test_pred.numpy(), 
                target_names=['Class 0', 'Class 1'],
                zero_division=0
            ))
        
        # Confusion matrix for better insight
        cm = confusion_matrix(data.y[test_mask].numpy(), test_pred.numpy())
        print(f"Confusion Matrix:\n{cm}")
    
    return model, train_losses, val_accuracies

def main():
    # Parse the GraphML file
    print("Parsing GraphML file...")
    nodes, node_features, node_labels, edges, node_mapping = parse_graphml_file(
        'fakeddit_multimodal_graph.graphml'  # Replace with your actual file path
    )
    
    if len(nodes) == 0:
        print("Error: No nodes found in the file!")
        return
    
    # Create PyG data object
    data, valid_indices = create_pyg_data(node_features, node_labels, edges)
    
    if data is None:
        print("Error: Could not create PyG data object!")
        return
    
    print(f"\nGraph Statistics:")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {data.num_features}")
    print(f"Number of classes: {len(torch.unique(data.y))}")
    print(f"Class distribution: {torch.bincount(data.y).tolist()}")
    
    # Check if we have enough samples for all classes
    class_counts = torch.bincount(data.y)
    if len(class_counts) < 2:
        print("Warning: Only one class present in the data. Cannot perform binary classification.")
        return
    
    # Create train/val/test masks with stratification
    indices = torch.arange(data.num_nodes)
    
    if len(indices) > 1:
        try:
            train_idx, temp_idx = train_test_split(
                indices, test_size=0.4, stratify=data.y.numpy()
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, stratify=data.y[temp_idx].numpy()
            )
        except ValueError as e:
            print(f"Stratification failed: {e}. Using random split instead.")
            train_idx, temp_idx = train_test_split(indices, test_size=0.4)
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.5)
    else:
        print("Warning: Not enough samples for proper train/val/test split")
        return
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # Initialize and train GNN models
    models = {
        'GCN': FakedditGNN(data.num_features, 64, 2, 'gcn'),
        'GAT': FakedditGNN(data.num_features, 32, 2, 'gat'),
        # 'GraphSAGE': FakedditGNN(data.num_features, 64, 2, 'sage')
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name} model...")
        print(f"{'='*50}")
        
        trained_model, train_losses, val_accuracies = train_model(
            model, data, train_mask, val_mask, test_mask, epochs=300
        )
        
        results[model_name] = {
            'model': trained_model,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }

if __name__ == "__main__":
    main()