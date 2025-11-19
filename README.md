# Multimodal system for identifying fake news with image, textual, and graph-based features
## Project overview
### Project idea
The spread of fake news has become a critical issue in digital media, often combining misleading text with manipulated images and propagating rapidly through social networks. This project addresses the challenge through a multi-faceted approach that integrates:

The system is intended for everyone who does not trust provocative news and wants to protect themselves from disinformation. By providing a more comprehensive analysis, the project aims to support responsible information sharing and strengthen public trust in digital media. for fakeddit

### Key features

- **Multimodal Analysis:** Combining text and image features using transformer architectures.
- **Explainable AI (XAI):** Providing reasoning behind detection decisions with GradCAM and LIME methods.
- **Graph Construction:** Analyzing relationships between news items based on Fakeddit dataset through similarity network.

## Technique

Our system employs a multimodal framework that integrates textual and visual information through symmetric cross-attention mechanisms.

Core model components:
- **Text Encoder:** Pre-trained BERT for processing news headlines
- **Image Encoder:** Vision Transformer (ViT) for visual feature extraction
- **Cross-Attention:** Bidirectional attention between text and image modalities
- **Contrastive Learning:** Alignment of multimodal representations in shared space

![model architecture](https://github.com/sarrtr/identifying-fake-news-with-image-textual-and-graph-features/blob/main/assets/model_architecture.png?raw=true)

## Dataset

**[Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection](https://github.com/entitize/Fakeddit?tab=readme-ov-file):** sourced from Reddit posts and combines text, images, metadata, and community signals to enable large-scale multimodal fake-news research; consists of over 1 million samples from multiple categories of fake news.

## Installation & Usage

### Quick start

Prerequisits: 
[Docker](https://www.docker.com/products/docker-desktop/) installed.

Installation steps:
1. Clone the repository.
``` git clone https://github.com/sarrtr/identifying-fake-news-with-image-textual-and-graph-features.git ```

2. Download [model, tokenizer](https://drive.google.com/drive/folders/1Cbc4gG6F7qb28IgFBhNVgyWNncOtU6Tt, and [dataset for graph](https://drive.google.com/file/d/1rQw68yzZoa6cLwDNslyjT1rRwK2uGcyL/view).
3. Put model and tokenizer to ```custom_model/models/checkpoints```, and folders 'images' and 'text' from archive to ```custom_model/code/graph/dataset```.
4. Run core app:

``` cd custom_model/code/deployment
docker-compose up --build ```

5. Run graph builder:

``` cd custom_model/code/graph
python graph_builder.py ```

### Usage example

User enters news title, uploads news image, and tunes LIME for Image hyperparameter - number of samples. It affects execution speed and stability: more samples we have - more time is needed for inference and more stable result is.

![input](https://github.com/sarrtr/identifying-fake-news-with-image-textual-and-graph-features/blob/main/assets/input.png?raw=true)

After pressing button "Analyze" user can see input and model prediction with confidence.

![input with prediction](https://github.com/sarrtr/identifying-fake-news-with-image-textual-and-graph-features/blob/main/assets/input_with_pred.png?raw=true)

Then text explanation takes place. Weights of tokens show influence of tokens on resulting prediction: positive weights increase the probability of predicting resulting class, negative weights work in opposite way. 

![text explanation](https://github.com/sarrtr/identifying-fake-news-with-image-textual-and-graph-features/blob/main/assets/text_explain.png?raw=true)

After that goes image explanation. 

Grad-CAM: creates a "heat map" of importance, where orange zones have high importance for prediction, and blue zones are indifferent for model.

LIME overlay: green zones have positive effect on predicted class, red zones have opposite effect.

![image explanation](https://github.com/sarrtr/identifying-fake-news-with-image-textual-and-graph-features/blob/main/assets/img_explain.jpg?raw=true)

The graph shows position of input among other 1500 samples from Fakeddit dataset. Nodes are considered to be not connected if edge between them has weight less than 0.4.

![graph](https://github.com/sarrtr/identifying-fake-news-with-image-textual-and-graph-features/blob/main/assets/graph.png?raw=true)

![graph zoom](https://github.com/sarrtr/identifying-fake-news-with-image-textual-and-graph-features/blob/main/assets/graph_zoom.png?raw=true)

After the main graph, charts with detailed analysis show up.

![graph analysis](https://github.com/sarrtr/identifying-fake-news-with-image-textual-and-graph-features/blob/main/assets/graph_analysis.png?raw=true)

## License
This project is licensed under the MIT License.