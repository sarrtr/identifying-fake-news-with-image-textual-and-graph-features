# Multimodal system for identifying fake news with image, textual, and graph-based features
## Project overview
### Project idea
The spread of fake news has become a serious issue in online media, often blending misleading text, manipulated images, and rapid social network propagation. This project proposes a multimodal system that combines textual, visual, and graph-based features to more effectively identify and analyze fake news.

The system is intended for everyone who does not trust provocative news and wants to protect themselves from disinformation. By providing a more comprehensive analysis, the project aims to support responsible information sharing and strengthen public trust in digital media.

### Technique

The system will contain three core modules:

- **Textual Processing:** Transformer-based NLP models extract semantic and stylistic signals from news text, enabling the system to capture misleading claims, exaggerated language, and AI-generated writing patterns.
- **Image Analysis:** Deep computer vision models (CNNs or Vision Transformers) examine attached images, detecting manipulated visuals, deepfakes, or mismatched content.
- **Graph Modeling:** Graph Neural Networks analyze the structure and dynamics of information spread to uncover coordinated or abnormal behavior.

## Datasets

- **[DeepFakeNews dataset]( https://zenodo.org/records/11186584):** contains a total of 509,916 images and has been enriched with 254,958 deepfake; it is also perfectly balanced, containing an equal number of pristine (authentic) and generated (deepfake) images.
- **[Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection](https://github.com/entitize/Fakeddit?tab=readme-ov-file):** multimodal text and image data, metadata, comment data, and finegrained fake news categorization; consists of over 1 million samples from multiple categories of fake news.
- **[FakeNewsNet dataset](https://www.kaggle.com/datasets/mdepak/fakenewsnet):** fake news data repository FakeNewsNet, which contains two comprehensive datasets that includes news content, social context, and dynamic information.
