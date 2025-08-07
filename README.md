# SiftMail: Intelligent Email Classification System

## Project Overview

SiftMail is an intelligent email classification system that automatically categorizes emails into predefined categories using advanced machine learning techniques. The system combines clustering analysis and deep learning (LSTM) to provide accurate email classification for improved email management and organization.

##  Project Goals

- **Automated Email Classification**: Automatically categorize emails into meaningful categories
- **Multi-Modal Analysis**: Combine text content, metadata, and temporal features
- **Scalable Architecture**: Handle large email datasets efficiently
- **High Accuracy**: Achieve robust classification performance across different email types

## Project Structure

```
siftmail/
├── Data_Preprocess.ipynb     # Data preprocessing and feature engineering
├── Cluster.ipynb             # Clustering analysis and visualization
├── LSTM.ipynb                # LSTM model development and training
├── Datasets/                 # All dataset files
│   ├── master_dataset.csv       # Main processed dataset
│   ├── emails.csv              # Raw email data
│   ├── preprocessed_data.csv   # Preprocessed email data
│   └── ...                     # Additional processed datasets
├── Models/                   # Trained models and artifacts
│   ├── advanced_lstm_best.keras # Best performing LSTM model
│   ├── tokenizer.pickle        # Text tokenizer
│   ├── knn_model.pkl           # KNN clustering model
│   └── ...                     # Additional model files
├── Logs/                     # Training and processing logs
├── Results/                  # Model evaluation results
└── README.md                    # This file
```
Please download the datasets from: https://drive.google.com/drive/folders/1SHUE30Y7IdPTI_IVYadPiSF-fvaPByAp?usp=sharing
## 🚀 Quick Start

### Prerequisites

pip install -r requirements.txt

### Installation

1. **Clone the repository**:
   ```bash
   git clone <https://github.com/07Albert/siftmail>
   cd siftmail
   ```

2. **Download required data**:
   - Place your email dataset in the `Datasets/` folder
   - Ensure the main dataset is named `master_dataset.csv`

3. **Run the notebooks in order**:
   ```bash
   jupyter notebook
   ```

## Workflow

### 1. Data Preprocessing (`Data_Preprocess.ipynb`)

**Purpose**: Clean, transform, and prepare email data for analysis

**Key Features**:
- Email parsing and text extraction
- Text cleaning and normalization
- Feature engineering (temporal, metadata)
- Word embedding generation
- Data validation and quality checks

**Output**:
- `preprocessed_data.csv`: Cleaned email data
- `vectorized_folders_df.csv`: Text embeddings
- `master_dataset.csv`: Final processed dataset

### 2. Clustering Analysis (`Cluster.ipynb`)

**Purpose**: Discover natural groupings in email data

**Key Features**:
- Dimensionality reduction (PCA, t-SNE)
- K-means clustering
- Cluster visualization
- Label distribution analysis
- Silhouette score evaluation

**Output**:
- `Cluster_Comparison.png`: Clustering results visualization
- `PCA_TSNE_Analysis.png`: Dimensionality reduction plots
- Cluster labels for supervised learning

### 3. LSTM Model Development (`LSTM.ipynb`)

**Purpose**: Build and train deep learning models for email classification

**Key Features**:
- Bidirectional LSTM architecture
- Advanced text preprocessing
- Hyperparameter optimization
- Model evaluation and comparison
- Prediction pipeline

**Output**:
- `advanced_lstm_best.keras`: Best performing model
- `tokenizer.pickle`: Text tokenizer
- Training logs and evaluation metrics

## Architecture

### Data Pipeline

```
Raw Emails → Preprocessing → Feature Engineering → Embeddings → Clustering → Labels
                                    ↓
                              LSTM Training → Classification Model
```

### Model Architecture

**LSTM Model**:
- **Embedding Layer**: 100-dimensional word embeddings
- **Bidirectional LSTM**: 128 units with dropout (0.2)
- **Second LSTM Layer**: 64 units with dropout (0.2)
- **Dense Layers**: 64 units with regularization
- **Output Layer**: Softmax activation for multi-class classification

**Key Features**:
- Spatial dropout for regularization
- Batch normalization
- L2 regularization
- Early stopping and learning rate scheduling

##  Dataset Information

### Email Categories

The system classifies emails into the following categories:

| Label | Category | Description |
|-------|----------|-------------|
| 0 | Other | Miscellaneous emails |
| 1 | Sent | Sent emails (filtered out) |
| 2 | Unclassified | Unclassified emails (filtered out) |
| 3 | Legal/Corporate | Legal and corporate communications |
| 4 | Personal | Personal emails |
| 5 | Business | Business-related emails |
| 6 | Technical | Technical support and IT emails |
| 7 | Financial | Financial and accounting emails |
| 8 | Administrative | Administrative tasks |
| 9 | Marketing | Marketing and promotional emails |
| 10 | Education | Educational content |

### Dataset Statistics

- **Total Emails**: ~100,000+ emails
- **Features**: Text content, metadata, temporal features
- **Embedding Dimension**: 100
- **Sequence Length**: 250 tokens
- **Vocabulary Size**: 50,000 words

##  Model Performance

### LSTM Model Results

- **Accuracy**: ~63% (test set)
- **Precision**: Weighted average across classes
- **Recall**: Balanced performance across categories
- **F1-Score**: Optimized for imbalanced dataset

### Key Improvements

- **Class Weighting**: Handles imbalanced data distribution
- **Regularization**: Prevents overfitting
- **Early Stopping**: Optimizes training duration
- **Learning Rate Scheduling**: Improves convergence

##  Configuration

### Model Parameters

```python
class Config:
    # Model parameters
    MAX_WORDS = 50000
    MAX_LENGTH = 250
    EMBEDDING_DIM = 100
    
    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Regularization
    DROPOUT_RATE = 0.3
    RECURRENT_DROPOUT_RATE = 0.2
    L2_REGULARIZATION = 0.01
```

## Usage Examples

### Training a New Model

```python
# Load and preprocess data
from data_preprocessing import load_and_preprocess_data
df = load_and_preprocess_data('Datasets/master_dataset.csv')

# Train LSTM model
from lstm_model import create_and_train_lstm
model, history = create_and_train_lstm(df)

# Evaluate model
from evaluation import evaluate_model
results = evaluate_model(model, X_test, Y_test)
```

### Making Predictions

```python
# Load trained model
model = tf.keras.models.load_model('Models/advanced_lstm_best.keras')

# Predict email category
text = "Please review the quarterly financial report"
prediction = predict_email_category(text, model, tokenizer)
print(f"Predicted category: {prediction['category']}")
print(f"Confidence: {prediction['confidence']:.3f}")
```


### Sample Predictions

```python
sample_texts = [
    "What are the legal consequences of this lawsuit?",
    "Please transfer the funds to account 12345",
    "Thank you for your interest in our MBA program"
]

for text in sample_texts:
    result = predict_email_category(text, model, tokenizer)
    print(f"Text: {text}")
    print(f"Predicted: {result['category']} (confidence: {result['confidence']:.3f})")
```

## Visualization

The project includes comprehensive visualizations:

- **Cluster Analysis**: Email grouping patterns
- **Training Curves**: Model learning progress
- **Confusion Matrix**: Classification performance
- **Feature Importance**: Key classification factors

## Key Features

### Advanced Text Processing
- **Text Cleaning**: Remove special characters, normalize text
- **Tokenization**: Convert text to numerical sequences
- **Padding**: Standardize sequence lengths
- **Embedding**: Convert tokens to dense vectors

### Model Optimization
- **Hyperparameter Tuning**: Optimized architecture
- **Regularization**: Prevent overfitting
- **Class Balancing**: Handle imbalanced data
- **Early Stopping**: Efficient training

### Evaluation Metrics
- **Accuracy**: Overall classification performance
- **Precision/Recall**: Per-class performance
- **F1-Score**: Balanced metric
- **Confusion Matrix**: Detailed error analysis


### Research Directions

- **Few-shot Learning**: Handle new email categories
- **Privacy-Preserving**: Federated learning approaches
- **Explainability**: Model interpretability features
- **Domain Adaptation**: Cross-domain generalization


### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints
- Write comprehensive comments

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

Albert Hong, Braulio Matos and Derick Xu

