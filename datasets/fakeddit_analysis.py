import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

class FakedditAnalyzer:
    def __init__(self):
        self.data = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        
    def load_data(self, train_path, validation_path, test_path):
        """
        Load Fakeddit dataset from TSV files
        """
        print("Loading datasets...")

        self.train_data = pd.read_csv(train_path, sep='\t')
        print(f"Training data shape: {self.train_data.shape}")
        
        self.validation_data = pd.read_csv(validation_path, sep='\t')
        print(f"Validation data shape: {self.validation_data.shape}")
        
        self.test_data = pd.read_csv(test_path, sep='\t')
        print(f"Test data shape: {self.test_data.shape}")
        
        # Combine all data for some analyses
        self.data = pd.concat([self.train_data, self.validation_data, self.test_data], ignore_index=True)
        print(f"Combined data shape: {self.data.shape}")
        
        return self.data
    
    def basic_analysis(self, dataset_name="combined"):
        """Perform basic data analysis on specified dataset"""
        if dataset_name == "train":
            data = self.train_data
        elif dataset_name == "validation":
            data = self.validation_data
        elif dataset_name == "test":
            data = self.test_data
        else:
            data = self.data
            
        print(f"\n=== Basic Analysis for {dataset_name} dataset ===")
        print("Shape:", data.shape)
        print("\nColumns:", data.columns.tolist())
        print("\nData Types:\n", data.dtypes)
        print("\nMissing Values:\n", data.isnull().sum())
        
        # Check for label columns
        label_cols = [col for col in data.columns if 'label' in col]
        if label_cols:
            for label_col in label_cols:
                if label_col in data.columns:
                    print(f"\nValue counts for {label_col}:")
                    print(data[label_col].value_counts())
        
        return data
    
    def visualize_label_distribution(self):
        """Visualize the distribution of labels across all datasets"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Check for label columns
        label_cols = [col for col in self.data.columns if 'label' in col]
        
        if not label_cols:
            print("No label columns found in the dataset")
            return
        
        for i, label_col in enumerate(label_cols[:4]):
            row, col = i // 2, i % 2
            
            # Count labels in each dataset
            train_counts = self.train_data[label_col].value_counts()
            val_counts = self.validation_data[label_col].value_counts()
            test_counts = self.test_data[label_col].value_counts()
            
            # Create DataFrame for plotting
            plot_data = pd.DataFrame({
                'Train': train_counts,
                'Validation': val_counts,
                'Test': test_counts
            }).fillna(0)
            
            plot_data.plot(kind='bar', ax=axes[row, col])
            axes[row, col].set_title(f'Distribution of {label_col} by Dataset')
            axes[row, col].set_xlabel('Label')
            axes[row, col].set_ylabel('Count')
            axes[row, col].legend()
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def text_analysis(self, text_column='Fakeddit'):
        """Analyze text features in the dataset"""
        if text_column not in self.data.columns:
            print(f"Column '{text_column}' not found in dataset")
            # Try to find a text column
            text_candidates = [col for col in self.data.columns if 'text' in col.lower() or 'title' in col.lower()]
            if text_candidates:
                text_column = text_candidates[0]
                print(f"Using '{text_column}' instead")
            else:
                print("No text columns found")
                return
        
        print(f"\n=== Text Analysis for {text_column} ===")
        
        self.data['text_length'] = self.data[text_column].astype(str).apply(len)
        self.data['word_count'] = self.data[text_column].astype(str).apply(lambda x: len(x.split()))
        
        print(f"Average length: {self.data['text_length'].mean():.2f} characters")
        print(f"Average word count: {self.data['word_count'].mean():.2f} words")
        print(f"Shortest text: {self.data['text_length'].min()} characters")
        print(f"Longest text: {self.data['text_length'].max()} characters")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        self.data['text_length'].hist(bins=50, ax=axes[0])
        axes[0].set_title('Text Length Distribution')
        axes[0].set_xlabel('Length (characters)')
        axes[0].set_ylabel('Frequency')
        
        # Word count distribution
        self.data['word_count'].hist(bins=50, ax=axes[1])
        axes[1].set_title('Word Count Distribution')
        axes[1].set_xlabel('Word Count')
        axes[1].set_ylabel('Frequency')
        
        # Correlation with label if available
        label_cols = [col for col in self.data.columns if 'label' in col]
        if label_cols:
            label_col = label_cols[0]  # Use first label column
            # Sample data for better visualization if dataset is large
            plot_sample = self.data.sample(min(1000, len(self.data)), random_state=42)
            sns.boxplot(x=plot_sample[label_col], y=plot_sample['text_length'], ax=axes[2])
            axes[2].set_title('Text Length by Label')
            axes[2].set_xlabel('Label')
            axes[2].set_ylabel('Text Length')
        
        plt.tight_layout()
        plt.show()
        
        # Generate word cloud
        print("\nGenerating word cloud...")
        all_text = ' '.join(self.data[text_column].astype(str).dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Text Content')
        plt.show()
    
    def analyze_correlations(self):
        """Analyze correlations between numerical features"""
        # Select numeric columns for correlation analysis
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            print("Calculating correlations...")
            plt.figure(figsize=(12, 8))
            correlation_matrix = self.data[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Feature Correlation Matrix')
            plt.show()
            
            # Print strong correlations
            strong_corr = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if abs(correlation_matrix.iloc[i, j]) > 0.5:
                        strong_corr.append((
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            correlation_matrix.iloc[i, j]
                        ))
            
            if strong_corr:
                print("\nStrong correlations (|r| > 0.5):")
                for corr in strong_corr:
                    print(f"{corr[0]} - {corr[1]}: {corr[2]:.3f}")
            else:
                print("\nNo strong correlations found (|r| > 0.5)")
        else:
            print("Not enough numeric columns for correlation analysis")
    
    def analyze_by_source(self, source_column='subreddit'):
        """Analyze data distribution by source (e.g., subreddit)"""
        if source_column not in self.data.columns:
            print(f"Column '{source_column}' not found in dataset")
            return
        
        print(f"\n=== Analysis by {source_column} ===")
        
        # Count by source
        source_counts = self.data[source_column].value_counts()
        print(f"Number of unique {source_column}s: {len(source_counts)}")
        print(f"Top 10 {source_column}s:")
        print(source_counts.head(10))
        
        # Plot distribution
        plt.figure(figsize=(12, 6))
        source_counts.head(20).plot(kind='bar')
        plt.title(f'Distribution by {source_column}')
        plt.xlabel(source_column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Check if we have label data to analyze source vs label relationship
        label_cols = [col for col in self.data.columns if 'label' in col]
        if label_cols:
            label_col = label_cols[0]
            # Get top sources
            top_sources = source_counts.head(5).index.tolist()
            source_label_data = self.data[self.data[source_column].isin(top_sources)]
            
            # Create cross-tabulation
            cross_tab = pd.crosstab(source_label_data[source_column], source_label_data[label_col])
            cross_tab.plot(kind='bar', figsize=(12, 6))
            plt.title(f'Label Distribution by {source_column}')
            plt.xlabel(source_column)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.legend(title=label_col)
            plt.tight_layout()
            plt.show()
    
    def sample_data_examples(self, n_samples=5, dataset_name="combined"):
        """Show sample data examples"""
        if dataset_name == "train":
            data = self.train_data
        elif dataset_name == "validation":
            data = self.validation_data
        elif dataset_name == "test":
            data = self.test_data
        else:
            data = self.data
            
        print(f"\nSample of {n_samples} examples from {dataset_name} dataset:")
        print("=" * 80)
        
        # Find text and label columns
        text_cols = [col for col in data.columns if 'text' in col.lower() or 'title' in col.lower()]
        label_cols = [col for col in data.columns if 'label' in col]
        
        for idx, row in data.head(n_samples).iterrows():
            print(f"Example {idx + 1}:")
            for label_col in label_cols:
                if label_col in row and pd.notna(row[label_col]):
                    print(f"{label_col}: {row[label_col]}")
            for text_col in text_cols:
                if text_col in row and pd.notna(row[text_col]):
                    text = str(row[text_col])
                    print(f"{text_col}: {text[:200]}..." if len(text) > 200 else f"{text_col}: {text}")
            print("-" * 80)

# Main execution function
def main():
    analyzer = FakedditAnalyzer()
    train_path = "multimodal_train.tsv"
    validation_path = "multimodal_validate.tsv"
    test_path = "multimodal_test_public.tsv"
    
    try:
        data = analyzer.load_data(train_path, validation_path, test_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the TSV files are in the current directory or provide the correct paths.")
        return
    print("\n" + "="*60)
    print("COMPREHENSIVE FAKEDDIT DATASET ANALYSIS")
    print("="*60)
    
    # Basic analysis for each dataset
    for dataset in ["train", "validation", "test", "combined"]:
        analyzer.basic_analysis(dataset)
    
    # Visualize label distribution
    analyzer.visualize_label_distribution()
    
    # Text analysis
    analyzer.text_analysis()
    
    # Correlation analysis
    analyzer.analyze_correlations()
    
    # Show sample examples
    analyzer.sample_data_examples(3, "train")
    analyzer.sample_data_examples(3, "test")
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()