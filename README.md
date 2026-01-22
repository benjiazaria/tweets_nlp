# Tweet Sentiment/Value Prediction with BERT

This project uses BERT embeddings combined with demographic features to predict sentiment/value ratings for tweets.

## Project Structure

```
├── data/                      # Raw and processed data
│   ├── id_tweet.tsv           # Original tweets (598 tweets)
│   ├── train_data.csv         # Training data (gender + tweet + value)
│   └── test_data.csv          # Test data
│
├── final/                     # Final scripts and results
│   ├── create_data.ipynb      # Data preparation pipeline
│   ├── main.ipynb             # Main BERT prediction model
│   ├── train_results.csv      # Training predictions
│   ├── test_results.csv       # Test predictions
│   └── tweets_statistics.csv  # Tweet-level statistics
```

## Data Pipeline

1. **Raw Data**: Tweet texts (`id_tweet.tsv`) + survey responses (not included in repo)
2. **Processing**: `create_data.ipynb` unpivots, cleans, and joins the data
3. **Output**: `train_data.csv` and `test_data.csv`

## Model Architecture

The main model (`main.ipynb`) uses:
- **BERT** (`bert-base-uncased`) for tweet embeddings (768 dimensions)
- **PCA** to reduce to 40 dimensions
- **One-hot encoded demographics** (Q21-Q31: age, gender, education, employment, opinions)
- **Fully Connected Network**: 100 → 50 → 30 → 1 (with dropout)

## Features

| Feature | Description |
|---------|-------------|
| Q21 | Age group (18-25, 26-30, 31-40, 41-50, 51-60, 60+) |
| Q22 | Gender (Male/Female) |
| Q23 | Education level |
| Q24 | Employment status |
| Q25-Q31 | Opinion questions (Agree/Disagree scale) |

## Results

- Predictions range: 0.18 to 0.95 (on 0-1 scale)
- ~80% of predictions within 0.3 of actual value

## Requirements

```
torch
transformers
tensorflow
scikit-learn
pandas
numpy
```

## Usage

1. Run `create_data.ipynb` to prepare the data
2. Run `main.ipynb` to train and evaluate the model


