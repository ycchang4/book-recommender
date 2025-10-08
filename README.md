# Book Recommender System

A semantic book recommendation system that uses natural language processing and machine learning to provide personalized book recommendations based on user preferences and book descriptions.

## Features

- Semantic search for books using vector embeddings
- Emotion-based book categorization
- Interactive Gradio dashboard for easy book discovery
- Data exploration and analysis tools
- Text classification for book categorization

## Project Structure

- `gradio-dashboard.py`: Interactive web interface for book recommendations
- `text-classification.ipynb`: Notebook for training and evaluating text classification models
- `vector-search.ipynb`: Implementation of semantic search using vector embeddings
- `data-exploration.ipynb`: Data analysis and visualization tools
- `books_with_emotions.csv`: Dataset with emotion-based book categorizations
- `books_with_categories.csv`: Dataset with category-based book classifications
- `books_cleaned.csv`: Cleaned and processed book dataset

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/book-recommender.git
cd book-recommender
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in the required environment variables

## Usage

### Running the Gradio Dashboard

To start the interactive book recommendation dashboard:
```bash
python gradio-dashboard.py
```

### Data Exploration

Open the Jupyter notebooks to explore the data and models:
- `data-exploration.ipynb` for data analysis
- `text-classification.ipynb` for model training
- `vector-search.ipynb` for semantic search implementation

## Data

The project uses several datasets sourced from Kaggle:
- Original dataset: [Kaggle Book Dataset](https://www.kaggle.com/datasets/your-dataset-link)


Note: The original dataset needs to be downloaded from Kaggle and processed using the provided notebooks to generate the cleaned and categorized versions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
