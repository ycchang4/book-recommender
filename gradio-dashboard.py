import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(
    documents,
    embedding=embedding_model
)

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    book_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(book_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category][:final_top_k]
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprise":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="angry", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(
            query: str,
            category: str,
            tone: str
    ):

        recommendations = retrieve_semantic_recommendations(query, category, tone)

        results = []

        for _, row in recommendations.iterrows():
            description = row["description"]
            truncated_desc_split = description.split()
            truncated_description = " ".join(truncated_desc_split[:30]) + "..."

            authors_split = row["authors"].split(";")
            if len(authors_split) == 2:
                author_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                author_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                author_str = row["authors"]

            caption = f"**{row['title']}**\n*by {author_str}*\n\n{truncated_description}"
            results.append((row["large_thumbnail"], caption))
        return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprise", "Angry", "Suspenseful", "Sad"]

# CSS for styling
custom_css = """
#main-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: bold;
    margin-bottom: 0.5em;
}

#subtitle {
    text-align: center;
    color: #666;
    font-size: 1.2em;
    margin-bottom: 2em;
}

.input-row {
    margin-bottom: 1.5em;
}

#search-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
    font-weight: bold;
    font-size: 1.1em;
    padding: 12px 24px;
    border-radius: 8px;
}

#search-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.gallery-item {
    border-radius: 8px;
    overflow: hidden;
    transition: transform 0.2s;
}

.gallery-item:hover {
    transform: scale(1.05);
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as dashboard:
    
    gr.Markdown("<h1 id='main-title'>📚 Discover Your Next Book</h1>")
    gr.Markdown("<p id='subtitle'>Find the perfect book using AI-powered semantic search</p>")
    
    with gr.Row(elem_classes="input-row"):
        with gr.Column(scale=3):
            user_query = gr.Textbox(
                label="📝 What kind of book are you looking for?",
                placeholder="e.g., A heartwarming story about friendship and second chances...",
                lines=2
            )
        
    with gr.Row(elem_classes="input-row"):
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="📖 Category",
                value="All",
                interactive=True
            )
        with gr.Column(scale=1):
            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="🎭 Emotional Tone",
                value="All",
                interactive=True
            )
        with gr.Column(scale=1):
            submit_button = gr.Button(
                "🔍 Find Books",
                variant="primary",
                elem_id="search-button"
            )
    
    gr.Markdown("---")
    gr.Markdown("## 📚 Your Personalized Recommendations")
    
    output = gr.Gallery(
        label="",
        columns=4,
        rows=4,
        height="auto",
        object_fit="contain",
        show_label=False
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )
    
    # Add example searches
    gr.Examples(
        examples=[
            ["A story about forgiveness and redemption", "All", "All"],
            ["Mystery with unexpected twists", "Fiction", "Surprise"],
            ["Adventure in magical worlds", "Fantasy", "Happy"],
            ["Deep philosophical questions about life", "All", "All"],
        ],
        inputs=[user_query, category_dropdown, tone_dropdown],
    )

if __name__ == "__main__":
    dashboard.launch(share=False)