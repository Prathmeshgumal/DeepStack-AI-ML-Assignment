import os
import re
import json
import pickle
import spacy
from typing import List, Set, Dict
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from mistralai import Mistral

load_dotenv()

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = "open-mistral-7b"
DATA_DIR = "data"
CHARACTER_MAP_FILE = "story_characters.json"
FAISS_INDEX_DIR = "faiss_index"
BM25_FILE = "bm25_retriever.pkl"

# Initialize Spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading SpaCy...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def extract_characters_hybrid(text, story_title, client):
    """
    Extracts ALL characters from the full story text.
    Combines SpaCy (Speed) + Mistral (Intellect).
    """
    characters = set()

    # 1. SpaCy (Fast)
    doc = nlp(text[:100000])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            characters.add(ent.text.strip())

    # 2. Mistral (Smart) - Only run on the first 4k chars to identify main cast
    if client:
        try:
            messages = [
                {"role": "user", "content": f"""
                Identify the distinct character names in this story.
                Return ONLY a JSON list of strings. Example: ["John", "Mary"].

                STORY: {story_title}
                TEXT: {text[:40000]}
                """}
            ]
            resp = client.chat.complete(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            llm_chars = json.loads(content)

            # Handle list or dict return
            if isinstance(llm_chars, list): characters.update(llm_chars)
            elif isinstance(llm_chars, dict):
                for val in llm_chars.values():
                    if isinstance(val, list): characters.update(val)
        except Exception as e:
            print(f"⚠️ Extraction skipped for {story_title}: {e}")

    # Clean duplicates and short noise
    return list({c for c in characters if len(c) > 2})

def identify_chars_in_chunk(chunk_text: str, global_chars: List[str]) -> List[str]:
    """Checks which global characters appear in this specific chunk."""
    present = []
    chunk_lower = chunk_text.lower()
    for char in global_chars:
        if char.lower() in chunk_lower:
            present.append(char)
    return present

def run_ingestion():
    print("🚀 Starting Ingestion Process...")
    
    if not MISTRAL_API_KEY:
        print("❌ Error: MISTRAL_API_KEY not found in .env file.")
        return

    client = Mistral(api_key=MISTRAL_API_KEY)

    print("📂 Loading documents...")
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: '{DATA_DIR}' directory not found.")
        return

    dir_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    raw_docs = dir_loader.load()

    if not raw_docs:
        print("⚠️ No documents found.")
        return

    global_character_map = {}
    processed_docs = []

    print(f"🕵️  Extracting characters for {len(raw_docs)} stories...")

    for doc in raw_docs:
        full_text = doc.page_content
        lines = full_text.split("\n")
        title = lines[0].strip() if lines else "Unknown"
        body = clean_text(full_text)

        # Extract once per story
        story_chars = extract_characters_hybrid(body, title, client)
        global_character_map[title] = story_chars
        print(f"   -> {title}: Found {len(story_chars)} characters.")

        # Update doc metadata
        doc.metadata["storyTitle"] = title
        doc.page_content = body
        processed_docs.append(doc)

    # Save the Map
    print(f"💾 Saving character map to {CHARACTER_MAP_FILE}...")
    with open(CHARACTER_MAP_FILE, "w") as f:
        json.dump(global_character_map, f, indent=4)

    print("✂️  Splitting and enriching chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_chunks = text_splitter.split_documents(processed_docs)

    # Enrich chunks
    for chunk in final_chunks:
        title = chunk.metadata.get("storyTitle")
        master_list = global_character_map.get(title, [])
        chunk_chars = identify_chars_in_chunk(chunk.page_content, master_list)
        chunk.metadata["chars_in_chunk"] = chunk_chars

    print("🧠 Creating Vector Index (FAISS)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(final_chunks, embedding_model)
    vector_db.save_local(FAISS_INDEX_DIR)

    print("🔤 Creating BM25 Index...")
    bm25_retriever = BM25Retriever.from_documents(final_chunks)
    bm25_retriever.k = 5
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25_retriever, f)

    print("✅ Ingestion Complete! Database ready.")