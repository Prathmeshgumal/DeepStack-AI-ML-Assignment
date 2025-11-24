import os
import json
import pickle
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from mistralai import Mistral

# Load environment variables
load_dotenv()

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = "open-mistral-7b"
CHARACTER_MAP_FILE = "story_characters.json"
FAISS_INDEX_DIR = "faiss_index"
BM25_FILE = "bm25_retriever.pkl"

def clean_and_validate_json(data):
    """
    Enforces schema but tries to fix common LLM mistakes.
    """
    if not isinstance(data, dict):
        return {"error": "Invalid JSON format received from LLM"}

    structured_data = {
        "name": data.get("name", "Unknown"),
        "storyTitle": data.get("storyTitle", "Unknown"),
        "summary": data.get("summary", "Summary not available."),
        "relations": [],
        "characterType": data.get("characterType", "Unknown")
    }

    # Validate Relations with Fuzzy Key Matching
    raw_relations = data.get("relations", [])
    if isinstance(raw_relations, list):
        for item in raw_relations:
            if isinstance(item, dict) and "name" in item:
                rel_value = item.get("relation") or item.get("relationship") or "Connected"
                structured_data["relations"].append({
                    "name": item["name"],
                    "relation": rel_value
                })

    return structured_data

def generate_character_info(character_name, verbose=False):
    if not MISTRAL_API_KEY:
        return {"error": "MISTRAL_API_KEY not configured."}
    
    client = Mistral(api_key=MISTRAL_API_KEY)

    # 1. Load Map
    if os.path.exists(CHARACTER_MAP_FILE):
        with open(CHARACTER_MAP_FILE, "r") as f:
            global_map = json.load(f)
    else:
        return {"error": "Character map not found. Please run compute-embeddings first."}

    # 2. Smart Alias Detection
    character_exists = False
    detected_story = None
    canonical_name = character_name
    known_aliases = []

    for title, chars in global_map.items():
        matches = [c for c in chars if character_name.lower() in c.lower()]
        if matches:
            character_exists = True
            detected_story = title
            known_aliases = matches
            # Heuristic: longest name is often the full name
            canonical_name = max(matches, key=len)
            break

    if not character_exists:
        return {"error": f"Character '{character_name}' not found in processed stories."}

    # 3. Retrieve (Hybrid)
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.load_local(FAISS_INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
        dense_retriever = vector_db.as_retriever(search_kwargs={"k": 6})
        
        with open(BM25_FILE, "rb") as f:
            sparse_retriever = pickle.load(f)
            sparse_retriever.k = 6

        dense_docs = dense_retriever.invoke(canonical_name)
        sparse_docs = sparse_retriever.invoke(canonical_name)
        
        # If the searched name is different from canonical, search original query too
        if canonical_name != character_name:
             sparse_docs.extend(sparse_retriever.invoke(character_name))
    except Exception as e:
        return {"error": f"Failed to load indexes: {str(e)}"}

    # Deduplicate & Filter
    all_docs = {d.page_content: d for d in sparse_docs + dense_docs}.values()
    filtered_docs = [d for d in all_docs if d.metadata.get("storyTitle") == detected_story]

    if not filtered_docs:
        filtered_docs = list(all_docs)

    # 4. Context Preparation
    context_text = ""
    for d in filtered_docs:
        local_chars = d.metadata.get("chars_in_chunk", [])
        context_text += f"""
        --- Snippet from "{d.metadata.get('storyTitle')}" ---
        [Characters in this scene: {local_chars}]
        Text: {d.page_content}
        """

    # 5. Generation
    messages = [
        {"role": "system", "content": "You are a JSON extractor. Output strictly valid JSON matching the provided schema."},
        {"role": "user", "content": f"""
        Task: Extract character details for "{canonical_name}" based ONLY on the text below.

        Aliases: The text may refer to them as {known_aliases} or with titles (Mr., Mrs.). Treat them as the same person.

        STORY TITLE: "{detected_story}"

        TEXT CONTEXT:
        {context_text}

        ### INSTRUCTIONS FOR RELATIONS ###
        - Extract clear relationships.
        - **IMPORTANT:** If no formal relationship (like "Father") is stated, **INFER** the relationship based on interactions.
          - Example: If they argue -> "Adversary" or "Conflict"
          - Example: If they work together -> "Colleague"
          - Example: If they talk -> "Acquaintance"

        ### REQUIRED OUTPUT FORMAT ###
        Example:
        {{
            "name": "Jon Snow",
            "storyTitle": "A Song of Ice and Fire",
            "summary": "Jon Snow is a brave leader...",
            "relations": [
                {{ "name": "Arya Stark", "relation": "Sister" }},
                {{ "name": "Cersei Lannister", "relation": "Enemy" }}
            ],
            "characterType": "Protagonist"
        }}

        ### YOUR OUTPUT FOR "{canonical_name}" ###
        """}
    ]

    try:
        resp = client.chat.complete(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"}
        )
        raw_json_str = resp.choices[0].message.content
        raw_json = json.loads(raw_json_str)

        if verbose:
            print(f"\n=== DEBUG: RAW LLM OUTPUT ===\n{json.dumps(raw_json, indent=2)}\n=============================\n")

        final_output = clean_and_validate_json(raw_json)
        
        # Ensure consistency in top-level fields
        final_output["name"] = canonical_name
        final_output["storyTitle"] = detected_story
        
        return final_output

    except Exception as e:
        return {"error": f"LLM Generation Failed: {e}"}