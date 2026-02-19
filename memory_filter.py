import json
import chromadb

# 1. Setup the local database
client = chromadb.Client()
collection = client.create_collection(name="persona_rag")

# 2. Load your JSON file
with open('Cybel_Example_Chats.json', 'r') as f:
    data = json.load(f)

# 3. Ingest and "Tag" the data (The Discrimination Part)
for i, entry in enumerate(data['example_dialogues']):
    # Add Character logic
    collection.add(
        documents=[entry['char']],
        metadatas=[{"speaker": "char"}], # Tagging as Character
        ids=[f"char_{i}"]
    )
    # Add User logic
    collection.add(
        documents=[entry['user']],
        metadatas=[{"speaker": "user"}], # Tagging as User
        ids=[f"user_{i}"]
    )

# 4. Function to "Discriminate" during search
def ask_rag(query, focus="char"):
    # This filter tells the database to ONLY look at specific tags
    results = collection.query(
        query_texts=[query],
        n_results=2,
        where={"speaker": focus} # Focus can be "char" or "user"
    )
    return results['documents']

# --- TESTING THE SYSTEM ---
print("Character Memories:", ask_rag("What is your purpose?", focus="char"))
print("User Memories:", ask_rag("What did the human ask about control?", focus="user"))