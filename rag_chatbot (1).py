import json
import os
import glob
import logging
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from transformers import pipeline
from pydantic import BaseModel, ConfigDict
from langchain_core.prompts.base import BasePromptTemplate

from json_loader import load_json_files
from retrieval_qa import create_retrieval_qa_chain

class MyModel(BaseModel):
    prompt: BasePromptTemplate

    model_config = ConfigDict(arbitrary_types_allowed=True)


# Creates a single directory
if not os.path.exists(r'./Logs'):
    os.mkdir("Logs")
if not os.path.exists(r'./Output'):
    os.mkdir("Output")

# Ensure directories exist
directory = os.path.abspath(r'C:\Users\EnocEscalona\Documents\Supercell_AI_Dev\Data\Cleaned_JSONs')
if not os.path.exists(directory):
    print(f"Cleaned JSON directory not found at {directory}. Please check the path.")
    exit()

# Set up logging to save chatbot interactions
logging.basicConfig(
    filename=r'C:\Users\EnocEscalona\Documents\Supercell_AI_Dev\Logs\chatbot_logs.txt', #r'./Logs/chatbot_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Directory containing your cleaned JSON files (on the laptop)
directory = r'C:\Users\EnocEscalona\Documents\Supercell_AI_Dev\Data\Cleaned_JSONs' #r'./Output'

# Step 1: Load the cleaned JSON files
documents = load_json_files(directory)

if not documents:
    print("No documents extracted from JSON files. Please check the file contents.")
    exit()

print(f"Loaded {len(documents)} documents for RAG.")

# Step 2: Create embeddings and index the documents
try:
    print("Step 2: Creating embeddings and indexing documents...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    print("Embeddings and vector store created.")
except Exception as e:
    print(f"Error creating embeddings or vector store: {e}")
    exit()

# Step 3: Set up the language model for generation
try:
    print("Step 3: Setting up the language model...")
    llm = HuggingFacePipeline.from_model_id(
        model_id="distilgpt2",
        task="text-generation",
        pipeline_kwargs={"max_length": 100, "num_return_sequences": 1},
    )
    print("Language model set up.")
except Exception as e:
    print(f"Error setting up the language model: {e}")
    exit()

# Ensure personality file exists
personality_file = os.path.abspath(r'C:\Users\EnocEscalona\Documents\Supercell_AI_Dev\Personality\src\robot_personality.json')
if not os.path.exists(personality_file):
    print(f"Personality file not found at {personality_file}. Please create robot_personality.json.")
    logging.error(f"Personality file not found at {personality_file}.")
    exit(1)

# Load the personality from robot_personality.json
try:
    with open(personality_file, 'r', encoding='utf-8') as f:
        personality_data = json.load(f)
except FileNotFoundError:
    print(f"Personality file not found at {personality_file}. Please create robot_personality.json.")
    logging.error(f"Personality file not found at {personality_file}.")
    exit(1)

# Ensure user interactions file exists
user_interactions_file = os.path.abspath(r'C:\Users\EnocEscalona\Documents\Supercell_AI_Dev\Personality\src\user_interactions.json')
if not os.path.exists(user_interactions_file):
    user_interactions = {"users": {}}
    with open(user_interactions_file, 'w', encoding='utf-8') as f:
        json.dump(user_interactions, f, indent=4)

# Load or initialize user interactions
try:
    with open(user_interactions_file, 'r', encoding='utf-8') as f:
        user_interactions = json.load(f)
except FileNotFoundError:
    user_interactions = {"users": {}}
    with open(user_interactions_file, 'w', encoding='utf-8') as f:
        json.dump(user_interactions, f, indent=4)

# Extract personality details
char_name = personality_data['char_name']
tars_prompt = personality_data['char_persona'] + "\n"

# Rudeness detection keywords
rude_keywords = ["stupid", "idiot", "shut up", "useless", "dumb"]

# Step 4: Create a RetrievalQA chain with the fused personality
try:
    rag_chain = create_retrieval_qa_chain(llm, vector_store, tars_prompt, char_name)
    print("RetrievalQA chain created.")
except Exception as e:
    print(f"Error creating the RetrievalQA chain: {e}")
    exit()

# Function to update user interactions
def update_user_interactions(user_id, is_rude=False, apologized=False):
    if user_id not in user_interactions["users"]:
        user_interactions["users"][user_id] = {"rudeness_score": 0, "requires_apology": False}
    
    user_data = user_interactions["users"][user_id]
    if is_rude:
        user_data["rudeness_score"] += 1
        if user_data["rudeness_score"] >= 2:  # Threshold for requiring an apology
            user_data["requires_apology"] = True
    elif apologized:
        user_data["rudeness_score"] = 0
        user_data["requires_apology"] = False
    
    with open(user_interactions_file, 'w', encoding='utf-8') as f:
        json.dump(user_interactions, f, indent=4)

# Step 5: Function to generate a response
def generate_response(user_id, user_input):
    """
    Generate a response using the RetrievalQA chain with the fused personality.

    Parameters:
    - user_id (str): Identifier for the user.
    - user_input (str): The user's input text.

    Returns:
    - str: The AI-generated response.
    """
    input_lower = user_input.lower()
    
    # Check if user requires an apology
    user_data = user_interactions["users"].get(user_id, {"rudeness_score": 0, "requires_apology": False})
    if user_data.get("requires_apology", False):
        if "sorry" in input_lower or "apologize" in input_lower:
            update_user_interactions(user_id, apologized=True)
            return next(item['response'] for item in personality_data['example_dialogue'] if item['user'].lower() == "i’m sorry for being rude.")
        return "I’m waiting for an apology, sweetie. I don’t respond to rudeness without respect."

    # Check for rudeness
    is_rude = any(keyword in input_lower for keyword in rude_keywords)
    if is_rude:
        update_user_interactions(user_id, is_rude=True)
        return next(item['response'] for item in personality_data['example_dialogue'] if item['user'].lower() == "just do what i say, you stupid robot!")

    try:
        result = rag_chain({"query": user_input})
        response = result['result']
        # Log the interaction
        logging.info(f"User: {user_input}")
        logging.info(f"Bot: {response}")
        logging.info("Retrieved Memories:")
        for doc in result['source_documents']:
            logging.info(f"- {doc.page_content}")
        logging.info("")
        return response
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I'm sorry, I couldn't process your request."

# Step 6: Chatbot loop (for standalone testing)
if __name__ == "__main__":
    user_id = "default_user"  # For a single-user system; can be modified for multi-user
    print(f"\n{char_name} is ready! Type your message (or 'quit' to exit).")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            response = generate_response(user_id, user_input)
            print(f"{char_name}: {response}")
        except Exception as e:
            print(f"Error in chatbot loop: {e}")