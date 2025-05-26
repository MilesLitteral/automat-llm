import json
import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from transformers import pipeline
from pydantic import BaseModel, ConfigDict
from langchain_core.prompts.base import BasePromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.docstore.document import Document

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure
#from dia.model import Dia

# Best practice: store your credentials in environment variables
weaviate_url = weaviate_url     = "enb5w7lzsiggptazuakxug.c0.us-east1.gcp.weaviate.cloud" #os.environ["WEAVIATE_URL"]
weaviate_api_key = "5aFrft85NhDXkz4GqS2OYJGv5XhlHu6GsOAo" #os.environ["WEAVIATE_API_KEY"]
#model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,                                    # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(weaviate_api_key),             # Replace with your Weaviate Cloud key
)

if(client.collections.get("Introduction") != None):
    questions = client.collections.get("Introduction")
else:
    questions = client.collections.create(
        name="Introduction",
        vectorizer_config=Configure.Vectorizer.text2vec_weaviate(), # Configure the Weaviate Embeddings integration
        generative_config=Configure.Generative.cohere()             # Configure the Cohere generative AI integration
    )


def load_json_files(directory):
    """Load and validate JSON files from a directory."""
    json_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
    documents = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                first_char = f.read(1)
                f.seek(0)
                if first_char == '[':
                    data = json.load(f)
                    documents.extend([Document(page=entry['text']) for entry in data if 'text' in entry])
                else:
                    for line in f:
                        entry = json.loads(line.strip())
                        if 'text' in entry:
                            documents.append(Document(page=entry['text']))
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return documents

def load_json_file(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
            documents.extend([Document(page_content=entry['text']) for entry in data if 'text' in entry])
        else:
            for line in f:
                entry = json.loads(line.strip())
                if 'text' in entry:
                    documents.append(Document(page_content=entry['text']))
    return documents

def load_json_as_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    raw_content = f.read()
                    # Optionally, reformat it as pretty-printed JSON
                    parsed = json.loads(raw_content)
                    pretty_json = json.dumps(parsed, indent=2)
                    documents.append(Document(page_content=pretty_json, metadata={"source": filename}))
                except Exception as e:
                    print(f"Skipping {filename} due to error: {e}")

    # Extract list of 'Entry' strings if the JSON is a list of dicts
    entries = [item['Entry'] for item in documents if 'Entry' in item]

    with questions.batch.fixed_size(batch_size=200) as batch:
        for d in entries:
            print(d)
            batch.add_object(
                {
                    "entry": d
                }
            )
            if batch.number_errors > 10:
                print("Batch import stopped due to excessive errors.")
                break

    failed_objects = questions.batch.failed_objects
    if failed_objects:
        print(f"Number of failed imports: {len(failed_objects)}")
        print(f"First failed object: {failed_objects[0]}")

    client.close()  # Free up resources
    return documents

class MyModel(BaseModel):
    prompt: BasePromptTemplate

    model_config = ConfigDict(arbitrary_types_allowed=True)

current_dir = os.getcwd()
# Creates a single directory
if not os.path.exists(r'./Logs'):
    os.mkdir("Logs")
if not os.path.exists(r'./Output'):
    os.mkdir("Output")

# Ensure directories exist
directory = os.path.abspath(f'{current_dir}/Input_JSON/')
if not os.path.exists(directory):
    print(f"Cleaned JSON directory not found at {directory}. Creating Input_JSON folder")
    os.mkdir(f'{current_dir}/Input_JSON')
    print("Exception, please load Cleaned_JSON with your json data")
    exit()

# Set up logging to save chatbot interactions
logging.basicConfig(
    filename=f'{current_dir}/chatbot_logs.txt', #r'./Logs/chatbot_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Directory containing your cleaned JSON files (on the laptop)
directory = f'{current_dir}/Input_JSON' #r'./Output'

# Step 1: Load the cleaned JSON files
documents = load_json_as_documents(directory) #f"{directory}/cleaned_SupercellAMemory0.json")

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
personality_file = f"{current_dir}/robot_personality.json"
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
user_interactions_file = f"{current_dir}/user_interactions.json"
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
    rag_chain = create_retrieval_chain(llm, vector_store.as_retriever())#, tars_prompt, char_name)
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
        response = result['result'] #f"[S1] {result['result']}"
        # Log the interaction
        logging.info(f"User: {user_input}")
        logging.info(f"Bot: {response}")
        logging.info("Retrieved Memories:")
        for doc in result['source_documents']:
            logging.info(f"- {doc.page}")
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
            #output = model.generate(f"[S1] {response}", use_torch_compile=True, verbose=True)
            #model.save_audio(f"response.mp3", output)
            print(f"{char_name}: {response}")
        except Exception as e:
            print(f"Error in chatbot loop: {e}")