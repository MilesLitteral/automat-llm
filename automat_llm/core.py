import os
import json
import logging
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.chains            import create_retrieval_chain
from langchain.docstore.document      import Document
from langchain.schema  import Document

current_dir = os.getcwd()

def load_json_as_documents(client, questions, directory):
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

def init_interactions():
    # Load or initialize user interactions
    try:
        user_interactions_file = f"{current_dir}/user_interactions.json"
        with open(user_interactions_file, 'r', encoding='utf-8') as f:
            user_interactions = json.load(f)
            return user_interactions
    except FileNotFoundError:
        user_interactions = {"users": {}}
        with open(user_interactions_file, 'w', encoding='utf-8') as f:
            json.dump(user_interactions, f, indent=4)


def load_personality_file():
    # Load the personality from robot_personality.json
    try:
        personality_file = f"{current_dir}/robot_personality.json"
        with open(personality_file, 'r', encoding='utf-8') as f:
            personality_data = json.load(f)
            return personality_data
    except FileNotFoundError:
        print(f"Personality file not found at {personality_file}. Please create robot_personality.json.")
        logging.error(f"Personality file not found at {personality_file}.")
        exit(1)


def create_rag_chain(client, user_id, documents):
    from   weaviate.classes.config import Configure
    try:
        print("Step 1: Creating embeddings and indexing documents...")
        if(client.collections.get("Embeddings") != None):
            embeddings   = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # TBA: #= client.collections.get("Embeddings")
            Embeddings_W = client.collections.get("Embeddings")
        else:
            embeddings   = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            Embeddings_W = client.collections.create(
                name="Embeddings",
                vectorizer_config=Configure.Vectorizer.text2vec_weaviate(), # Configure the Weaviate Embeddings integration
                generative_config=Configure.Generative.cohere()             # Configure the Cohere generative AI integration
            )
        vector_store = FAISS.from_documents(documents, embeddings)
        print("Embeddings and vector store created.")
        uuid = Embeddings_W.data.insert(
            properties={
                "user_id":    user_id,
                "embeddings": embeddings,
            },
            vector=vector_store[0]
        )

        print(f"Embeddings uploaded with ID: {uuid}")  # the return value is the object's UUID
        print("Step 2: Setting up the language model...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a snarky but helpful assistant."),
            ("human",  "{input}\n\nUse this context if helpful:\n{context}")
        ])

        llm = HuggingFacePipeline.from_model_id(
            model_id="distilgpt2", #these models fix the error GPT-2 encounters but you need ~14Gb free to use this one: "tiiuae/falcon-7b-instruct", ~100Gb: #"mistralai/Mistral-7B-Instruct-v0.1", ~ 90Gb: #TheBloke/dolphin-2.7-mixtral-8x7b-GGUF 
            task="text-generation",
            pipeline_kwargs={"max_length": 100, "num_return_sequences": 1}
        )

        llm_chain = prompt | llm  # This is now a Runnable
        print("Language model set up.")
        rag_chain = create_retrieval_chain(vector_store.as_retriever(), llm_chain)
        print("RetrievalQA chain created.")
        return rag_chain
    except Exception as e:
        print(f"Error creating the RetrievalQA chain: {e}")
        exit()

def create_rag_chain_falcon(client, user_id, documents):
    try:
        print("Step 1: Creating embeddings and indexing documents...")
        embeddings   = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents, embeddings)
        print("Embeddings and vector store created.")
        Embeddings_W = client.collections.get("JeopardyQuestion")
        uuid = Embeddings_W.data.insert(
            properties={
                "user_id":    user_id,
                "embeddings": embeddings,
            },
            vector=vector_store[0]
        )

        print(f"Embeddings uploaded with ID: {uuid}")  # the return value is the object's UUID
        print("Step 2: Setting up the language model...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a snarky but helpful assistant."),
            ("human", "{input}\n\nUse this context if helpful:\n{context}")
        ])

        llm = HuggingFacePipeline.from_model_id(
            model_id="tiiuae/falcon-7b-instruct",
            task="text-generation",
            pipeline_kwargs={"max_length": 100, "num_return_sequences": 1}
        )

        llm_chain = prompt | llm  # This is now a Runnable
        print("Language model set up.")
        rag_chain = create_retrieval_chain(vector_store.as_retriever(), llm_chain)
        print("RetrievalQA chain created.")
        return rag_chain
    except Exception as e:
        print(f"Error creating the RetrievalQA chain: {e}")
        exit()

def create_rag_chain_mistral(client, user_id, documents):
    try:
        print("Step 1: Creating embeddings and indexing documents...")
        embeddings   = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents, embeddings)
        print("Embeddings and vector store created.")
        Embeddings_W = client.collections.get("JeopardyQuestion")
        uuid = Embeddings_W.data.insert(
            properties={
                "user_id":    user_id,
                "embeddings": embeddings,
            },
            vector=vector_store[0]
        )

        print(f"Embeddings uploaded with ID: {uuid}")  # the return value is the object's UUID
        print("Step 2: Setting up the language model...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a snarky but helpful assistant."),
            ("human", "{input}\n\nUse this context if helpful:\n{context}")
        ])

        llm = HuggingFacePipeline.from_model_id(
            model_id="mistralai/Mistral-7B-v0.1",
            task="text-generation",
            pipeline_kwargs={"max_length": 100, "num_return_sequences": 1}
        )

        llm_chain = prompt | llm  # This is now a Runnable
        print("Language model set up.")
        rag_chain = create_retrieval_chain(vector_store.as_retriever(), llm_chain)
        print("RetrievalQA chain created.")
        return rag_chain
    except Exception as e:
        print(f"Error creating the RetrievalQA chain: {e}")
        exit()


def create_rag_chain_mixtral(client, user_id, documents):
    try:
        print("Step 1: Creating embeddings and indexing documents...")
        embeddings   = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents, embeddings)
        print("Embeddings and vector store created.")
        Embeddings_W = client.collections.get("JeopardyQuestion")
        uuid = Embeddings_W.data.insert(
            properties={
                "user_id":    user_id,
                "embeddings": embeddings,
            },
            vector=vector_store[0]
        )

        print(f"Embeddings uploaded with ID: {uuid}")  # the return value is the object's UUID
        print("Step 2: Setting up the language model...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a snarky but helpful assistant."),
            ("human", "{input}\n\nUse this context if helpful:\n{context}")
        ])

        llm = HuggingFacePipeline.from_model_id(
            model_id="TheBloke/dolphin-2.7-mixtral-8x7b-GGUF",
            task="text-generation",
            pipeline_kwargs={"max_length": 100, "num_return_sequences": 1}
        )

        llm_chain = prompt | llm  # This is now a Runnable
        print("Language model set up.")
        rag_chain = create_retrieval_chain(vector_store.as_retriever(), llm_chain)
        print("RetrievalQA chain created.")
        return rag_chain
    except Exception as e:
        print(f"Error creating the RetrievalQA chain: {e}")
        exit()

# Function to update user interactions
def update_user_interactions(user_id, user_interactions_file, user_interactions, is_rude=False, apologized=False):
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
def generate_response(user_id, user_interactions, user_input, rude_keywords, personality_data, rag_chain):
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
        result = rag_chain.invoke({"input": user_input})
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
    