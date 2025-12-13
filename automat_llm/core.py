import os
import json
import logging
from   langchain_huggingface            import HuggingFaceEmbeddings, HuggingFacePipeline
from   langchain_community.vectorstores import FAISS
from   langchain_core.prompts.chat      import ChatPromptTemplate
from   langchain.chains.retrieval       import create_retrieval_chain
from   langchain.docstore.document      import Document
from   langchain.schema.document        import Document
from   langchain_groq                   import ChatGroq

current_dir = os.getcwd()

def load_json_as_documents(client, directory):
    documents = []
    collection = client.collections.use("MyCollection") #TBA: use(f"{user_id}_Collection")
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

    with collection.batch.fixed_size(batch_size=200) as batch:
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

    failed_objects = collection.batch.failed_objects
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
    from weaviate.classes.config import Configure
    try:
        print("Step 1: Creating embeddings and indexing documents...")
        client.connect()
        if client.collections.get("Embeddings") is not None:
            embeddings   = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            Embeddings_W = client.collections.get("Embeddings")
        else:
            embeddings   = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            Embeddings_W = client.collections.create(
                name="Embeddings",
                vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
                generative_config=Configure.Generative.cohere()
            )

        # Create FAISS vector store
        vector_store = FAISS.from_documents(documents, embeddings)
        print("Embeddings and vector store created.")

        # Upload embeddings directly
        text_list = [doc.page_content for doc in documents]
        vectors   = embeddings.embed_documents(text_list)
        for i, vec in enumerate(vectors):
            uuid = Embeddings_W.data.insert(
                properties={
                    "user_id": user_id,
                    "text":    text_list[i],
                },
                vector=vec
            )
            print(f"Embedding Doc {i+1}/{len(vectors)} with UUID: {uuid}")

        print("Step 2: Setting up the language model...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a snarky but helpful assistant."),
            ("human",  "{input}\n\nUse this context if helpful:\n{context}")
        ])

        llm = ChatGroq(
            temperature=0.5,
            model="openai/gpt-oss-20b",
            max_tokens=5000,
            api_key=os.environ.get("GROQ_API_KEY")
        )

        #llm = HuggingFacePipeline.from_model_id(
        #    model_id="xai-org/grok-1",
        #    task="text-generation",
        #    pipeline_kwargs={"max_length": 8000, "num_return_sequences": 1}
        #)

        llm_chain = prompt | llm
        print("Language model set up.")

        rag_chain = create_retrieval_chain(vector_store.as_retriever(), llm_chain)
        print("RetrievalQA chain created.")
        return rag_chain

    except Exception as e:
        print(f"Error creating the RetrievalQA chain: {e}")
        return None

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
        return next(
            item['response']
            for item in personality_data['example_dialogue']
            if item['user'].lower() == "just do what i say, you stupid robot!"
        )

    try:
        result = rag_chain.invoke({"input": user_input})
        #print(f'Contents of Result: {result}')

        # Safely get the answer text
        answer = result.get("answer") or result.get("result")
        if hasattr(answer, "content"):
            response = answer.content
        else:
            response = str(answer)

        # Log the interaction
        logging.info(f"User: {user_input}")
        logging.info(f"Bot: {response}")
        logging.info("Retrieved Memories:")

        # Safely iterate retrieved docs
        docs = result.get("source_documents") or result.get("context", [])
        for doc in docs:
            page_info = doc.metadata.get("page") or doc.metadata.get("source") or "unknown page"
            logging.info(f"- [{page_info}] {doc.page_content[:200]}")  # first 200 chars

        logging.info("")
        return response

    except Exception as e:
        print(f"Error generating response: {e}")
        logging.error(f"Error generating response: {e}", exc_info=True)
        return "I'm sorry, I couldn't process your request."

        