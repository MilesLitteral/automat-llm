import os
import logging
import argparse

import weaviate
from   weaviate.classes.init import Auth
from   weaviate.classes.config import Configure

from dia.model import Dia
from playsound import playsound
from automat_llm.core   import load_json_as_documents, load_personality_file, init_interactions, generate_response, create_rag_chain
from automat_llm.config import load_config, save_config, update_config

config      = load_config()
current_dir = os.getcwd()

# Best practice: store your credentials in environment variables
weaviate_url     = "enb5w7lzsiggptazuakxug.c0.us-east1.gcp.weaviate.cloud" #os.environ["WEAVIATE_URL"]
weaviate_api_key = "5aFrft85NhDXkz4GqS2OYJGv5XhlHu6GsOAo"                  #os.environ["WEAVIATE_API_KEY"]
user_id          = config["default_user"]  # Automat-User-Id, In the future this will be in a config the user can set.
                                           # It is made for a single-user system; can be modified for multi-user

# Ensure directories exist
directory = os.path.abspath(f'{current_dir}/Input_JSON/')
if not os.path.exists(directory):
    print(f"Cleaned JSON directory not found at {directory}. Creating Input_JSON folder")
    os.mkdir(f'{current_dir}/Input_JSON')
    print("UnhandledException: please load Cleaned_, or Cybel Memory JSON into Input_JSON")
    exit()

# Set up logging to save chatbot interactions
logging.basicConfig(
    filename=f'{current_dir}/Logs/chatbot_logs.txt', #r'./Logs/chatbot_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,                                    # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(weaviate_api_key),             # Replace with your Weaviate Cloud key
)

personality_data  = load_personality_file()
user_interactions = init_interactions()
documents         = load_json_as_documents(directory)

if not documents:
    print("No documents extracted from JSON files. Please check the file contents.")
    exit()

print(f"Loaded {len(documents)} documents for RAG.")

# Extract personality details
char_name     = personality_data['char_name']

# Rudeness detection keywords
rude_keywords = ["stupid", "idiot", "shut up", "useless", "dumb"]
rag_chain     = create_rag_chain(client, user_id, documents)

# Chatbot loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of boolean flag with argparse.")
    parser.add_argument("--set", metavar="KEY=VALUE", help="Set a configuration value (e.g., user.name=Alice)")
    parser.add_argument("--use_dia", action="store_true", help="Enable Dia audio model use and output") # Boolean flag
    args = parser.parse_args()

    if args.use_dia:
        dia_model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")
        print("Audio mode is ON")
    else:
        print("Audio mode is OFF")

    if args.set:
        if "=" not in args.set:
            parser.error("Argument to --set must be in key=value format.")
        key, value = args.set.split("=", 1)

        # Type inference (primitive)
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)

        config = update_config(config, key, value)
        save_config(config)
        print(f"Updated {key} to {value}")

    print(f"Current config:\n{config}")
    print(f"\n{char_name} is ready! Type your message (or 'quit' to exit).")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            response = generate_response(user_id, user_interactions, user_input, rude_keywords, personality_data, rag_chain)
            if(args.use_dia):
                output = dia_model.generate(f"[S1] {response}", use_torch_compile=True, verbose=True)
                dia_model.save_audio(f"response.mp3", output)
                playsound("response.mp3")
            print(f"{char_name}: {response}")
        except Exception as e:
            print(f"Error in chatbot loop: {e}")