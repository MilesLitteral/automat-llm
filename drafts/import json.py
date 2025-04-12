# ...existing code...

class Chatbot:
    def __init__(self):
        # ...existing code...
        self.memory_file = os.path.join(CLEANED_JSON_DIR, "supercellmemory.json")
        self.memory = self._load_memory()

    def _load_memory(self):
        """Load memory from the supercellmemory JSON file."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_memory(self):
        """Save memory to the supercellmemory JSON file."""
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=4)

    def get_response(self, user_input):
        """Generate a response from the chatbot and store the interaction in memory."""
        try:
            result = self.qa_chain({"query": user_input})
            response = result["result"]

            # Store the interaction in memory
            self.memory.append({"user": user_input, "bot": response})
            self._save_memory()

            return response
        except Exception as e:
            return f"{self.char_name}: I'm sorry, I couldn't process your request."
