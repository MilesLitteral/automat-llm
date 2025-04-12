class Chatbot:
    def __init__(self):
        self.responses = {
            "hello": "Hi there! How can I help you?",
            "how are you": "I'm just a bot, but I'm doing great! How about you?",
            "bye": "Goodbye! Have a great day!"
        }

    def get_response(self, user_input):
        user_input = user_input.lower()
        return self.responses.get(user_input, "I'm sorry, I don't understand that.")
