def chatbot_loop(char_name, generate_response):
    """Run the chatbot loop for user interaction."""
    print(f"\n{char_name} is ready! Type your message (or 'quit' to exit).")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            response = generate_response(user_input)
            print(f"{char_name}: {response}")
        except Exception as e:
            print(f"Error in chatbot loop: {e}")
