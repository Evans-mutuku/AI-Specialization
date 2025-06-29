class RuleBasedChatbot:
    def __init__(self):
        self.rules = {
            "hi": "Hello! How can I assist you today?",
            "hello": "Hi there! What can I help you with?",
            "how are you": "I'm just a program, but I'm functioning as expected!",
            "bye": "Goodbye! Have a great day!",
            "what is your name": "I am a rule-based chatbot built with Python.",
            "help": "Sure, I can help. Ask me about greetings, my name, or say goodbye."
        }

    def respond(self, message):
        message = message.lower().strip()
        for pattern in self.rules:
            if pattern in message:
                return self.rules[pattern]
        return "I'm not sure how to respond to that. Try saying 'help'."

# Example usage:
chatbot = RuleBasedChatbot()
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Goodbye!")
        break
    print("Chatbot:", chatbot.respond(user_input))
