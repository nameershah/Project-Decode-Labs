# ============================================================
# Project 1: Rule-Based AI Chatbot
# Decode Labs AI Internship | Batch 2026
# ============================================================

KNOWLEDGE_BASE = {
    "hello": "Hey! I'm ARIA, your AI assistant. How can I help you today?",
    "hi": "Hey! I'm ARIA, your AI assistant. How can I help you today?",
    "hey": "Hey! I'm ARIA, your AI assistant. How can I help you today?",
    "how are you": "Running at full capacity! All systems operational. How about you?",
    "what is ai": "AI is the simulation of human intelligence by machines — enabling them to learn, reason, and make decisions.",
    "what is machine learning": "Machine learning is a subset of AI where systems learn from data and improve over time without being explicitly programmed.",
    "what is deep learning": "Deep learning uses neural networks with many layers to recognize patterns in large amounts of data like images, audio, and text.",
    "who are you": "I'm ARIA — Automated Rule-based Intelligence Assistant. Built on pure logic, no hallucinations guaranteed.",
    "what can you do": "I can answer questions about AI, machine learning, and deep learning. Try asking me something!",
    "help": "You can ask me about AI, machine learning, or deep learning. Type 'quit' or 'exit' to end the session.",
    "thanks": "Happy to help! Let me know if you have more questions.",
    "thank you": "Happy to help! Let me know if you have more questions.",
    "bye": "Goodbye! Session terminated. See you next time.",
    "good morning": "Good morning! Ready to talk AI?",
    "good night": "Good night! ARIA going into standby mode.",
}

FALLBACK = "I don't understand that yet. Try asking about AI, machine learning, or type 'help' for options."
EXIT_COMMANDS = {"quit", "exit", "bye", "goodbye"}


def get_response(user_input: str) -> str:
    clean = user_input.lower().strip()

    if clean in EXIT_COMMANDS:
        return None  # Signal to exit

    return KNOWLEDGE_BASE.get(clean, FALLBACK)


def run_chatbot():
    print("=" * 55)
    print("  ARIA - Automated Rule-based Intelligence Assistant")
    print("  Decode Labs | Project 1 | Batch 2026")
    print("  Type 'exit' or 'quit' to end the session.")
    print("=" * 55)
    print()

    while True:
        try:
            raw_input_text = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nARIA: Session interrupted. Goodbye!")
            break

        response = get_response(raw_input_text)

        if response is None:
            print("ARIA: Goodbye! Session terminated.")
            break

        print(f"ARIA: {response}")
        print()


if __name__ == "__main__":
    run_chatbot()
