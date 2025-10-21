"""
Gallery Genie (G.G) Inc. - Simple Tool Selector
A minimal, single-file tool selection system
"""

import sys

from art import text2art  # pip install art
from ascii_magic import AsciiArt  # pip install ascii-magic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOOLS = [
    {
        'name': 'Calculator',
        'keywords': 'calculate math add subtract multiply divide numbers arithmetic',
        'emoji': '🧮'
    },
    {
        'name': 'Web Search',
        'keywords': 'search google find information internet lookup browse web',
        'emoji': '🔍'
    },
    {
        'name': 'Database Query',
        'keywords': 'database query sql data records retrieve fetch db',
        'emoji': '💾'
    },
    {
        'name': 'Image Processor',
        'keywords': 'image photo picture edit resize crop filter visual',
        'emoji': '🎨'
    },
    {
        'name': 'File Organizer',
        'keywords': 'file folder organize sort move rename directory files',
        'emoji': '📁'
    }
]


def find_best_tool(user_input: str, tools: list) -> tuple:
    """
    Match user input to best tool using simple TF-IDF.
    Returns: (tool_dict, confidence_score)
    """
    if not user_input.strip():
        return None, 0.0

    # Prepare tool descriptions (keywords)
    tool_texts = [tool['keywords'] for tool in tools]

    # Add user input
    all_texts = tool_texts + [user_input.lower()]

    # Calculate similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_texts)

    # User query is the last vector
    query_vector = vectors[-1]
    tool_vectors = vectors[:-1]

    # Get similarity scores
    scores = cosine_similarity(query_vector, tool_vectors).flatten()

    # Find best match
    best_idx = scores.argmax()
    best_score = scores[best_idx]

    return tools[best_idx], best_score


def show_menu(tools: list) -> dict:
    """Show numbered menu, user types number."""
    print("\n📋 Available Tools:")
    print("─" * 40)

    for i, tool in enumerate(tools, 1):
        print(f"  {i}. {tool['emoji']} {tool['name']}")

    print("─" * 40)

    while True:
        try:
            choice = input("\nEnter tool number (or 'q' to quit): ").strip()

            if choice.lower() == 'q':
                return None

            choice_num = int(choice)
            if 1 <= choice_num <= len(tools):
                return tools[choice_num - 1]
            else:
                print("❌ Invalid number. Try again.")
        except ValueError:
            print("❌ Please enter a number.")


def welcome():
    # Big title
    print(text2art("Welcome to Gallery Genie (G.G) Inc", "cybermedium"))
    print()

    # Intern's apology
    print("Umm.... Hello I am an intern at G.G, and this is my first day so... I am sorry in advance")
    print()

    # Genie image
    try:
        my_art = AsciiArt.from_image('Genie.jpg')
        print(my_art.to_ascii(width_ratio=3.7, columns=60))
    except FileNotFoundError:
        print("🧞 [Genie image not found - place Genie.jpg in same folder] 🧞")
    print()

    # Get name
    name = input("I am G, what's your name: ").strip() or "Friend"
    print()

    # Greeting
    print(text2art(f'Hello {name}', 'rounded'))

    return name


def main():
    """Run the Gallery Genie tool selector."""

    # Welcome screen
    user_name = welcome()

    # Main loop
    while True:
        print(f"\n🎉 Welcome  {user_name}! You just tell me in your natural language and i will get you a tool.\n")
        # Smart NLP matching
        print("\n What do you need help with?")
        user_query = input("Describe your task: ").strip()

        if not user_query:
            print(" Please describe your task!")
            continue

        tool, confidence = find_best_tool(user_query, TOOLS)

        if confidence > 0.2:  # Threshold
            print(f"\n Did you mean?? {tool['emoji']} {tool['name']}")

            confirm = input("\nAm I right??? (y/n): ").strip().lower()
            if confirm == 'y':
                sys.exit("\n Aww.... :(")
        else:
            print("\n Hmm, I'm not sure what you need it is my first day after all lets try..")
            show_menu(TOOLS)
            sys.exit("\n Aww.... :(")



if __name__ == "__main__":
    main()
