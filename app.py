import sys
import numpy as np
from art import text2art
from ascii_magic import AsciiArt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os
import platform
import subprocess

TOOLS = [
    {'name': 'ASCII-fy', 'keywords': 'ascii art retro monochrome blocky pixelated', 'emoji': '•ᴗ•'},
    {'name': 'get-aesthetic-photo', 'keywords': 'blur smooth soft gaussian aesthetic photo', 'emoji': '🌫️'},
    {'name': 'pencil-sketch', 'keywords': 'grayscale invert blur divide pencil drawing cartoon', 'emoji': '🖍️'},
    {'name': 'get-your-old-timey-look', 'keywords': 'vintage old sepia tone warm photo', 'emoji': '📻'},
    {'name': 'Get-your-evil-form', 'keywords': 'invert colors negative cyberpunk glitch contrast', 'emoji': '🌈'},
    {'name': '8-bit-game-look', 'keywords': 'pixelated mosaic zoom compression digital art', 'emoji': '🟩'},
    {'name': 'Cartoonify', 'keywords': 'blur edge color cartoon comic simplify', 'emoji': '🧩'},
]

def find_best_tool(user_input: str, tools: list) -> tuple:
    if not user_input.strip():
        return None, 0.0
    tool_texts = [tool['keywords'] for tool in tools]
    all_texts = tool_texts + [user_input.lower()]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_texts)
    query_vector = vectors[-1]
    tool_vectors = vectors[:-1]
    scores = cosine_similarity(query_vector, tool_vectors).flatten()
    best_idx = scores.argmax()
    best_score = scores[best_idx]
    return tools[best_idx], best_score

def show_menu(tools: list) -> dict:
    print("\n Available Tools:")
    print("─" * 40)
    for i, tool in enumerate(tools, 1):
        print(f"  {i}. {tool['emoji']} {tool['name']}")
    print("─" * 40)
    while True:
        try:
            choice = input("\nEnter tool number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                sys.exit("Bye..")
            choice_num = int(choice)
            if 1 <= choice_num <= len(tools):
                return tools[choice_num - 1]
            else:
                print("❌ Invalid number. Try again.")
        except ValueError:
            print("❌ Please enter a number.")

def open_image_with_default_viewer(image_path):
    if platform.system() == 'Darwin':
        subprocess.call(['open', image_path])
    elif platform.system() == 'Windows':
        os.startfile(image_path)
    else:
        subprocess.call(['xdg-open', image_path])

def all_tools(tool_name):
    while True:
        image_path = input("Enter the path to your image: ").strip()
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image '{image_path}'. Try again.")
                continue

            if tool_name == "get-aesthetic-photo":
                output_image = cv2.GaussianBlur(image, (9, 9), 0)
            elif tool_name == "pencil-sketch":
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                output_image = cv2.filter2D(image, -1, kernel)
            elif tool_name == "get-your-old-timey-look":
                output_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                output_image = cv2.applyColorMap(output_image, cv2.COLORMAP_BONE)
            elif tool_name == "Get-your-evil-form":
                output_image = cv2.bitwise_not(image)
            elif tool_name == "8-bit-game-look":
                small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR)
                output_image = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            elif tool_name == "Cartoonify":
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                inv = 255 - gray
                blur = cv2.GaussianBlur(inv, (21, 21), 0)
                output_image = cv2.divide(gray, 255 - blur, scale=256)
                output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
            elif tool_name == "ASCII-fy":
                ascii_art = AsciiArt.from_image(image_path)
                ascii_art.to_image_file('output_ascii.png', width_ratio=3.7, columns=60)
                print("ASCII art saved as output_ascii.png")
                open_image_with_default_viewer('output_ascii.png')
                break
            else:
                print("Unknown tool.")
                break

            output_path = f"output_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, output_image)
            print(f"Processed image saved as '{output_path}'")
            open_image_with_default_viewer(output_path)
            break
        except Exception as e:
            print(f"Error: {e}")

def welcome():
    print(text2art("Welcome to Gallery Genie (G.G) Inc", "cybermedium"))
    print()
    print("Umm.... Hello I am an intern at G.G, and this is my first day so... I am sorry in advance")
    print()
    try:
        my_art = AsciiArt.from_image('Genie.jpg')
        print(my_art.to_ascii(width_ratio=3.7, columns=60))
    except FileNotFoundError:
        print("🧞 [Genie image not found - place Genie.jpg in same folder] 🧞")
    print()
    name = input("I am G, what's your name: ").strip() or "Friend"
    print()
    print(text2art(f'Hello {name}', 'rounded'))
    return name

def main():
    user_name = welcome()
    while True:
        print(f"\n🎉 Welcome {user_name}! You just tell me in your natural language and I will get you a tool.\n")
        user_query = input("Describe your task: ").strip()
        if not user_query:
            print("Please describe your task!")
            continue
        tool, confidence = find_best_tool(user_query, TOOLS)
        if confidence > 0.2:
            print(f"\nDid you mean?? {tool['emoji']} {tool['name']}")
            confirm = input("\nAm I right??? (y/n): ").strip().lower()
            if confirm == 'y':
                all_tools(tool['name'])
        else:
            print("\nHmm, I'm not sure what you need it is my first day after all let's try..")
            selected_tool = show_menu(TOOLS)
            if selected_tool is None:
                print("Goodbye!")
                break
            all_tools(selected_tool['name'])

if __name__ == "__main__":
    main()
