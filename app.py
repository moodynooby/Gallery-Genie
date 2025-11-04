import json
import os
import platform
import secrets
import subprocess
import sys
from datetime import datetime

import cv2
from art import text2art
from ascii_magic import AsciiArt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# App paths
APP_DIR = os.path.join(os.path.expanduser("~"), ".gallery_genie")
SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")

# Tool definitions
TOOLS = [
    {'name': 'ascii-art',     'title': 'ASCII Art',       'keywords': 'ascii text art terminal retro monospaced grayscale', 'emoji': ''},
    {'name': 'aesthetic-blur','title': 'Aesthetic Blur',  'keywords': 'gaussian blur soft focus dreamy smooth denoise',     'emoji': ''},
    {'name': 'vintage-sepia', 'title': 'Vintage Sepia',   'keywords': 'sepia warm tone old photo vintage film look',        'emoji': ''},
    {'name': 'invert-colors', 'title': 'Invert Colors',   'keywords': 'negative invert cyberpunk high contrast neon',       'emoji': ''},
    {'name': 'pixelate-8bit', 'title': '8-bit Pixelate',  'keywords': 'pixelate mosaic low-res retro 8-bit chunky pixels',  'emoji': ''},
    {'name': 'pencil-sketch', 'title': 'Pencil Sketch',   'keywords': 'sketch edges grayscale dodge blur hand-drawn',       'emoji': ''},
]

# ---------------- settings ----------------
def ensure_app_dir():
    os.makedirs(APP_DIR, exist_ok=True)  # create config dir [web:59]

def load_settings():
    ensure_app_dir()
    if not os.path.exists(SETTINGS_PATH):
        return {}
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_settings(data: dict):
    ensure_app_dir()
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ---------------- auth-like flow (local) ----------------
def setup():
    """First-run onboarding: ask name, create local token, store defaults."""
    settings = load_settings()
    print(text2art("Gallery Genie", "cybermedium"))
    print("First time here. Let’s set things up.")
    name = input("Name (Enter for 'Friend'): ").strip() or "Friend"
    token = secrets.token_urlsafe(24)  # local remember-me token
    settings.update({
        "name": name,
        "token": token,
        "created_at": datetime.now().isoformat() + "Z",
        "last_path": settings.get("last_path", ""),
        "login_count": 0,
    })
    save_settings(settings)
    print(f"Setup complete. Welcome, {name}.")
    return settings

def login():
    """Check token presence; if missing, guide to setup."""
    settings = load_settings()
    if not settings.get("token"):
        print("No profile found. Run `python app.py setup` to get started.")
        if input("Run setup now? (Y/n): ").strip().lower() in ("", "y", "yes"):
            settings = setup()
        else:
            sys.exit("Exiting.")
    settings["login_count"] = int(settings.get("login_count", 0)) + 1
    settings["last_login"] = datetime.now().isoformat() + "Z"
    save_settings(settings)
    print(f"Welcome back, {settings.get('name','Friend')}.")
    return settings

# ---------------- utility ----------------
def output_path_from(input_path: str, tool_slug: str) -> str:
    base = os.path.basename(input_path)
    root, ext = os.path.splitext(base)
    ext = ext if ext else ".png"
    return f"{root}.{tool_slug}{ext}"  # descriptive naming [web:101]

def open_with_default_viewer(path: str):
    if not os.path.exists(path):
        return
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.call(["open", path])
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.call(["xdg-open", path])
    except Exception:
        pass

def find_best_tool(user_input: str, tools: list) -> tuple:
    text = (user_input or "").strip().lower()
    if not text:
        return None, 0.0
    corpus = [t['keywords'] for t in tools] + [text]
    vecs = TfidfVectorizer().fit_transform(corpus)
    scores = cosine_similarity(vecs[-1], vecs[:-1]).flatten()
    if scores.size == 0:
        return None, 0.0
    i = int(scores.argmax())
    return tools[i], float(scores[i])

# ---------------- effects ----------------
def do_ascii_art(image_path: str):
    out_path = output_path_from(image_path, "ascii-art")
    art = AsciiArt.from_image(image_path)
    art.to_image_file(out_path)
    print(f"Saved: {out_path}")

def do_aesthetic_blur(img):
    return cv2.GaussianBlur(img, (9, 9), 0)

def do_vintage_sepia(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_BONE)

def do_invert_colors(img):
    return cv2.bitwise_not(img)

def do_pixelate_8bit(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def do_pencil_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    denom = cv2.bitwise_not(blur)
    sketch = cv2.divide(gray, denom, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

# ---------------- prompts ----------------
def prompt_path(settings: dict) -> str:
    saved = settings.get("last_path")
    if saved and os.path.exists(saved):
        ans = input(f"Use last image path? [{saved}] (Y/n): ").strip().lower() or "y"
        if ans.startswith("y"):
            return saved
    while True:
        path = input("Enter image path: ").strip().strip('"').strip("'")
        if not path:
            print("Path is empty.")
            continue
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        settings["last_path"] = path
        save_settings(settings)
        return path

def show_menu(tools: list) -> dict:
    print("\nTools")
    print("─" * 40)
    for i, t in enumerate(tools, 1):
        print(f"  {i}. {t['title']}")
    print("─" * 40)
    while True:
        c = input("Pick a number (or 'q' to quit): ").strip().lower()
        if c == 'q':
            sys.exit("Goodbye.")
        try:
            n = int(c)
            if 1 <= n <= len(tools):
                return tools[n - 1]
            print("Invalid number.")
        except ValueError:
            print("Enter a number.")

def process_image(tool_name: str, image_path: str):
    if tool_name == "ascii-art":
        try:
            do_ascii_art(image_path)
            open_with_default_viewer(output_path_from(image_path, "ascii-art"))
        except Exception as e:
            print(f"Error: {e}")
        finally:
            print("Tip: Save multiple versions to compare styles side by side.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Couldn’t read that as an image. Try another file.")
        return

    try:
        if tool_name == "aesthetic-blur":
            out = do_aesthetic_blur(img)
        elif tool_name == "vintage-sepia":
            out = do_vintage_sepia(img)
        elif tool_name == "invert-colors":
            out = do_invert_colors(img)
        elif tool_name == "pixelate-8bit":
            out = do_pixelate_8bit(img)
        elif tool_name == "pencil-sketch":
            out = do_pencil_sketch(img)
        else:
            print("Unknown tool.")
            return

        out_path = output_path_from(image_path, tool_name)
        cv2.imwrite(out_path, out)
        print(f"Saved: {out_path}")

        h, w = out.shape[:2]
        mood = {
            "aesthetic-blur": "soft, low-detail look",
            "vintage-sepia": "warm, aged tones",
            "invert-colors": "high-contrast negative",
            "pixelate-8bit": "chunky retro blocks",
            "pencil-sketch": "hand-drawn lines",
        }.get(tool_name, "new style")
        print(f"Output size: {w}×{h}. Style: {mood}.")
        print("Tip: Run another tool on the saved file to layer effects.")
        open_with_default_viewer(out_path)
    except Exception as e:
        print(f"Error: {e}")

def suggest_tool(query: str):
    tool, score = find_best_tool(query, TOOLS)
    if tool and score > 0.20:
        print(f"Suggested tool: {tool['title']} (match {score:.2f})")
        if input("Use this? (y/n): ").strip().lower() in ("y", "yes"):
            return tool
    print("Let’s choose from the menu.")
    return show_menu(TOOLS)

# ---------------- entry ----------------
def main():
    args = [a.lower() for a in sys.argv[1:]]
    if "setup" in args:
        setup()
        return
    settings = login()  # remember-me token + name [web:59]

    print(text2art("Gallery Genie", "cybermedium"))
    print("Tell me what you want to do, and I’ll suggest a tool.")
    if settings.get("login_count", 1) == 1:
        print("Tip: Run `python app.py setup` anytime to change your name or reset the token.")

    while True:
        print("\nDescribe your task (or 'q' to exit).")
        query = input("> ").strip()
        if query.lower() in ("q", "quit", "exit"):
            sys.exit("Goodbye.")
        if not query:
            print("A short description helps pick the right tool.")
            continue
        tool = suggest_tool(query)
        path = prompt_path(settings)
        process_image(tool['name'], path)

if __name__ == "__main__":
    main()
