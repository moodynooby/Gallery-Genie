import json
import os

# Force CPU-only before torch initializes CUDA backends (reduces dGPU heat)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import random
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
from ascii_magic import AsciiArt
from sentence_transformers import SentenceTransformer, util
import torch
import threading

if hasattr(torch, "set_num_threads"):
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(1)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'your-secret-key-here'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

TOOLS = [
    {
        "name": "ascii-art",
        "title": "ASCII Art",
        "description": "Convert images to ASCII text art with monospaced characters. Creates retro terminal-style artwork with grayscale text representation. Perfect for vintage computer aesthetics and text-based visuals.",
        "icon": '<i class="bi bi-code-square"></i>'
    },
    {
        "name": "aesthetic-blur",
        "title": "Aesthetic Blur",
        "description": "Apply Gaussian blur for soft focus dreamy effect. Creates smooth blurred background with gentle bokeh. `Great `for romantic portraits, calming visuals, and reducing sharp details.",
        "icon": '<i class="bi bi-droplet-half"></i>'
    },
    {
        "name": "vintage-sepia",
        "title": "Vintage Sepia",
        "description": "Add warm sepia tones for old photograph look. Creates nostalgic vintage film aesthetic with brownish warm colors. Perfect for antique appearance, historical feel, and retro photography style.",
        "icon": '<i class="bi bi-camera-reels"></i>'
    },
    {
        "name": "invert-colors",
        "title": "Evil Self",
        "description": "Invert all colors to create negative image with high contrast. Makes dark areas bright and vice versa. Ideal for cyberpunk neon aesthetic, x-ray effect, and surreal artistic look.",
        "icon": '<i class="bi bi-palette"></i>'
    },
    {
        "name": "pixelate-8bit",
        "title": "8-bit Pixelate",
        "description": "Create chunky pixelated mosaic with retro 8-bit video game style. Reduces image to low resolution blocky pixels. Perfect for pixel art, retro gaming aesthetic, and minimalist geometric look.",
        "icon": '<i class="bi bi-grid-3x3"></i>'
    },
    {
        "name": "pencil-sketch",
        "title": "Pencil Sketch",
        "description": "Transform photo into hand-drawn pencil sketch with artistic edges. Creates black and white drawing effect with sketchy lines. Great for artistic portraits, illustration style, and charcoal drawing look.",
        "icon": '<i class="bi bi-pencil"></i>'
    },
]

_model = None
_tool_embeddings = None
_tool_embedding_lock = threading.Lock()

GENIE_RESPONSES = {
    "greeting": [
        "Greetings, creative soul! I'm Gallery Genie, your magical image transformation companion! ✨",
        "Welcome to my mystical gallery! I'm Gallery Genie, here to grant your artistic wishes! 🧞",
        "Hey there, art lover! Gallery Genie at your service, ready to make some magic! 🎨",
    ],
    "upload_prompt": [
        "Upload an image and let me work my magic! What transformation do you have in mind?",
        "Drop your image here and tell me your vision - I'll make it reality!",
        "Ready to create something amazing? Upload an image and describe your dream effect!",
    ],
    "processing": [
        "Ah, excellent choice! Let me weave some magic into your image... ✨",
        "Wonderful! Casting my artistic spell on your image right now... 🪄",
        "Perfect! Watch as I transform your image with a touch of magic... 🌟",
    ],
    "success": [
        "Voilà! Your masterpiece is ready! What do you think? 🎨",
        "Ta-da! I've worked my magic! Your transformed image awaits! ✨",
        "Behold! Your image has been transformed by the power of Gallery Genie! 🌟",
    ],
    "suggestions": [
        "May I suggest trying {} for a stunning effect?",
        "How about {} to really make your image pop?",
        "I think {} would look absolutely magical on this!",
    ],
    "tips": {
        "ascii-art": "Pro tip: ASCII art works best with high-contrast images and portraits!",
        "aesthetic-blur": "Try this on backgrounds for that dreamy bokeh effect!",
        "vintage-sepia": "Perfect for giving your photos that nostalgic, timeless feel!",
        "invert-colors": "This creates stunning cyberpunk vibes - especially great for night scenes!",
        "pixelate-8bit": "Retro gaming aesthetic activated! Works best on colorful images!",
        "pencil-sketch": "Turn any photo into art! Works wonderfully on well-lit portraits!",
    }
}

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def get_tool_embeddings():
    """Cache tool description embeddings to avoid re-encoding each request."""
    global _tool_embeddings
    if _tool_embeddings is None:
        with _tool_embedding_lock:
            if _tool_embeddings is None:
                model = get_model()
                tool_descriptions = [t["description"] for t in TOOLS]
                with torch.no_grad():
                    _tool_embeddings = model.encode(
                        tool_descriptions,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                    )
    return _tool_embeddings


# Preload the model in a background thread so it's warmed before the first query.
# This keeps startup responsive while loading the heavy SentenceTransformer.
def _preload_model_background():
    try:
        print("Preloading sentence-transformers model in background...")
        get_model()
        print("Model preload complete.")
    except Exception as e:
        print("Model preload failed:", e)

# Start preload in daemon thread at module import/startup.
threading.Thread(target=_preload_model_background, daemon=True).start()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def output_path_from(input_filename: str, tool_slug: str) -> str:
    root, ext = os.path.splitext(input_filename)
    ext = ext if ext else ".png"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{root}_{tool_slug}_{timestamp}{ext}"

def do_ascii_art(image_path: str, output_path: str):
    art = AsciiArt.from_image(image_path)
    art.to_image_file(output_path)

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

def find_best_tool(user_input: str) -> dict | None:
    text = (user_input or "").strip()
    if not text:
        return None
    
    try:
        model = get_model()
        tool_embeddings = get_tool_embeddings()
        with torch.no_grad():
            query_embedding = model.encode(
                text,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        similarities = util.cos_sim(query_embedding, tool_embeddings)[0]
        best_idx = int(torch.argmax(similarities).item())
        best_score = float(similarities[best_idx].item())

        result = {
            "tool": TOOLS[best_idx],
            "confidence": best_score
        }

        return result
    except Exception as e:
        return None

@app.route('/')
def index():
    return render_template('index.html', tools=TOOLS)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({
            "response": random.choice(GENIE_RESPONSES["upload_prompt"]),
            "type": "prompt"
        })
    
    result = find_best_tool(message)
    
    if result and result['confidence'] > 0.35:
        tool = result['tool']
        response = f"Got it! Based on your request, I'll use {tool['title']} - {tool['description'].split('.')[0]}. " \
                  f"Would you like to proceed with this transformation?"
        return jsonify({
            "response": response,
            "type": "suggestion",
            "tool": tool,
            "confidence": result['confidence']
        })
    else:
        return jsonify({
            "response": f"I couldn't quite understand which effect you'd like to apply. Here are the available tools to choose from:",
            "type": "menu",
            "tools": TOOLS
        })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and file.filename and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        return jsonify({
            "success": True,
            "filename": unique_filename,
            "message": random.choice(GENIE_RESPONSES["upload_prompt"])
        })
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/process', methods=['POST'])
def process_image():
    data = request.json or {}
    filename = data.get('filename')
    tool_name = data.get('tool')
    
    if not filename or not tool_name:
        return jsonify({"error": "Missing parameters"}), 400
    
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(input_path):
        return jsonify({"error": "File not found"}), 404
    
    output_filename = output_path_from(filename, tool_name)
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    
    try:
        if tool_name == "ascii-art":
            do_ascii_art(input_path, output_path)
        else:
            img = cv2.imread(input_path)
            if img is None:
                return jsonify({"error": "Unable to read image"}), 400
            
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
                return jsonify({"error": "Unknown tool"}), 400
            
            cv2.imwrite(output_path, out)
        
        success_msg = random.choice(GENIE_RESPONSES["success"])
        tip = GENIE_RESPONSES["tips"].get(tool_name, "Looking great!")
        
        return jsonify({
            "success": True,
            "output_file": f"/processed/{output_filename}",
            "message": f"{success_msg}\n\n💡 {tip}"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/api/genie-greet', methods=['GET'])
def genie_greet():
    return jsonify({
        "message": random.choice(GENIE_RESPONSES["greeting"])
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
