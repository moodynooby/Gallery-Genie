import os
import random
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
from ascii_magic import AsciiArt

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
        "description": "Convert an image into ASCII text art with monospaced characters. Creates retro terminal-style artwork with grayscale text representation and text-only pixels, perfect for vintage computer aesthetics and text-based visuals.",
        "icon": '<i class="bi bi-code-square"></i>',
        "keywords": ["ascii", "text", "code", "terminal", "retro", "vintage", "computer"]
    },
    {
        "name": "aesthetic-blur",
        "title": "Aesthetic Blur",
        "description": "Apply a soft Gaussian blur filter for a dreamy out-of-focus look. Creates smooth blurred backgrounds with gentle bokeh and softened details, great for portraits, calming visuals, and hiding distractions.",
        "icon": '<i class="bi bi-droplet-half"></i>',
        "keywords": ["blur", "soft", "dreamy", "bokeh", "out of focus", "smooth", "calming"]
    },
    {
        "name": "vintage-sepia",
        "title": "Vintage Sepia",
        "description": "Add warm sepia tones for an old photograph look. Creates a nostalgic vintage film aesthetic with brownish warm colors, perfect for antique appearance, historical feeling, and retro photography style.",
        "icon": '<i class="bi bi-camera-reels"></i>',
        "keywords": ["sepia", "vintage", "old", "antique", "retro", "nostalgic", "brown", "warm"]
    },
    {
        "name": "invert-colors",
        "title": "Evil Self",
        "description": "Invert all colors to create a high-contrast negative image. Makes dark areas bright and bright areas dark, ideal for cyberpunk neon aesthetics, x-ray style effects, and surreal artistic looks.",
        "icon": '<i class="bi bi-palette"></i>',
        "keywords": ["invert", "negative", "evil", "cyberpunk", "neon", "x-ray", "opposite"]
    },
    {
        "name": "pixelate-8bit",
        "title": "8-bit Pixelate",
        "description": "Create a chunky pixelated mosaic with a retro 8-bit video game style. Reduces the image to low-resolution blocky pixels, perfect for pixel art, retro gaming aesthetics, and minimalist geometric looks.",
        "icon": '<i class="bi bi-grid-3x3"></i>',
        "keywords": ["pixel", "8bit", "8-bit", "retro", "game", "chunky", "blocky", "mosaic"]
    },
    {
        "name": "pencil-sketch",
        "title": "Pencil Sketch",
        "description": "Transform a photo into a hand-drawn pencil sketch with artistic edges. Creates a black and white drawing effect with sketchy lines and shading, great for artistic portraits, illustration styles, and charcoal drawing looks.",
        "icon": '<i class="bi bi-pencil"></i>',
        "keywords": ["sketch", "pencil", "drawing", "artistic", "hand drawn", "charcoal", "black white"]
    },
    {
        "name": "oil-painting",
        "title": "Oil Painting",
        "description": "Transform your image into a beautiful oil painting with rich textures and brush strokes. Creates an artistic painted look with smooth color blending and artistic brush effects, perfect for classic art styles.",
        "icon": '<i class="bi bi-paint-bucket"></i>',
        "keywords": ["oil", "painting", "paint", "artistic", "brush", "canvas", "art"]
    },
    {
        "name": "cartoon-anime",
        "title": "Cartoon Style",
        "description": "Convert your photo into a vibrant cartoon or anime style with bold colors and smooth edges. Creates a fun animated look with enhanced colors and simplified details, great for playful portraits and fun visuals.",
        "icon": '<i class="bi bi-palette-fill"></i>',
        "keywords": ["cartoon", "anime", "animated", "comic", "fun", "colorful", "bold"]
    },
    {
        "name": "black-white",
        "title": "Black & White",
        "description": "Convert your image to classic black and white photography. Creates timeless monochrome images with proper grayscale conversion, perfect for elegant portraits and dramatic compositions.",
        "icon": '<i class="bi bi-circle-half"></i>',
        "keywords": ["black", "white", "grayscale", "monochrome", "bw", "classic", "timeless"]
    },
    {
        "name": "emboss",
        "title": "Emboss",
        "description": "Apply an embossed 3D effect that makes your image look like it's carved in stone or metal. Creates a raised relief effect with depth and dimension, great for textural and artistic looks.",
        "icon": '<i class="bi bi-layers"></i>',
        "keywords": ["emboss", "3d", "relief", "texture", "depth", "carved", "raised"]
    },
    {
        "name": "edge-detection",
        "title": "Edge Detection",
        "description": "Extract and highlight the edges in your image for a technical line art look. Creates a high-contrast outline effect showing only the important edges and contours, perfect for technical drawings.",
        "icon": '<i class="bi bi-diagram-3"></i>',
        "keywords": ["edge", "outline", "contour", "line", "technical", "drawing", "detection"]
    },
    {
        "name": "watercolor",
        "title": "Watercolor",
        "description": "Transform your image into a beautiful watercolor painting with soft color bleeding and artistic washes. Creates a dreamy painted effect with soft edges and flowing colors, perfect for artistic portraits.",
        "icon": '<i class="bi bi-water"></i>',
        "keywords": ["watercolor", "water", "paint", "soft", "flowing", "artistic", "dreamy"]
    },
    {
        "name": "posterize",
        "title": "Posterize",
        "description": "Reduce the number of colors to create a bold posterized effect. Creates vibrant high-contrast images with limited color palettes, perfect for pop art styles and bold graphic designs.",
        "icon": '<i class="bi bi-grid-fill"></i>',
        "keywords": ["posterize", "poster", "pop art", "bold", "vibrant", "limited colors", "graphic"]
    },
    {
        "name": "neon-glow",
        "title": "Neon Glow",
        "description": "Add a vibrant neon glow effect to your image with bright colors and glowing edges. Creates a cyberpunk aesthetic with electric colors and glowing highlights, perfect for night scenes and futuristic looks.",
        "icon": '<i class="bi bi-lightning-fill"></i>',
        "keywords": ["neon", "glow", "cyberpunk", "electric", "bright", "futuristic", "night"]
    },
    {
        "name": "mirror-horizontal",
        "title": "Mirror Flip",
        "description": "Flip your image horizontally to create a mirror reflection effect. Creates a flipped version of your image, great for creating symmetrical compositions and artistic reflections.",
        "icon": '<i class="bi bi-arrow-left-right"></i>',
        "keywords": ["mirror", "flip", "horizontal", "reflect", "symmetry", "reverse", "mirrored"]
    },
    {
        "name": "color-boost",
        "title": "Color Boost",
        "description": "Enhance and boost the colors in your image for a vibrant, saturated look. Increases color intensity and saturation to make your image pop with vivid, eye-catching colors.",
        "icon": '<i class="bi bi-brightness-high-fill"></i>',
        "keywords": ["color", "boost", "saturate", "vibrant", "intense", "bright", "enhance"]
    },
]

GENIE_RESPONSES = {
    "greeting": [
        "✨ Greetings, mortal! I am Gallery Genie, master of image transformations! Upload an image and watch me work my magic!",
        "🧞‍♂️ Ah, a new visitor! I am the Gallery Genie, and I grant wishes for beautiful image effects. Show me your image!",
        "🌟 Welcome, seeker of visual wonders! I am Gallery Genie, and I shall transform your images with mystical powers!",
    ],
    "upload_prompt": [
        "✨ Magnificent! Your image has been received. Now, what transformation shall I perform? Describe your wish or choose from my magical effects!",
        "🧞‍♂️ Excellent! Your image is ready. Tell me your desire, or browse my collection of enchanting effects!",
        "🌟 Splendid! The image is mine to transform. What artistic vision shall I bring to life?",
    ],
    "processing": [
        "✨ *waves hands mystically* Let me weave my magic upon your image...",
        "🧞‍♂️ *rubs hands together* Working my genie powers... transforming as we speak!",
        "🌟 *sparkles appear* The transformation begins! Watch the magic unfold...",
    ],
    "success": [
        "✨ Behold! Your wish has been granted! The transformation is complete!",
        "🧞‍♂️ Ta-da! The magic is done! Your image has been transformed by my powers!",
        "🌟 Voilà! The enchantment is complete! Your image now bears my mystical touch!",
    ],
    "tips": {
        "ascii-art": "✨ Pro tip: This spell works best with images that have strong contrast!",
        "aesthetic-blur": "🧞‍♂️ This magic creates dreamy, ethereal backgrounds - perfect for portraits!",
        "vintage-sepia": "🌟 A classic enchantment that brings nostalgic warmth to any image!",
        "invert-colors": "✨ This creates a cyberpunk realm effect - especially powerful on night scenes!",
        "pixelate-8bit": "🧞‍♂️ Retro gaming magic! Perfect for colorful, vibrant images!",
        "pencil-sketch": "🌟 Transform portraits into artistic masterpieces with this spell!",
        "oil-painting": "✨ Best cast upon detailed images for maximum artistic impact!",
        "cartoon-anime": "🧞‍♂️ Fun and playful magic - perfect for portraits that need character!",
        "black-white": "🌟 Timeless elegance through monochrome enchantment!",
        "emboss": "✨ Creates depth and texture - great for artistic effects!",
        "edge-detection": "🌟 Technical magic that works best on clear, well-defined images!",
        "watercolor": "✨ Perfect for artistic portraits with soft, flowing colors!",
        "posterize": "🧞‍♂️ Bold pop art magic - creates vibrant, eye-catching results!",
        "neon-glow": "✨ Cyberpunk enchantment - most powerful on darker images!",
        "mirror-horizontal": "🌟 Symmetry magic - try it for artistic compositions!",
        "color-boost": "🧞‍♂️ Makes colors pop with vibrant intensity!",
    }
}

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

def do_oil_painting(img):
    # Oil painting effect using bilateral filter and color quantization
    filtered = cv2.bilateralFilter(img, 9, 75, 75)
    # Reduce colors for painting effect
    data = filtered.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(img.shape)

def do_cartoon_anime(img):
    # Cartoon effect: bilateral filter + edge enhancement
    filtered = cv2.bilateralFilter(img, 9, 300, 300)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(filtered, edges)

def do_black_white(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

def do_emboss(img):
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]], dtype=np.float32)
    embossed = cv2.filter2D(img, -1, kernel)
    embossed = cv2.cvtColor(embossed, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)

def do_edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def do_watercolor(img):
    # Watercolor effect: multiple bilateral filters + slight blur
    filtered = cv2.bilateralFilter(img, 9, 75, 75)
    filtered = cv2.bilateralFilter(filtered, 9, 75, 75)
    return cv2.GaussianBlur(filtered, (5, 5), 0)

def do_posterize(img):
    # Posterize: reduce color levels
    img_float = img.astype(np.float32)
    levels = 4
    posterized = np.round(img_float / 255.0 * (levels - 1)) / (levels - 1) * 255.0
    return posterized.astype(np.uint8)

def do_neon_glow(img):
    # Neon glow: edge detection + color enhancement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_HOT)
    # Blend with original
    return cv2.addWeighted(img, 0.7, edges_colored, 0.3, 0)

def do_mirror_horizontal(img):
    return cv2.flip(img, 1)

def do_color_boost(img):
    # Increase saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.5)  # Boost saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def fuzzy_match_score(text, target):
    """Simple fuzzy matching score based on substring and character similarity"""
    text = text.lower()
    target = target.lower()
    
    # Exact match
    if text == target:
        return 100
    
    # Contains match
    if text in target or target in text:
        return 80
    
    # Word-by-word matching
    text_words = set(text.split())
    target_words = set(target.split())
    if text_words & target_words:  # Intersection
        return 60
    
    # Character similarity (simple ratio)
    common_chars = sum(1 for c in text if c in target)
    if len(target) > 0:
        return int((common_chars / len(target)) * 40)
    return 0

def find_best_tool(user_input: str) -> dict | None:
    text = (user_input or "").strip().lower()
    if not text:
        return None
    
    # Simple keyword matching - much lighter than sentence-transformers
    best_tool = None
    best_score = 0
    
    for tool in TOOLS:
        score = 0
        keywords = tool.get("keywords", [])
        tool_name = tool.get("name", "").lower()
        tool_title = tool.get("title", "").lower()
        
        # Check if tool name or title is mentioned
        if tool_name.replace("-", " ") in text or tool_name.replace("-", "") in text:
            score += 2
        if tool_title in text:
            score += 2
        
        # Check keywords
        for keyword in keywords:
            if keyword.lower() in text:
                score += 1
        
        if score > best_score:
            best_score = score
            best_tool = tool
    
    if best_tool and best_score > 0:
        return {
            "tool": best_tool,
            "confidence": min(best_score / max(len(best_tool.get("keywords", [])), 1), 1.0)
        }
    
    return None

def fuzzy_search_tools(user_input: str, limit: int = 5) -> list:
    """Fuzzy search to find multiple matching tools"""
    text = (user_input or "").strip().lower()
    if not text or len(text) < 2:
        return []
    
    results = []
    
    for tool in TOOLS:
        score = 0
        tool_name = tool.get("name", "")
        tool_title = tool.get("title", "")
        keywords = tool.get("keywords", [])
        
        # Fuzzy match on title
        title_score = fuzzy_match_score(text, tool_title)
        score = max(score, title_score)
        
        # Fuzzy match on name
        name_score = fuzzy_match_score(text, tool_name.replace("-", " "))
        score = max(score, name_score)
        
        # Check keywords with fuzzy matching
        for keyword in keywords:
            keyword_score = fuzzy_match_score(text, keyword)
            score = max(score, keyword_score)
        
        # Also check exact keyword matches (boost score)
        for keyword in keywords:
            if keyword.lower() in text:
                score += 10
        
        if score > 20:  # Threshold for showing results
            results.append({
                "tool": tool,
                "score": score
            })
    
    # Sort by score descending and return top results
    results.sort(key=lambda x: x["score"], reverse=True)
    return [r["tool"] for r in results[:limit]]

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
    
    # First try exact/best match
    result = find_best_tool(message)
    
    if result and result['confidence'] > 0.2:
        tool = result['tool']
        responses = [
            f"✨ Ah, I sense you desire {tool['title']}! Shall I grant this wish?",
            f"🧞‍♂️ Perfect! {tool['title']} would work wonders here. Apply this magic?",
            f"🌟 Excellent choice! {tool['title']} it is! Ready to transform?",
        ]
        response = random.choice(responses)
        return jsonify({
            "response": response,
            "type": "suggestion",
            "tool": tool,
            "confidence": result['confidence']
        })
    else:
        # Use fuzzy search to find multiple matches
        fuzzy_results = fuzzy_search_tools(message, limit=5)
        
        if fuzzy_results:
            responses = [
                f"✨ I found {len(fuzzy_results)} magical effect(s) that might match your wish:",
                f"🧞‍♂️ Behold! {len(fuzzy_results)} enchantments that could fulfill your desire:",
                f"🌟 My powers reveal {len(fuzzy_results)} effects that may suit your needs:",
            ]
            return jsonify({
                "response": random.choice(responses),
                "type": "fuzzy_results",
                "tools": fuzzy_results
            })
        else:
            responses = [
                "✨ Hmm, I'm not sure which magic you seek. Browse my collection of effects:",
                "🧞‍♂️ Your wish is unclear to me. Choose from my magical effects:",
                "🌟 I need more clarity! Select from my enchanted effects below:",
            ]
            return jsonify({
                "response": random.choice(responses),
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
    use_processed = data.get('use_processed', False)  # If True, use processed image as input
    
    if not filename or not tool_name:
        return jsonify({"error": "Missing parameters"}), 400
    
    # Determine input path - either from uploads or processed folder
    if use_processed:
        input_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    else:
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
            elif tool_name == "oil-painting":
                out = do_oil_painting(img)
            elif tool_name == "cartoon-anime":
                out = do_cartoon_anime(img)
            elif tool_name == "black-white":
                out = do_black_white(img)
            elif tool_name == "emboss":
                out = do_emboss(img)
            elif tool_name == "edge-detection":
                out = do_edge_detection(img)
            elif tool_name == "watercolor":
                out = do_watercolor(img)
            elif tool_name == "posterize":
                out = do_posterize(img)
            elif tool_name == "neon-glow":
                out = do_neon_glow(img)
            elif tool_name == "mirror-horizontal":
                out = do_mirror_horizontal(img)
            elif tool_name == "color-boost":
                out = do_color_boost(img)
            else:
                return jsonify({"error": "Unknown tool"}), 400
            
            cv2.imwrite(output_path, out)
        
        success_msg = random.choice(GENIE_RESPONSES["success"])
        tip = GENIE_RESPONSES["tips"].get(tool_name, "")
        
        return jsonify({
            "success": True,
            "output_file": f"/processed/{output_filename}",
            "message": success_msg,
            "tip": tip
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
