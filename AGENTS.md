# Gallery Genie - AI Coding Instructions

## Project Overview
**Gallery Genie** is a Flask-based web application that provides AI-powered image transformation tools. The core architecture consists of:
- **Backend**: Flask server (`app.py`) with 6 image processing tools and AI-driven tool recommendation
- **Frontend**: Single-page application in `templates/index.html` with real-time chat interface
- **AI Core**: Semantic similarity matching using SentenceTransformer to recommend tools based on user intent

## Architecture & Data Flow

### Tool Recommendation Pipeline
The app uses `sentence-transformers` (all-MiniLM-L6-v2 model) for semantic understanding:
1. User inputs natural language description in chat
2. `find_best_tool()` encodes user input + all tool descriptions as embeddings
3. Cosine similarity ranking selects best-matching tool (confidence threshold: 0.35)
4. Three response types returned: `suggestion` (high confidence), `menu` (low confidence), `prompt` (empty input)

**Key insight**: Tool selection is confidence-based, not hardcoded - this allows extensibility when adding new transformation functions.

### Image Processing Flow
1. Upload endpoint receives file → sanitizes filename → stores in `static/uploads/`
2. Process endpoint applies transformation → saves to `static/processed/` with timestamped filename
3. Each tool has dedicated function: `do_<tool_name>(image)` pattern (e.g., `do_pixelate_8bit()`)
4. Special handling: `ascii-art` uses `ascii_magic` library (different I/O), others use `cv2.imread/imwrite`

### Frontend-Backend Contract
- Chat messages trigger `/api/chat` → returns JSON with `response`, `type`, and optional `tool`/`tools`
- File uploads send multipart form to `/api/upload` → receive JSON with `filename` for later processing
- Processing requests POST to `/api/process` with `filename` and `tool` parameters
- All responses include character data (GENIE_RESPONSES) for personality

## Adding New Image Transformation Tools

1. **Define tool metadata in TOOLS list** (line 26-67):
   ```python
   {"name": "tool-slug", "title": "Display Name", "description": "Detailed description for embeddings...", "icon": "..."}
   ```
   *Note: Description quality directly impacts semantic matching accuracy*

2. **Implement transformation function** using naming convention `do_<tool_slug>()`:
   - Accept `cv2` image object (BGR format)
   - Return processed image
   - For special I/O (like ascii-art), handle path-based processing

3. **Add router case** in `/api/process` endpoint (line 231-244)

4. **Add response personality** in GENIE_RESPONSES["tips"] (line 106-111)

## Critical Patterns & Conventions

### Image Path Management
- Uploaded files: `static/uploads/{timestamp}_{original_filename}`
- Processed files: `static/processed/{root}_{tool_slug}_{timestamp}{ext}`
- Use `output_path_from()` utility to generate consistent output paths

### Error Handling
- File validation: extension check (`ALLOWED_EXTENSIONS`), size limit (16MB), file existence
- Graceful degradation: if tool confidence < 0.35, show menu instead of guessing
- All errors return JSON with "error" key for consistent frontend handling

### Personality Integration
- Four response categories: greetings, prompts, processing, success
- Each category contains 2-3 variations - randomly selected to feel natural
- Tool-specific tips always included in success response

## Development & Testing

### Running Locally
```bash
pip install -r requirements.txt
python app.py
# Server runs on http://0.0.0.0:5000 with debug=True
```

### Key Dependencies
- **Flask**: Web server framework
- **opencv-python (cv2)**: Image manipulation (5 of 6 effects)
- **sentence-transformers**: Semantic tool recommendation
- **ascii-magic**: ASCII art generation (special case)
- **torch**: Required by sentence-transformers for GPU support

### Static Folders
Must exist before first request: `static/uploads/`, `static/processed/`
- Created automatically on startup (line 17-18)
- Ensure server has write permissions

## Important Notes
- Model loading is lazy (`get_model()` caches globally) - first tool recommendation request takes ~2-3 seconds
- Confidence threshold (0.35) should be tuned empirically based on new tools
- ASCII art effect has inverted process (path-based) vs others (in-memory cv2 processing)
- Frontend expects `/processed/{filename}` as absolute URL path for serving processed images
