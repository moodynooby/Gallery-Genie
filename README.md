# **Project Title: GALLERY GENIE**


## Objectives of developing software/application

- To democratize image editing to anyone with a computer without needing a PHD in photoshop or image editing in general
- To create an interactive command-line application that applies various image processing effects to user-provided images without need for a heavy GUI to also support older systems.
- To implement intelligent tool suggestion using natural language processing techniques (TF-IDF and cosine similarity) based on user descriptions.
- To support cross-platform image viewing and processing, enhancing accessibility for users on different operating systems.

## End-Users of the software

- General Users: Individuals interested in experimenting with image effects for creative or fun purposes, such as hobbyists or students.
- Design Enthusiasts: Users who want quick transformations like sepia, sketches, or pixelation without complex software like Photoshop.

## Listing of functionality/features/main modules

- Setup Module: Handles first-time onboarding, including user name input and local token generation for session persistence.
- Login Module: Simulates authentication with token check, updates login counts, and greets returning users.
- Tool Suggestion Engine: Uses scikit-learn's TF-IDF vectorizer and cosine similarity to recommend image effects based on user's natural language query (e.g., "make it retro" suggests pixelate-8bit).
- Image Processing Modules: Implements effects using OpenCV and ascii_magic:
- ASCII Art: Converts images to ASCII art and saves as image.
- Aesthetic Blur: Applies Gaussian blur for a soft, dreamy effect.
- Vintage Sepia: Converts to grayscale and applies warm sepia tones.
- Invert Colors: Inverts the image for a negative/cyberpunk look.
- 8-bit Pixelate: Resizes to low resolution and upscales for pixelated retro style.
- Pencil Sketch: Generates a sketch-like effect using edge detection and blending.
- Path Management: Remembers and prompts for last used image path, validates file existence, and generates descriptive output filenames.
- Output Handling: Saves processed images with effect-specific names and opens them in the default system viewer.
- Interactive Menu: Allows manual tool selection if suggestion is not used or score is low.
