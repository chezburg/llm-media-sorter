"""
LLM Media Sorter - AI-powered photo/video organization tool

Analyzes media files using a local LLM (via Ollama API) and sorts them into
keep/trash categories based on custom criteria.
"""

import os
import re
import base64
import json
import shutil
import cv2
from PIL import Image
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed


# =============================================================================
# FREQUENTLY CHANGED VARIABLES - Edit these to customize behavior
# =============================================================================

# Directory paths
SOURCE_DIR = "./photos"       # Source directory containing media to sort
TRASH_DIR = "./trash"       # Destination for files to discard
API_URL = "http://localhost:1234/v1/chat/completions"  # Ollama API endpoint

# Sorting behavior
CONFIDENCE_THRESHOLD = 7    # 1-10 scale: files with score >= this get kept, lower get trashed
MAX_WORKERS = 4             # Number of concurrent workers (adjust based on VRAM)

# Supported file extensions
EXTENSIONS_IMG = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic')
EXTENSIONS_VID = ('.mp4', '.mov', '.avi', '.mkv')

# Custom rules - add personal criteria for what should be trashed
# Examples: ["no cats", "no memes", "no screenshots", "no internet content"]
# These rules are injected into the LLM prompt to guide classification
USER_RULES = []


# =============================================================================
# LLM CONFIGURATION - Adjust these for different models/use cases
# =============================================================================

# Model to use (must be available in your Ollama instance)
MODEL = "google/gemma-4-e4b"

# LLM parameters
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 40
MIN_P = 0.0
PRESENCE_PENALTY = 2.0
REPETITION_PENALTY = 1.0


# =============================================================================
# SYSTEM PROMPT - Customize the LLM instructions
# =============================================================================

SYSTEM_PROMPT = (
    "Analyze this image/video frame. Be STRICT but fair."
    "\n\nReturn ONLY a JSON object: "
    '{"reasoning": "...", "confidence_keep": 1-10, "category": "meme/reaction/screenshot/photo/video/travel/polaroid/etc"}'
    "\n\nKEEP criteria (high confidence_keep = definitely keep):"
    "\n- Personal life moments, real photos of people/places you care about"
    "\n- Travel photos, vacation pictures, polaroids"
    "\n- Photos of friends, family, real memories"
    "\n- High quality photos, well-taken shots"
    "\n\nTRASH criteria (low confidence_keep = definitely trash):"
    "\n- Reaction images, memes, internet content"
    "\n- Screenshots, reposts, low-quality random downloads"
    "\n- Content that doesn't represent YOUR real life"
    "\n\nIMPORTANT: confidence_keep 8-10 = definitely keep (real memories). confidence_keep 1-3 = definitely trash."
)


# =============================================================================
# IMPLEMENTATION - Do not edit unless you know what you're doing
# =============================================================================

# Initialize trash directory
if not os.path.exists(TRASH_DIR):
    os.makedirs(TRASH_DIR)


def build_prompt():
    """Build the complete prompt by injecting user rules if provided."""
    prompt = SYSTEM_PROMPT
    
    if USER_RULES:
        rules_text = "\n\nCUSTOM TRASH RULES:"
        for rule in USER_RULES:
            rules_text += f"\n- {rule}"
        prompt += rules_text
    
    return prompt


def encode_image_and_resize(image_path):
    """Resizes image to max 1024px to save VRAM and encodes to base64."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((1024, 1024))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def get_video_frame(video_path):
    """Extracts a middle frame from a video to judge content."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
    ret, frame = cap.read()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    return None


def analyze_file(file_path):
    """Analyzes a single media file using the LLM."""
    ext = os.path.splitext(file_path)[1].lower()
    img_b64 = None

    try:
        if ext in EXTENSIONS_IMG:
            img_b64 = encode_image_and_resize(file_path)
        elif ext in EXTENSIONS_VID:
            img_b64 = get_video_frame(file_path)
        
        if not img_b64:
            return None

        prompt = build_prompt()

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "min_p": MIN_P,
            "presence_penalty": PRESENCE_PENALTY,
            "repetition_penalty": REPETITION_PENALTY
        }

        response = requests.post(API_URL, json=payload, timeout=300)
        
        if response.status_code != 200:
            print(f"Server Error ({response.status_code}): {response.text}")
            return None
        
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            print(f"Invalid JSON response: {response.text[:500]}")
            return None

        if 'choices' not in response_data:
            print(f"Unexpected Response Format: {response_data}")
            return None

        raw_content = response_data['choices'][0]['message']['content']
        
        json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if not json_match:
            print(f"Could not find JSON in response: {raw_content[:200]}")
            return None
        
        return json_match.group(0)

    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Raw response: {response.text[:500]}")
        return None
    except Exception as e:
        print(f"Script Error processing {file_path}: {e}")
        return None


def process_file(filename):
    """Process a single file - analyze and move to trash or keep."""
    file_path = os.path.join(SOURCE_DIR, filename)
    result = analyze_file(file_path)
    
    if result:
        data = json.loads(result)
        reason = data.get("reasoning")
        conf = data.get("confidence_keep", 0)
        
        print(f"[{filename}] Keep Conf: {conf}/10 | Reason: {reason}")

        if conf >= CONFIDENCE_THRESHOLD:
            print(f"---> KEEPING (score >= {CONFIDENCE_THRESHOLD})")
        else:
            shutil.move(file_path, os.path.join(TRASH_DIR, filename))
            print(f"---> MOVED TO TRASH (score < {CONFIDENCE_THRESHOLD})")
    else:
        print(f"[{filename}] Failed to analyze, skipping")


def main():
    """Main entry point."""
    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(EXTENSIONS_IMG + EXTENSIONS_VID)]
    print(f"Found {len(files)} files. Starting the sort with {MAX_WORKERS} concurrent workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, filename): filename for filename in files}
        
        for i, future in enumerate(as_completed(futures), 1):
            filename = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[{filename}] Error: {e}")
            print(f"Progress: {i}/{len(files)} completed")


if __name__ == "__main__":
    main()