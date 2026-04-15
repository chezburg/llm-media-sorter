"""
AI Image Sorter - AI-powered photo/video organization tool

Analyzes media files using a local LLM (via LMStudio API) and sorts them into
keep/trash categories based on custom criteria.
"""

import os
import re
import base64
import json
import shutil
import time
import cv2
import threading
from PIL import Image
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# FREQUENTLY CHANGED VARIABLES - Edit these to customize behavior
# =============================================================================

# Directory paths
SOURCE_DIR = "./photos"          # Source directory containing media to sort
KEEP_DIR = "./keep"              # Destination for files to keep
REVIEW_DIR = "./review"          # Destination for uncertain files
TRASH_DIR = "./trash"            # Destination for files to discard
FAILED_DIR = "./failed"          # Destination for failed analyses
API_URL = "http://localhost:1234/v1/chat/completions"  # LMStudio API endpoint

# Sorting behavior
CONFIDENCE_THRESHOLD = 7         # 1-10 scale: 7+ keep, 4-6 review, 1-3 trash
MAX_WORKERS = 4                  # Number of concurrent workers (adjust based on VRAM)
CHECKPOINT_INTERVAL = 50        # Files to process before model checkpoint/reload
SLEEP_DURATION = 30              # Seconds to sleep during checkpoint

# Model settings
MODEL = "google/gemma-4-e4b"        # Ollama model to use
TEMPERATURE = 0.0               # Stable cache
SEED = 42                       # Stable cache

# Supported file extensions
EXTENSIONS_IMG = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic')
EXTENSIONS_VID = ('.mp4', '.mov', '.avi', '.mkv')

# Custom rules - add criteria for what should be trashed
# Examples: ["no memes", "no screenshots", "no internet content", "no low quality"]
# These rules are injected into the LLM prompt to guide classification
USER_RULES = []

# Thread-local storage for persistent sessions
thread_local = threading.local()

def get_session():
    """Returns a thread-local requests.Session object."""
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


# =============================================================================
# INITIALIZATION - Creates output directories
# =============================================================================

for dir_path in [KEEP_DIR, REVIEW_DIR, TRASH_DIR, FAILED_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def encode_image_and_resize(image_path):
    """Resizes image to max 1024px to save VRAM and encodes to base64."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((1024, 1024))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def get_video_frames(video_path, frame_count=3):
    """Extracts N frames from a video (start, middle, end) to judge content."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < frame_count:
        return None
    
    frames = []
    positions = [0, total_frames // 2, total_frames - 1]
    
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            frames.append(base64.b64encode(buffer).decode('utf-8'))
    
    cap.release()
    return frames if len(frames) == frame_count else None


# =============================================================================
# SYSTEM PROMPT - Customize the LLM instructions
# =============================================================================

def build_prompt(file_size_mb):
    """Build the complete prompt by injecting user rules if provided."""
    prompt = (
        "Analyze this image/video frame. Be STRICT but fair."
        f"\n\nFile size: {file_size_mb:.1f} MB"
        "\n\nReturn ONLY a JSON object: "
        '{"reasoning": "...", "confidence_keep": 1-10, "category": "meme/reaction/screenshot/photo/video/travel/art/etc"}'
        "\n\nKEEP criteria (high confidence_keep = definitely keep):"
        "\n- Meaningful personal moments you want to preserve"
        "\n- High quality, well-taken photos"
        "\n- Content with sentimental or personal value"
        "\n\nTRASH criteria (low confidence_keep = definitely trash):"
        "\n- Reaction images, memes, internet content"
        "\n- Screenshots, reposts, low-quality content"
        "\n- Content you don't need to keep"
        "\n\nIMPORTANT: Files >= 50MB are often original quality photos - consider this as a very positive signal for keep."
        "\n\nIMPORTANT: confidence_keep 8-10 = definitely keep. confidence_keep 1-3 = definitely trash."
    )
    
    if USER_RULES:
        rules_text = "\n\nCUSTOM TRASH RULES:"
        for rule in USER_RULES:
            rules_text += f"\n- {rule}"
        prompt += rules_text
    
    return prompt


# =============================================================================
# LLM ANALYSIS
# =============================================================================

def analyze_file(file_path, worker_id):
    ext = os.path.splitext(file_path)[1].lower()
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    img_b64 = None

    try:
        if ext in EXTENSIONS_IMG:
            img_b64 = encode_image_and_resize(file_path)
        elif ext in EXTENSIONS_VID:
            frames = get_video_frames(file_path, frame_count=3)
            if frames:
                img_b64 = frames
        
        if not img_b64:
            return None

        prompt = build_prompt(file_size_mb)

        content = [{"type": "text", "text": prompt}]
        
        if isinstance(img_b64, list):
            for frame_b64 in img_b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})
        else:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "user": f"worker_{worker_id}",
            "temperature": TEMPERATURE,
            "seed": SEED
        }

        session = get_session()
        response = session.post(API_URL, json=payload, timeout=300)

        
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
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return None
    except Exception as e:
        print(f"Script Error processing {file_path}: {e}")
        return None


# =============================================================================
# MODEL MANAGEMENT - Checkpoint/reload cycle
# =============================================================================

def manage_model():
    """Performs the unload/reload cycle for model checkpointing."""
    print(f"\n{'='*60}")
    print(f"[CHECKPOINT] Unloading model... Sleeping for {SLEEP_DURATION}s.")
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Force Unload
    try:
        requests.post("http://localhost:1234/api/v1/models/unload", 
                      json={"instance_id": MODEL}, 
                      headers=headers, timeout=15)
    except Exception as e:
        print(f"[CHECKPOINT] Unload failed: {e}")
    
    time.sleep(SLEEP_DURATION)
    
    # Force Reload
    print(f"[CHECKPOINT] Reloading model...")
    load_payload = {
        "model": MODEL,
        "context_length": 16384,
        "flash_attention": True
    }
    try:
        requests.post("http://localhost:1234/api/v1/models/load", 
                      json=load_payload, 
                      headers=headers, timeout=60)
    except Exception as e:
        print(f"[CHECKPOINT] Reload failed: {e}")
    
    print(f"{'='*60}\n")


# =============================================================================
# FILE PROCESSING
# =============================================================================

def process_file(filename, worker_id):
    start_time = time.perf_counter()
    file_path = os.path.join(SOURCE_DIR, filename)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    result = analyze_file(file_path, worker_id)
    elapsed = time.perf_counter() - start_time
    
    if result:
        data = json.loads(result)
        reason = data.get("reasoning")
        conf = data.get("confidence_keep", 0)
        
        print(f"[{filename}] {file_size_mb:.1f}MB | Keep Conf: {conf}/10 | Reason: {reason}")

        if conf >= CONFIDENCE_THRESHOLD:
            shutil.move(file_path, os.path.join(KEEP_DIR, filename))
            print(f"---> MOVED TO KEEP (score >= {CONFIDENCE_THRESHOLD})")
        elif conf >= 4:
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))
            print(f"---> MOVED TO REVIEW (score 4-6)")
        else:
            shutil.move(file_path, os.path.join(TRASH_DIR, filename))
            print(f"---> MOVED TO TRASH (score 1-3)")
    else:
        shutil.move(file_path, os.path.join(FAILED_DIR, filename))
        print(f"[{filename}] Failed to analyze, moved to failed")
    
    return elapsed


# =============================================================================
# MAIN
# =============================================================================

def main():
    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(EXTENSIONS_IMG + EXTENSIONS_VID)]
    print(f"Found {len(files)} files. Starting the sort with {MAX_WORKERS} concurrent workers...")

    ema_time = None
    processed_count = 0
    
    # Process files in chunks of CHECKPOINT_INTERVAL
    for chunk_start in range(0, len(files), CHECKPOINT_INTERVAL):
        chunk = files[chunk_start:chunk_start + CHECKPOINT_INTERVAL]
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Round-robin assignment of worker IDs
            futures = {executor.submit(process_file, filename, (processed_count + i) % MAX_WORKERS): filename 
                       for i, filename in enumerate(chunk)}
            
            for future in as_completed(futures):
                processed_count += 1
                filename = futures[future]
                try:
                    elapsed = future.result()
                    
                    if ema_time is None:
                        ema_time = elapsed
                    else:
                        ema_time = 0.3 * elapsed + 0.7 * ema_time
                    
                    remaining = len(files) - processed_count
                    eta_seconds = remaining * ema_time
                    
                    if eta_seconds < 60:
                        eta_str = f"{int(eta_seconds)}s"
                    elif eta_seconds < 3600:
                        eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                    else:
                        eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"
                    
                except Exception as e:
                    print(f"[{filename}] Error: {e}")
                    remaining = len(files) - processed_count
                    eta_str = "calculating..."
                
                print(f"Progress: {processed_count}/{len(files)} | ETA: {eta_str} remaining")

        # After each chunk (except the last one), perform the checkpoint
        if processed_count < len(files):
            manage_model()


if __name__ == "__main__":
    main()
