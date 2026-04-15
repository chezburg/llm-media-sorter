# AI Image Sorter

AI-powered photo/video organization tool using local LLM (LMStudio) to automatically sort your media files into keep/trash categories.

## Features

- **AI-Powered Classification**: Uses local LLM via LMStudio API to analyze and categorize media
- **Three-Tier Sorting**: Sort into keep, review, or trash folders
- **Custom Rules**: Define your own trash criteria
- **Concurrent Processing**: Process multiple files in parallel for speed
- **Checkpoint/Reload**: Periodic model checkpoint to prevent memory issues
- **Configurable**: Easy-to-edit variables at the top of the script

## Requirements

- Python 3.8+
- [LMStudio](https://lmstudio.ai/) running locally with a vision-capable model
- Python packages:
  - `pip install opencv-python pillow requests`

## Quick Start

1. **Install LMStudio** and download a vision model

2. **Install dependencies**:
   ```bash
   pip install opencv-python pillow requests
   ```

3. **Configure** the script by editing the variables at the top of `sort.py`:
   - Set `SOURCE_DIR` to your photo directory
   - Set `API_URL` to your LMStudio endpoint (default: `http://localhost:1234/v1/chat/completions`)
   - Set `MODEL` to your preferred model
   - Add custom rules to `USER_RULES` if needed

4. **Run**:
   ```bash
   python sort.py
   ```

## Configuration

### Directory Paths

| Variable | Description | Default |
|----------|-------------|---------|
| `SOURCE_DIR` | Directory containing media to sort | `./photos` |
| `KEEP_DIR` | Destination for files to keep | `./keep` |
| `REVIEW_DIR` | Destination for uncertain files | `./review` |
| `TRASH_DIR` | Destination for files to discard | `./trash` |
| `FAILED_DIR` | Destination for failed analyses | `./failed` |

### Sorting Behavior

| Variable | Description | Default |
|----------|-------------|---------|
| `CONFIDENCE_THRESHOLD` | Min score to keep (1-10) | `7` |
| `MAX_WORKERS` | Concurrent workers | `4` |
| `CHECKPOINT_INTERVAL` | Files before model checkpoint | `50` |
| `SLEEP_DURATION` | Seconds to sleep during checkpoint | `30` |

### Model Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL` | Model to use | `google/gemma-4-e4b` |
| `TEMPERATURE` | LLM temperature | `0.0` |
| `SEED` | LLM seed for consistency | `42` |

### Custom Rules

Add criteria to `USER_RULES` list to guide the LLM:

```python
USER_RULES = [
    "no memes",
    "no screenshots",
    "no internet content",
    "no low quality images"
]
```

## How It Works

1. Scans `SOURCE_DIR` for image/video files
2. For each file:
   - Extracts representative frames (for videos) or encodes the image
   - Sends to LLM with prompt asking for classification
   - LLM returns JSON with reasoning, confidence (1-10), and category
3. If confidence >= `CONFIDENCE_THRESHOLD`, file moves to `KEEP_DIR`
4. If confidence 4-6, file moves to `REVIEW_DIR`
5. If confidence <= 3, file moves to `TRASH_DIR`
6. Every `CHECKPOINT_INTERVAL` files, the model is checkpointed/reloaded to prevent memory drift

## Deduplication

After sorting, use `dedupe.ps1` to remove duplicate files based on content hash:

```powershell
.\dedupe.ps1
```

By default runs in dry-run mode. Set `$DryRun = $false` to actually delete duplicates.
