# LLM Media Sorter

AI-powered photo/video organization tool using local LLM (Ollama) to automatically sort your media files into keep/trash categories.

## Features

- **AI-Powered Classification**: Uses local LLM via Ollama API to analyze and categorize media
- **Custom Rules**: Define your own trash criteria (e.g., "no cats", "no memes")
- **Concurrent Processing**: Process multiple files in parallel for speed
- **Configurable**: Easy-to-edit variables at the top of the script

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) running locally with a vision-capable model
- Python packages:
  - `pip install opencv-python pillow requests`

## Quick Start

1. **Install Ollama** and pull a vision model:
   ```bash
   ollama pull llama3.2-vision
   ```

2. **Install dependencies**:
   ```bash
   pip install opencv-python pillow requests
   ```

3. **Configure** the script by editing the variables at the top of `sort.py`:
   - Set `SOURCE_DIR` to your photo directory
   - Set `API_URL` to your Ollama endpoint (default: `http://localhost:1234/v1/chat/completions`)
   - Add custom rules to `USER_RULES` if needed

4. **Run**:
   ```bash
   python sort.py
   ```

## Configuration

### Frequently Changed Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SOURCE_DIR` | Directory containing media to sort | `./photos` |
| `TRASH_DIR` | Destination for discarded files | `./trash` |
| `API_URL` | Ollama API endpoint | `http://localhost:1234/v1/chat/completions` |
| `CONFIDENCE_THRESHOLD` | Min score to keep (1-10) | `7` |
| `MAX_WORKERS` | Concurrent workers | `4` |
| `USER_RULES` | Custom trash criteria | `[]` |

### Custom Rules

Add personal criteria to `USER_RULES` list:

```python
USER_RULES = [
    "no cats",
    "no memes",
    "no screenshots",
    "no internet content"
]
```

### Model Options

By default uses `google/gemma-4-e4b`. Change `MODEL` to use a different Ollama model:

```python
MODEL = "llama3.2-vision"
```

## How It Works

1. Scans `SOURCE_DIR` for image/video files
2. For each file:
   - Extracts a representative frame (for videos) or encodes the image
   - Sends to LLM with prompt asking for classification
   - LLM returns JSON with reasoning, confidence (1-10), and category
3. If confidence >= `CONFIDENCE_THRESHOLD`, file stays
4. Otherwise, file is moved to `TRASH_DIR`

## License

MIT