# FramePack Studio

FramePack Studio is an enhanced version of the HunyuanVideo FramePack model, designed to create intricate video scenes with improved prompt adherence and advanced generation capabilities. This tool extends the original FramePack functionality with timestamped prompts, better quality control, and a user-friendly interface.

## Current Features

- **Timestamped Prompts**: Define different prompts for specific time segments in your video
- **Improved Prompt Adherence**: Better alignment between your text descriptions and the generated video
- **Queue System**: Process multiple generation jobs without blocking the interface

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU with at least 8GB VRAM (16GB+ recommended)
- [Git LFS](https://git-lfs.github.com/) installed

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/framepack-studio.git
   cd framepack-studio
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Create a Hugging Face token if you want to access gated models:
   - Create an account on [Hugging Face](https://huggingface.co/)
   - Generate a token in your account settings
   - Set it as an environment variable: `export HF_TOKEN=your_token_here`

## Usage

Run the studio interface:

```bash
python studio.py
```

Additional command line options:
- `--share`: Create a public Gradio link to share your interface
- `--server`: Specify the server address (default: 0.0.0.0)
- `--port`: Specify a custom port
- `--inbrowser`: Automatically open the interface in your browser

## Working with Timestamped Prompts

You can create videos with changing prompts over time using the following syntax:

```
[0s] A serene forest with sunlight filtering through the trees
[5s] A deer appears in the clearing
[10s] The deer drinks from a small stream
```

Each timestamp defines when that prompt should start influencing the generation. The system will smoothly transition between prompts for a cohesive video.

## Credits
Many thanks to [Lvmin Zhang](https://github.com/lllyasviel) for the absolutely amazing work on the original [FramePack](https://github.com/lllyasviel/FramePack) code!

## Cite

    @article{zhang2025framepack,
        title={Packing Input Frame Contexts in Next-Frame Prediction Models for Video Generation},
        author={Lvmin Zhang and Maneesh Agrawala},
        journal={Arxiv},
        year={2025}
    }
