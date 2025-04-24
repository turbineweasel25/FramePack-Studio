# FramePack Studio

FramePack Studio is an enhanced version of the FramePack demo script, designed to create intricate video scenes with improved prompt adherence. This is very much a work in progress, expect some bugs and broken features. 
![screencapture-127-0-0-1-7860-2025-04-22-20_48_38](https://github.com/user-attachments/assets/96925b79-9edf-4260-bf5b-8b30201ddca4)

## Current Features

- **Timestamped Prompts**: Define different prompts for specific time segments in your video
- **Improved Prompt Adherence**: Better alignment between your text descriptions and the generated video
- **Queue System**: Process multiple generation jobs without blocking the interface
- **Metadata Saving**: Prompt and seed are encoded into the output PNG, all other generation metadata is saved in a JSON file for easier importing

## Fresh Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU with at least 8GB VRAM (16GB+ recommended)

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

## Add to an Existing FramePack Installation

### Setup

1. Drop studio.py, the 'modules' folder and requirements.txt into the root of your FramePack installation.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

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

Each timestamp defines when that prompt should start influencing the generation. The system will (hopefully) smoothly transition between prompts for a cohesive video.

## Credits
Many thanks to [Lvmin Zhang](https://github.com/lllyasviel) for the absolutely amazing work on the original [FramePack](https://github.com/lllyasviel/FramePack) code!

## Cite

    @article{zhang2025framepack,
        title={Packing Input Frame Contexts in Next-Frame Prediction Models for Video Generation},
        author={Lvmin Zhang and Maneesh Agrawala},
        journal={Arxiv},
        year={2025}
    }
