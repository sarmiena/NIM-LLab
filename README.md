# NIM-LLab

Simple tool to set up NVIDIA NIMs using NVIDIA's [nim-llm](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/llm-nim) container. Download models, configure everything, and deploy with one command. NOTE: Only GGUF supported right now

## What it does

- Downloads model configs from HuggingFace
- Downloads files from quantized repos
- Sets up clean directory structure
- Deploys with NVIDIA NIM Docker containers
- Tests the API endpoint

## Prerequisites

- Docker with NVIDIA Container Runtime
- NGC CLI tool (`ngc`)
- Python 3.8+
- NVIDIA GPU

## Quick Start

1. **Clone and setup:**
   ```bash
   git clone <repo-url>
   cd NIM-LLab
   python3 run-gguf.py
   ```

2. **Enter your tokens when prompted:**
   - NGC API Key (starts with `nvapi-`)
   - HuggingFace Token (starts with `hf_`)

3. **Select model:**
   - Choose existing local model, or
   - Download new model from HuggingFace

4. **Wait for deployment**
   - Script automatically starts NIM container
   - Monitors logs until service is ready

## Directory Structure

Models are organized as:
```
workdir/
├── Llama-3.2-3B-Instruct/          # Model name
│   └── bartowski-Q4_K_M/           # Author-Quantization
│       ├── README.md               # Source info
│       ├── config.json             # Model configs
│       └── *.gguf                  # Model files
└── .cache/nim/                     # NIM cache
```

## Test the API

Once deployed, test with curl:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy-token" \
  -d '{
    "model": "Llama-3.2-3B-Instruct-Q4_K_M",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Example Workflow

1. **Download configs**: `meta-llama/Llama-3.2-3B-Instruct`
2. **Download GGUF**: `bartowski/Llama-3.2-3B-Instruct-GGUF`
3. **Select quantization**: `Q4_K_M` 
4. **Deploy**: Automatic Docker container with NIM
5. **Test**: API available at `localhost:8000`

## Environment Variables

Set automatically by the script:
- `FINAL_MODEL_DIR`: Path to deployed model
- `CONTAINER_NAME`: Docker container name
- `LOCAL_NIM_CACHE`: NIM cache directory

## Tips

- First run takes longer (downloads Docker image)
- Use Q4_K_M for good speed/quality balance
- Check `docker logs GGUF-NIM` if issues occur
- Service startup can take 5-10 minutes

## Requirements

See `requirements.txt`:
- docker
- requests  
- huggingface-hub
