import os
import getpass
from lib.nim_selector import select_nim_version
from lib.model_downloader import download_model_configs
from lib.run_command import run_command
from lib.colors import *
from lib.utility import *

print(yellow_text("🚀 Starting GGUF NIM Setup..."))

# Replace with your actual NGC API key
if not os.environ.get("NGC_API_KEY", "").startswith("nvapi-"):
    ngc_api_key = getpass.getpass("Enter your NGC API Key: ")
    assert ngc_api_key.startswith("nvapi-"), red_text("❌ Not a valid NGC API key - must start with 'nvapi-'")
    os.environ["NGC_API_KEY"] = ngc_api_key
    print(green_text("✓ NGC API Key set successfully"))
else:
    print(green_text("✓ NGC API Key already configured"))

if not os.environ.get("HF_TOKEN", "").startswith("hf_"):
    hf_token = getpass.getpass("Enter your Huggingface Token: ")
    assert hf_token.startswith("hf_"), red_text("❌ Not a valid HuggingFace token - must start with 'hf_'")
    os.environ["HF_TOKEN"] = hf_token
    print(green_text("✓ Hugging Face token set successfully"))
else:
    print(green_text("✓ Hugging Face token already configured"))

print("\n📦 Setting up Python environment...")

if not os.path.exists('venv'):
    print("Virtual environment doesn't exist. Creating it now...")
    result = run_command(['python3', '-m', 'venv', 'venv'])
    if result['success']:
        print(green_text("✓ Virtual environment created successfully"))
    else:
        print(red_text(f"❌ Failed to create virtual environment: {result.get('stderr', 'Unknown error')}"))
        exit(1)
else:
    print(green_text("✓ Virtual environment already exists"))

print("Installing Python dependencies...")
result = run_command(['pip', 'install', '-r', 'requirements.txt'])
if result['success']:
    print(green_text("✓ Dependencies installed successfully"))
else:
    print(red_text(f"❌ Failed to install dependencies: {result.get('stderr', 'Unknown error')}"))
    exit(1)

print("\n🐳 Selecting and pulling Docker image...")
selected_version = select_nim_version(auto_pull=True)
print(green_text(f"✓ Selected NIM version: {selected_version}"))

print("\n📁 Setting up working directories...")

base_work_dir = os.path.abspath("./workdir")
os.environ["BASE_WORK_DIR"] = base_work_dir

os.environ["CONTAINER_NAME"] = "GGUF-NIM"
os.environ["LOCAL_NIM_CACHE"] = os.path.join(base_work_dir, ".cache/nim")
os.environ["GGUF_WORK_DIR"] = os.path.join(base_work_dir, "gguf_models")

# Create necessary directories
print("Creating necessary directories...")
try:
    os.makedirs(os.environ["LOCAL_NIM_CACHE"], exist_ok=True)
    print(green_text(f"✓ Cache directory: {os.environ['LOCAL_NIM_CACHE']}"))

    os.makedirs(os.environ["GGUF_WORK_DIR"], exist_ok=True)
    print(green_text(f"✓ GGUF models directory: {os.environ['GGUF_WORK_DIR']}"))

except Exception as e:
    print(red_text(f"❌ Failed to create directories: {e}"))
    exit(1)

print("\n🤗 Downloading HuggingFace model configuration...")
config_dir = download_model_configs(base_work_dir)

if config_dir:
    print(green_text("\n🎉 Setup completed successfully!"))
    print(f"Selected NIM version: {selected_version}")
    print(f"Base Model config directory: {os.environ['NIM_MODEL_DIR']}")
else:
    print(red_text("❌ Failed to download model configuration. Exiting."))
    exit(1)
