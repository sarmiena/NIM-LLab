import os
import getpass
from lib.nim_selector import select_nim_version
from lib.run_command import run_command
from lib.colors import *
from lib.utility import *


# Replace with your actual NGC API key
if not os.environ.get("NGC_API_KEY", "").startswith("nvapi-"):
    ngc_api_key = getpass.getpass("Enter your NGC API Key: ")
    assert ngc_api_key.startswith("nvapi-"), "Not a valid key"
    os.environ["NGC_API_KEY"] = ngc_api_key
    print(green_text("✓ NGC API Key set successfully"))


if not os.environ.get("HF_TOKEN", "").startswith("hf_"):
    hf_token = getpass.getpass("Enter your Huggingface Token: ")
    assert hf_token.startswith("hf_"), "Not a valid key"
    os.environ["HF_TOKEN"] = hf_token
    print(green_text("✓ Hugging Face token set successfully"))

if not os.path.exists('venv'):
    print("venv doesn't exist. Creating it now via: python3 -m venv venv")
    run_command(['python3', '-m', 'venv', 'venv'])

run_command(['source', 'venv/bin/activate'])
run_command(['pip', 'install', '-r', 'requirements.txt'])

selected_version = select_nim_version(auto_pull=True)

base_work_dir = os.path.abspath("./workdir")
os.environ["BASE_WORK_DIR"] = base_work_dir

os.environ["CONTAINER_NAME"] = "GGUF-NIM"
os.environ["LOCAL_NIM_CACHE"] = os.path.join(base_work_dir, ".cache/nim")
os.environ["GGUF_WORK_DIR"] = os.path.join(base_work_dir, "gguf_models")

# Create necessary directories
os.makedirs(os.environ["LOCAL_NIM_CACHE"], exist_ok=True)
os.makedirs(os.environ["GGUF_WORK_DIR"], exist_ok=True)

print(green_text("Directories created successfully"))
