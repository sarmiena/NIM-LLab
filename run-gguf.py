import os
import getpass
from lib.nim_selector import select_nim_version
from lib.model_downloader import download_model_configs, download_or_select_gguf_model
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

# Create necessary directories
print("Creating necessary directories...")
try:
    os.makedirs(os.environ["LOCAL_NIM_CACHE"], exist_ok=True)
    print(green_text(f"✓ Cache directory: {os.environ['LOCAL_NIM_CACHE']}"))

    os.makedirs(base_work_dir, exist_ok=True)
    print(green_text(f"✓ Base work directory: {base_work_dir}"))

except Exception as e:
    print(red_text(f"❌ Failed to create directories: {e}"))
    exit(1)

print("\n📦 Selecting GGUF model...")
print(yellow_text("You can use an existing local GGUF model or download a new one from HuggingFace."))
print(yellow_text("New structure: workdir/{model_name}/{author-quantization}/"))

# Scan for local models and allow selection or download new
final_model_dir = download_or_select_gguf_model("", base_work_dir)

if final_model_dir:
    print(green_text("\n🎉 Setup completed successfully!"))
    print(f"Selected NIM version: {selected_version}")

    if os.environ.get("NIM_MODEL_DIR"):
        print(f"Base model config directory: {os.environ['NIM_MODEL_DIR']}")

    print(f"Final model directory (with GGUF files): {final_model_dir}")

    # Update environment variable to point to final directory
    os.environ["FINAL_MODEL_DIR"] = final_model_dir
    print(green_text(f"✓ FINAL_MODEL_DIR set to: {final_model_dir}"))

    print("\n📋 Summary:")
    if os.environ.get("NIM_MODEL_DIR"):
        print(f"• Base model configs: {os.environ['NIM_MODEL_DIR']}")
    print(f"• Complete model (configs + GGUF): {final_model_dir}")
    print(f"• NIM Docker version: {selected_version}")
    print(f"• Directory structure: workdir/{os.path.basename(os.path.dirname(final_model_dir))}/{os.path.basename(final_model_dir)}")

    print(green_text("\n✅ Ready to start your GGUF NIM service!"))

    # Deploy the model with Docker
    print(yellow_text("\n🚀 Deploying GGUF NIM container..."))

    # Extract quantization info from the final model directory name
    quantization = os.path.basename(final_model_dir).split('-')[-1]  # e.g., "bartowski-Q4_K_M" -> "Q4_K_M"

    # Extract base model name for the served model name
    base_model_name = os.path.basename(os.path.dirname(final_model_dir))  # e.g., "Llama-3.2-3B-Instruct"

    # Build the NIM image name
    nim_image = f"nvcr.io/nim/nvidia/llm-nim:{selected_version}"

    print(f"Deploying {quantization} quantization")
    print(f"Model: {base_model_name}")
    print(f"Docker image: {nim_image}")

    # Build the docker run command
    docker_cmd = [
        'docker', 'run', '-it', '--rm',
        '--name', os.environ["CONTAINER_NAME"],
        '--runtime=nvidia',
        '--gpus', 'all',
        '--shm-size=16GB',
        '-e', f'NIM_MODEL_NAME=/opt/models/gguf_model',
        '-e', f'NIM_SERVED_MODEL_NAME={base_model_name}-{quantization}',
        '-v', f'{final_model_dir}:/opt/models/gguf_model',
        '-v', f'{os.environ["LOCAL_NIM_CACHE"]}:/opt/nim/.cache',
        '-u', f'{os.getuid()}',
        '-p', '8000:8000',
        '-d',
        nim_image
    ]

    print(yellow_text("Starting Docker container..."))
    print(f"Command: {' '.join(docker_cmd)}")

    # Run the docker command
    result = run_command(docker_cmd)

    if result['success']:
        container_id = result['stdout'].strip()
        print(green_text(f"✓ Container started successfully!"))
        print(f"Container ID: {container_id}")
        print(f"Container name: {os.environ['CONTAINER_NAME']}")

        print(yellow_text("\n⏳ Waiting for NIM service to be ready..."))
        print("This may take several minutes for the first run...")

        # Check if service is ready using log monitoring
        if check_service_ready_from_logs(os.environ["CONTAINER_NAME"], print_logs=True, timeout=900):
            print(green_text("\n🎉 NIM service is ready!"))
            print(f"API endpoint: http://localhost:8000")
            print(f"Model: {base_model_name}-{quantization}")
            print(f"Health check: http://localhost:8000/v1/health/ready")

            # Optional: Test the service
            test_prompt = input("\nWould you like to test the service with a prompt? (y/n): ").strip().lower()
            if test_prompt == 'y':
                user_prompt = input("Enter your test prompt: ").strip()
                if user_prompt:
                    print(yellow_text("\n🤖 Generating response..."))
                    response = generate_text(f"{base_model_name}-{quantization}", user_prompt)
                    if response:
                        print(green_text("\n✅ Response:"))
                        print(response)
                    else:
                        print(red_text("❌ Failed to generate response"))

        else:
            print(red_text("\n❌ Service failed to start properly"))
            print("Check container logs with: docker logs " + os.environ["CONTAINER_NAME"])

    else:
        print(red_text(f"❌ Failed to start container: {result.get('stderr', result.get('error', 'Unknown error'))}"))
        print("Make sure Docker is running and you have NVIDIA Container Runtime installed")
        exit(1)

else:
    print(red_text("❌ Failed to set up model. Exiting."))
    exit(1)
