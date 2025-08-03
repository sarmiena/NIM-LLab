from huggingface_hub import hf_hub_download, list_repo_files
from .colors import green_text, red_text, yellow_text
import os
import shutil
import glob
import re
from collections import defaultdict

# Try to import RepositoryNotFoundError, fall back to general exception handling
try:
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    try:
        from huggingface_hub import RepositoryNotFoundError
    except ImportError:
        # If we can't import it, we'll use a general Exception
        RepositoryNotFoundError = Exception


def download_model_configs(base_work_dir: str) -> str:
    """
    Download configuration files from a HuggingFace model repository.

    Args:
        base_work_dir (str): Base working directory for downloads

    Returns:
        str: Path to the config directory
    """

    while True:
        # Get model repository from user
        model_repo = input("\nEnter HuggingFace model (format: profile/model-name): ").strip()

        if not model_repo:
            print(red_text("Please enter a valid model repository."))
            continue

        if '/' not in model_repo:
            print(red_text("Please use the format: profile/model-name (e.g., meta-llama/Llama-3.2-3B-Instruct)"))
            continue

        # Create temporary directory for this model
        config_temp_dir = os.path.join("/tmp", "gguf_config_temp")
        os.makedirs(config_temp_dir, exist_ok=True)

        print(f"\nDownloading config files from {model_repo}...")

        # Files we need to find (just the base names)
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json"
        ]

        # Find the actual paths of required files in the repository
        found_files = {}

        try:
            # First, check if repository exists by trying to list files
            repo_files = []  # Initialize here
            try:
                repo_files = list_repo_files(
                    repo_id=model_repo,
                    token=os.getenv("HF_TOKEN")
                )
                print(green_text(f"✓ Repository {model_repo} found"))

                for required_file in required_files:
                    # Look for the file anywhere in the repository structure
                    matching_files = [f for f in repo_files if f.endswith(required_file)]

                    if matching_files:
                        # If multiple matches, prefer root directory, then shortest path
                        best_match = min(matching_files, key=lambda x: (x.count('/'), len(x)))
                        found_files[required_file] = best_match
                        print(green_text(f"Found {required_file} at: {best_match}"))
                    else:
                        raise Exception(f"{required_file} not found anywhere in repository")

            except RepositoryNotFoundError:
                print(red_text(f"❌ Repository '{model_repo}' not found. Please check the model name and try again."))
                continue
            except Exception as e:
                print(red_text(f"❌ Error accessing repository: {e}"))
                continue

            # Check if all required files were found
            missing_files = [f for f in required_files if f not in found_files]

            if missing_files:
                print(red_text(f"\n❌ Required files not found in repository: {', '.join(missing_files)}"))
                print(yellow_text("Available files in repository:"))
                # Show repository structure
                for f in sorted(repo_files)[:20]:
                    print(f"  {f}")
                if len(repo_files) > 20:
                    print(f"  ... and {len(repo_files) - 20} more files")

                retry = input("Would you like to try a different model? (y/n): ").strip().lower()
                if retry != 'y':
                    break
                continue

            # All files found, now download them
            downloaded_files = []
            for required_file, actual_path in found_files.items():
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=model_repo,
                        filename=actual_path,
                        local_dir=config_temp_dir,
                        token=os.getenv("HF_TOKEN")
                    )
                    downloaded_files.append(actual_path)
                    print(green_text(f"✓ {actual_path} downloaded successfully"))

                except Exception as e:
                    print(red_text(f"❌ Error downloading {actual_path}: {e}"))
                    retry = input("Would you like to try a different model? (y/n): ").strip().lower()
                    if retry != 'y':
                        return None
                    break
            else:
                # All files downloaded successfully
                print(green_text(f"\n✓ Successfully downloaded {len(downloaded_files)} config files"))
                model_dir_name = model_repo.replace("/", "-")
                os.environ["NIM_MODEL_DIR"] = os.path.join(os.environ["GGUF_WORK_DIR"], model_dir_name)
                os.makedirs(os.environ["NIM_MODEL_DIR"], exist_ok=True)
                shutil.copytree(config_temp_dir, os.environ["NIM_MODEL_DIR"], dirs_exist_ok=True)

                for file_path in glob.glob(os.path.join(config_temp_dir, '*')):
                    if os.path.isfile(file_path):
                        os.remove(file_path)

                print(f"Files saved to: {os.environ['NIM_MODEL_DIR']}")

                return config_temp_dir

        except KeyboardInterrupt:
            print(red_text("\n\nOperation cancelled by user."))
            break
        except Exception as e:
            print(red_text(f"❌ Unexpected error: {e}"))
            retry = input("Would you like to try a different model? (y/n): ").strip().lower()
            if retry != 'y':
                break

    return None


def extract_gguf_files(repo_files: list) -> dict:
    """
    Extract and organize GGUF files from repository file list.
    Groups multi-part files together and removes part numbers from display.

    Args:
        repo_files (list): List of all files in the repository

    Returns:
        dict: Dictionary mapping display names to file lists
    """
    gguf_files = [f for f in repo_files if f.endswith('.gguf')]

    if not gguf_files:
        return {}

    # Group multi-part files
    grouped_files = defaultdict(list)

    for file in gguf_files:
        # Remove directory path for processing
        filename = os.path.basename(file)

        # Check for multi-part pattern (e.g., model-00001-of-00002.gguf)
        multipart_match = re.match(r'^(.*)-\d+-of-\d+\.gguf$', filename)

        if multipart_match:
            # Multi-part file - use base name without part info
            base_name = multipart_match.group(1) + '.gguf'
            grouped_files[base_name].append(file)
        else:
            # Single file
            grouped_files[filename].append(file)

    # Sort the files within each group to ensure proper order for multi-part files
    for key in grouped_files:
        grouped_files[key].sort()

    return dict(grouped_files)


def display_gguf_menu(gguf_groups: dict) -> tuple:
    """
    Display available GGUF files and get user selection.

    Args:
        gguf_groups (dict): Dictionary mapping display names to file lists

    Returns:
        tuple: (selected_display_name, list_of_actual_files)
    """
    print("\nAvailable GGUF files:")
    print("-" * 50)

    # Sort by file size estimate (larger models typically have more descriptive names)
    sorted_items = sorted(gguf_groups.items(), key=lambda x: (len(x[1]), x[0]))

    for i, (display_name, file_list) in enumerate(sorted_items, 1):
        parts_info = f" ({len(file_list)} parts)" if len(file_list) > 1 else ""
        print(f"{i:2d}. {display_name}{parts_info}")

    print("-" * 50)

    while True:
        try:
            choice = input(f"Select a GGUF file (1-{len(sorted_items)}) or 'q' to quit: ").strip()

            if choice.lower() == 'q':
                print("Exiting...")
                return None, None

            choice_num = int(choice)
            if 1 <= choice_num <= len(sorted_items):
                selected_item = sorted_items[choice_num - 1]
                display_name, file_list = selected_item
                print(f"Selected: {display_name}")
                if len(file_list) > 1:
                    print(f"This will download {len(file_list)} parts")
                return display_name, file_list
            else:
                print(f"Please enter a number between 1 and {len(sorted_items)}")

        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None, None


def scan_local_gguf_files(gguf_work_dir: str) -> dict:
    """
    Scan for existing GGUF files in the local working directory.
    Expected structure: gguf_work_dir/profile-model-name/specific-quant-name/

    Args:
        gguf_work_dir (str): Path to the GGUF working directory

    Returns:
        dict: Dictionary mapping display paths to full directory paths
    """
    local_models = {}

    if not os.path.exists(gguf_work_dir):
        print(yellow_text(f"GGUF work directory does not exist: {gguf_work_dir}"))
        return local_models

    print(f"Scanning directory: {gguf_work_dir}")

    # Walk through the directory structure looking for .gguf files
    for root, dirs, files in os.walk(gguf_work_dir):
        gguf_files = [f for f in files if f.endswith('.gguf')]

        if gguf_files:
            print(f"Found {len(gguf_files)} GGUF files in: {root}")

            # Create a relative path from the work directory
            rel_path = os.path.relpath(root, gguf_work_dir)

            if rel_path == '.':
                display_name = f"Root directory ({len(gguf_files)} GGUF files)"
            else:
                # Just use the relative path as display name for now
                display_name = f"{rel_path} ({len(gguf_files)} GGUF files)"

            local_models[display_name] = root

    print(f"Found {len(local_models)} directories with GGUF files")
    return local_models


def display_local_gguf_menu(local_models: dict) -> str:
    """
    Display available local GGUF models and get user selection.

    Args:
        local_models (dict): Dictionary mapping display names to directory paths

    Returns:
        str: Selected directory path, "download_new" for new download, or None for quit
    """
    if not local_models:
        print(yellow_text("No local GGUF models found."))
        return "download_new"

    print("\nLocal GGUF models found:")
    print("-" * 80)

    # Simple alphabetical sorting
    sorted_items = sorted(local_models.items())

    for i, (display_name, model_path) in enumerate(sorted_items, 1):
        print(f"{i:2d}. {display_name}")

    # Add option to download new model
    download_option = len(sorted_items) + 1
    print(f"{download_option:2d}. Choose new from HF")

    print("-" * 80)

    while True:
        try:
            choice = input(f"Select a model (1-{download_option}) or 'q' to quit: ").strip()

            if choice.lower() == 'q':
                print("Exiting...")
                return None

            choice_num = int(choice)
            if 1 <= choice_num <= len(sorted_items):
                selected_item = sorted_items[choice_num - 1]
                display_name, model_path = selected_item
                print(f"Selected local model: {display_name}")
                print(f"Using directory: {model_path}")
                return model_path
            elif choice_num == download_option:
                print("Will download new model from HuggingFace...")
                return "download_new"
            else:
                print(f"Please enter a number between 1 and {download_option}")

        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


def download_or_select_gguf_model(model_repo: str, base_work_dir: str) -> str:
    """
    Allow user to select from local GGUF files or download new ones from HuggingFace.

    Args:
        model_repo (str): HuggingFace model repository (profile/model-name) - can be empty
        base_work_dir (str): Base working directory for downloads

    Returns:
        str: Path to the final model directory, or None if failed
    """

    # First, scan for local GGUF models
    print("\n🔍 Scanning for local GGUF models...")
    local_models = scan_local_gguf_files(os.environ["GGUF_WORK_DIR"])

    # Display menu with local models and option to download new
    selected_path = display_local_gguf_menu(local_models)

    if selected_path is None:
        return None
    elif selected_path == "download_new":
        # User wants to download new model - need to get config files first
        print("\n🤗 First, we need to download model configuration files...")
        config_dir = download_model_configs(base_work_dir)

        if not config_dir:
            print(red_text("❌ Failed to download model configuration. Cannot proceed."))
            return None

        print(green_text("✓ Model configuration downloaded successfully"))

        # Get the base model repository name for GGUF download
        base_model_repo = None
        if os.environ.get("NIM_MODEL_DIR"):
            model_dir_name = os.path.basename(os.environ["NIM_MODEL_DIR"])
            base_model_repo = model_dir_name.replace("-", "/")

        # Now download GGUF files
        return download_gguf_files(base_model_repo or "", base_work_dir)
    else:
        # User selected a local model
        print(f"Using local model at: {selected_path}")

        # Check if config files exist in the selected directory
        config_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]
        missing_configs = []

        for config_file in config_files:
            if not os.path.exists(os.path.join(selected_path, config_file)):
                missing_configs.append(config_file)

        if missing_configs:
            print(yellow_text(f"⚠️  Missing config files in selected directory: {', '.join(missing_configs)}"))

            # If we don't have a base model config directory, we need to download configs
            if not os.environ.get("NIM_MODEL_DIR") or not os.path.exists(os.environ["NIM_MODEL_DIR"]):
                print("📥 Need to download configuration files first...")
                config_dir = download_model_configs(base_work_dir)

                if not config_dir:
                    print(red_text("❌ Failed to download model configuration."))
                    return None

            # Copy config files from the base model directory
            if os.environ.get("NIM_MODEL_DIR") and os.path.exists(os.environ["NIM_MODEL_DIR"]):
                print("Copying configuration files...")
                try:
                    for config_file in missing_configs:
                        src_path = os.path.join(os.environ["NIM_MODEL_DIR"], config_file)
                        if os.path.exists(src_path):
                            dst_path = os.path.join(selected_path, config_file)
                            shutil.copy2(src_path, dst_path)
                            print(green_text(f"✓ Copied {config_file}"))
                        else:
                            print(yellow_text(f"⚠️  {config_file} not found in base model directory"))

                    print(green_text("✓ Configuration files copied successfully"))
                except Exception as e:
                    print(red_text(f"❌ Error copying config files: {e}"))
                    return None
        else:
            print(green_text("✓ All configuration files present"))

        return selected_path


def download_gguf_files(model_repo: str, base_work_dir: str) -> str:
    """
    Download GGUF files from a HuggingFace model repository.

    Args:
        model_repo (str): HuggingFace model repository (profile/model-name)
        base_work_dir (str): Base working directory for downloads

    Returns:
        str: Path to the final model directory, or None if failed
    """

    while True:
        # Get GGUF model repository from user
        gguf_repo = input(f"\nEnter GGUF model repository (format: profile/model-name): ").strip()

        if '/' not in gguf_repo:
            print(red_text("Please use the format: profile/model-name (e.g., bartowski/Llama-3.2-3B-Instruct-GGUF)"))
            continue

        print(f"\nScanning {gguf_repo} for GGUF files...")

        try:
            # List repository files
            try:
                repo_files = list_repo_files(
                    repo_id=gguf_repo,
                    token=os.getenv("HF_TOKEN")
                )
                print(green_text(f"✓ Repository {gguf_repo} found"))
            except RepositoryNotFoundError:
                print(red_text(f"❌ Repository '{gguf_repo}' not found. Please check the model name and try again."))
                continue
            except Exception as e:
                print(red_text(f"❌ Error accessing repository: {e}"))
                continue

            # Extract and organize GGUF files
            gguf_groups = extract_gguf_files(repo_files)

            if not gguf_groups:
                print(red_text(f"\n❌ No GGUF files found in repository: {gguf_repo}"))
                print(yellow_text("Available files in repository:"))
                for f in sorted(repo_files)[:20]:
                    print(f"  {f}")
                if len(repo_files) > 20:
                    print(f"  ... and {len(repo_files) - 20} more files")

                retry = input("Would you like to try a different repository? (y/n): ").strip().lower()
                if retry != 'y':
                    return None
                continue

            # Display menu and get selection
            selected_display, selected_files = display_gguf_menu(gguf_groups)

            if not selected_display:
                return None

            # Create directory structure: profile-model-name/selected-file-name/
            model_dir_name = gguf_repo.replace("/", "-")
            # Remove .gguf extension for directory name
            file_dir_name = selected_display.replace(".gguf", "")
            final_model_dir = os.path.join(os.environ["GGUF_WORK_DIR"], model_dir_name, file_dir_name)

            os.makedirs(final_model_dir, exist_ok=True)
            print(f"Created directory: {final_model_dir}")

            # Download selected GGUF files
            print(f"\nDownloading {len(selected_files)} GGUF file(s)...")
            downloaded_files = []

            for file_path in selected_files:
                try:
                    print(f"Downloading {file_path}...")
                    downloaded_path = hf_hub_download(
                        repo_id=gguf_repo,
                        filename=file_path,
                        local_dir=final_model_dir,
                        token=os.getenv("HF_TOKEN")
                    )
                    downloaded_files.append(file_path)
                    print(green_text(f"✓ {os.path.basename(file_path)} downloaded successfully"))

                except Exception as e:
                    print(red_text(f"❌ Error downloading {file_path}: {e}"))
                    retry = input("Would you like to try a different model? (y/n): ").strip().lower()
                    if retry != 'y':
                        return None
                    break
            else:
                # All GGUF files downloaded successfully
                print(green_text(f"\n✓ Successfully downloaded {len(downloaded_files)} GGUF file(s)"))

                # Copy config files from the base model directory to final directory
                if os.environ.get("NIM_MODEL_DIR") and os.path.exists(os.environ["NIM_MODEL_DIR"]):
                    print("Copying configuration files...")
                    try:
                        for file_name in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
                            src_path = os.path.join(os.environ["NIM_MODEL_DIR"], file_name)
                            if os.path.exists(src_path):
                                dst_path = os.path.join(final_model_dir, file_name)
                                shutil.copy2(src_path, dst_path)
                                print(green_text(f"✓ Copied {file_name}"))

                        print(green_text("✓ Configuration files copied successfully"))
                    except Exception as e:
                        print(red_text(f"❌ Error copying config files: {e}"))
                        return None
                else:
                    print(yellow_text("⚠️  No config files found to copy. Make sure to run download_model_configs first."))

                print(f"Final model directory: {final_model_dir}")
                return final_model_dir

        except KeyboardInterrupt:
            print(red_text("\n\nOperation cancelled by user."))
            break
        except Exception as e:
            print(red_text(f"❌ Unexpected error: {e}"))
            retry = input("Would you like to try a different repository? (y/n): ").strip().lower()
            if retry != 'y':
                break

    return None


# Example usage
if __name__ == "__main__":
    base_work_dir = os.path.abspath(".")

    # First download config files
    config_dir = download_model_configs(base_work_dir)

    if config_dir:
        print(green_text(f"\nConfig files ready at: {config_dir}"))

        # Then download GGUF files
        model_repo = input("Enter the base model repository name (used for config files): ").strip()
        final_dir = download_or_select_gguf_model(model_repo, base_work_dir)

        if final_dir:
            print(green_text(f"\nComplete model ready at: {final_dir}"))
        else:
            print(red_text("\nFailed to download GGUF files."))
    else:
        print(red_text("\nNo config files downloaded."))
