from huggingface_hub import hf_hub_download, list_repo_files, RepositoryNotFoundError
from .colors import green_text, red_text, yellow_text
import os


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
        config_temp_dir = os.path.join(base_work_dir, "gguf_config_temp")
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
        for required_file in required_files:
            # Look for the file anywhere in the repository structure
            matching_files = [f for f in repo_files if f.endswith(required_file)]

            if matching_files:
                # If multiple matches, prefer root directory, then shortest path
                best_match = min(matching_files, key=lambda x: (x.count('/'), len(x)))
                found_files[required_file] = best_match
                print(green_text(f"Found {required_file} at: {best_match}"))
            else:
                print(red_text(f"❌ {required_file} not found anywhere in repository"))

        try:
            # First, check if repository exists by trying to list files
            try:
                repo_files = list_repo_files(
                    repo_id=model_repo,
                    token=os.getenv("HF_TOKEN")
                )
                print(green_text(f"✓ Repository {model_repo} found"))
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
                print(green_text(f"Files saved to: {config_temp_dir}"))
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


# Example usage
if __name__ == "__main__":
    base_work_dir = os.path.abspath(".")
    config_dir = download_model_configs(base_work_dir)

    if config_dir:
        print(green_text(f"\nConfig files ready at: {config_dir}"))
    else:
        print(red_text("\nNo config files downloaded."))
