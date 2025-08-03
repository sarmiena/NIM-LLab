#!/usr/bin/env python3
"""
NGC NIM Version Selector
A command-line tool to fetch available NIM versions and pull a selected Docker image.
"""

import subprocess
import json
import sys
import docker
from .colors import *
from typing import List, Dict, Any
from .run_command import run_command

def get_nim_info() -> Dict[str, Any]:
    """Fetch NIM registry information using ngc command."""
    print("Fetching NIM registry information...")
    
    result = run_command(['ngc', 'registry', 'image', 'info', 'nim/nvidia/llm-nim'])
    
    if not result['success']:
        print(red_text(f"Error running ngc command: {result.get('stderr', result.get('error', 'Unknown error'))}"))
        sys.exit(1)
    
    try:
        # Parse the JSON output
        nim_info = json.loads(result['stdout'])
        return nim_info
    except json.JSONDecodeError as e:
        print(red_text(f"Error parsing JSON response: {e}"))
        sys.exit(1)


def extract_and_sort_versions(nim_info: Dict[str, Any]) -> List[str]:
    """Extract tags, remove 'latest', and sort versions."""
    tags = nim_info.get('tags', [])
    
    # Remove 'latest' from the list
    filtered_tags = [tag for tag in tags if tag != 'latest']
    
    # Sort versions - put numeric versions first, then others
    def version_sort_key(version: str) -> tuple:
        try:
            # Try to parse as version numbers (e.g., "1.12.0" -> [1, 12, 0])
            parts = [int(x) for x in version.split('.')]
            return (0, parts)  # 0 for numeric versions (higher priority)
        except ValueError:
            # For non-numeric versions, sort alphabetically
            return (1, version)  # 1 for non-numeric versions (lower priority)
    
    sorted_tags = sorted(filtered_tags, key=version_sort_key, reverse=True)
    return sorted_tags


def display_versions_menu(versions: List[str]) -> str:
    """Display available versions and get user selection."""
    print("\nAvailable NIM versions:")
    print("-" * 30)
    
    for i, version in enumerate(versions, 1):
        print(f"{i:2d}. {version}")
    
    print("-" * 30)
    
    while True:
        try:
            choice = input(f"Select a version (1-{len(versions)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Exiting...")
                sys.exit(0)
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(versions):
                selected_version = versions[choice_num - 1]
                print(f"Selected version: {selected_version}")
                return selected_version
            else:
                print(f"Please enter a number between 1 and {len(versions)}")
                
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


def pull_docker_image(version: str) -> None:
    """Pull the selected Docker image."""
    image_url = f"nvcr.io/nim/nvidia/llm-nim:{version}"
    client = docker.from_env()

    try:
        client.images.get(image_url)
        print(green_text(f"Using existing docker image {image_url}"))
        return
    except docker.errors.ImageNotFound:
        print(f"Pulling Docker image: {image_url}. This may take several minutes...")

    result = run_command(['docker', 'pull', image_url])
    
    if result['success']:
        print(green_text(f"Successfully pulled {image_url}"))
    else:
        print(red_text(f"Error pulling Docker image: {result.get('stderr', result.get('error', 'Unknown error'))}"))
        sys.exit(1)


def select_nim_version(auto_pull: bool = True) -> str:
    """
    Main function to select a NIM version.
    
    Args:
        auto_pull (bool): If True, automatically pulls the Docker image. 
                         If False, only returns the selected version.
    
    Returns:
        str: The selected version tag
    """
    print("NGC NIM Version Selector")
    print("=" * 30)
    
    # Step 1: Get NIM registry information
    nim_info = get_nim_info()
    
    # Step 2 & 3: Extract and sort versions, remove 'latest'
    versions = extract_and_sort_versions(nim_info)
    
    if not versions:
        print(red_text("No versions found!"))
        sys.exit(1)
    
    # Step 4: Ask user to pick a version
    selected_version = display_versions_menu(versions)
    
    # Step 5: Optionally pull the Docker image
    if auto_pull:
        pull_docker_image(selected_version)
    
    return selected_version


def main():
    """Command-line entry point."""
    selected_version = select_nim_version(auto_pull=True)
    print(f"Selected version: {selected_version}")


if __name__ == "__main__":
    main()
