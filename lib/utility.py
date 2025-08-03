import requests
import time
import docker
import os
from .colors import *

def check_service_ready_from_logs(container_name, print_logs=False, timeout=600):
    """
    Check if NIM service is ready by monitoring Docker logs for 'Application startup complete' message.

    Args:
        container_name (str): Name of the Docker container
        print_logs (bool): Whether to print logs while monitoring (default: False)
        timeout (int): Maximum time to wait in seconds (default: 600)

    Returns:
        bool: True if service is ready, False if timeout reached
    """
    print(yellow_text("Waiting for NIM service to start..."))
    start_time = time.time()

    try:
        client = docker.from_env()
        container = client.containers.get(container_name)

        # Stream logs in real-time using the blocking generator
        log_buffer = ""
        for log_chunk in container.logs(stdout=True, stderr=True, follow=True, stream=True):
            # Check timeout
            if time.time() - start_time > timeout:
                print(red_text(f"Timeout reached ({timeout}s). Service may not have started properly."))
                return False

            # Decode chunk and add to buffer
            chunk = log_chunk.decode('utf-8', errors='ignore')
            log_buffer += chunk

            # Process complete lines
            while '\n' in log_buffer:
                line, log_buffer = log_buffer.split('\n', 1)
                line = line.strip()

                if print_logs and line:
                    print(f"[LOG] {line}")

                # Check for startup complete message
                if "Application startup complete" in line:
                    print(green_text("Application startup complete! Service is ready."))
                    return True

    except Exception as e:
        print(red_text(f"Error: {e}"))
        return False

    print(red_text(f"Timeout reached ({timeout}s). Service may not have started properly."))
    return False

def check_service_ready():
    """Fallback health check using HTTP endpoint"""
    url = 'http://localhost:8000/v1/health/ready'
    print("Checking service health endpoint...")

    while True:
        try:
            response = requests.get(url, headers={'accept': 'application/json'})
            if response.status_code == 200 and response.json().get("message") == "Service is ready.":
                print(green_text("Service ready!"))
                break
        except requests.ConnectionError:
            pass
        print("⏳ Still starting...")
        time.sleep(30)

def generate_text(model, prompt, max_tokens=1000, temperature=0.7):
    """Generate text using the NIM service"""
    try:
        response = requests.post(
            f"http://localhost:8000/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(red_text(f"Error making request: {e}"))
        return None
