#!/bin/bash
# Copy our fixed file over the broken one
cp /opt/start/utils.py /opt/nim/llm/nim_llm_sdk/hub/

# Run the original entrypoint
exec /opt/nvidia/nvidia_entrypoint.sh /opt/nim/start_server.sh
