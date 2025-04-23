#!/bin/sh
set -e

# Execute the command passed to the entrypoint (e.g., uvicorn)
echo "Executing command: $@"
exec "$@" 