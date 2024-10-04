#!/bin/sh

# Base directory from the input argument
BASE_DIR="$1"

# Command to run after setting up the environment
COMMAND="$2"

# Path to your virtual environment
VIRTUAL_ENV="$BASE_DIR/.env"

# Check if the virtual environment exists
if [ ! -d "$VIRTUAL_ENV" ]; then
    echo "Virtual environment not found at $VIRTUAL_ENV"
    exit 1
fi

# Set the PATH to include the virtual environment's bin directory
_OLD_VIRTUAL_PATH="$PATH"
PATH="$VIRTUAL_ENV/bin:$PATH"
export PATH

# Unset PYTHONHOME if set
if [ -n "${PYTHONHOME:-}" ]; then
    unset PYTHONHOME
fi

# Run the command if provided
if [ -n "$COMMAND" ]; then
    echo "Running command: $COMMAND"
    exec $COMMAND
else
    echo "No command provided to run."
    exit 1
fi
