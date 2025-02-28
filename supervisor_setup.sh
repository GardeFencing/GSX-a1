#!/bin/bash

if [ -z "$GSX_A1_PATH" ]; then
  echo "Error: GSX_A1_PATH is not set."
  exit 1
fi

# Use sed to replace the placeholder
sudo sed "s|__GSX_A1_PATH__|$GSX_A1_PATH|g" vidclip.conf > /etc/supervisor/conf.d/vidclip.conf

# Reload Supervisor configuration
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart vidclip
