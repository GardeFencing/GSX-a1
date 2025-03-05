#!/bin/bash

# Get absolute paths dynamically
GSX_A1_PATH=$(pwd)
NODE_PATH=$(command -v node || echo "/usr/bin/node")

# Replace placeholders dynamically
sudo sed "s|__GSX_A1_PATH__|$GSX_A1_PATH|g" vidclip.conf > /etc/supervisor/conf.d/vidclip.conf
sudo sed -e "s|__GSX_A1_PATH__|$GSX_A1_PATH|g" -e "s|__NODE_PATH__|$NODE_PATH|g" convert.conf | sudo tee /etc/supervisor/conf.d/convert.conf > /dev/null

# Reload Supervisor configuration
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart vidclip
sudo supervisorctl restart convert
sudo supervisorctl status vidclip
sudo supervisorctl status convert