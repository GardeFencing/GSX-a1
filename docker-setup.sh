#!/bin/sh

if [ ! -f /app/models/MiniCPM-V-2_6-gguf/$MODEL ]; then
    echo "Downloading $MODEL file"
    exit 1
    wget https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/$MODEL -O MiniCPM-V-2_6-gguf/$MODEL
fi

if [ ! -f /app/models/MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf ]; then
    echo "Downloading mmproj-model-f16.gguf file"
    exit 1
    wget https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/mmproj-model-f16.gguf -O MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf
fi

gunicorn web_app:app \
    --workers 2 \
    --threads 4 \
    --worker-class=gthread \
    -b 0.0.0.0:8080 \
    --timeout 3600 \
    --graceful-timeout 300 \
    --keep-alive 120 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --log-level debug \
    --capture-output \
    --enable-stdio-inheritance \
    --access-logfile - \
    --error-logfile -