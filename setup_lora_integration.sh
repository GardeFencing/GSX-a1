#!/bin/bash
# Script to integrate LoRA fine-tuned model with RAG for video analysis

set -e

echo "===== Setting up LoRA and RAG integration for fencing video analysis ====="

# Create necessary directories
echo "Checking required directories..."
mkdir -p MiniCPM-V/finetune/output/output_fencing_lora/merged_model
mkdir -p GardeRAG/model_cache
mkdir -p dataset/images

# Install additional dependencies if needed
echo "Installing dependencies..."
pip install PyPDF2 nltk redis python-dotenv Flask pillow opencv-python langchain-huggingface torch transformers datasets peft deepspeed

# Extract fencing knowledge for RAG
echo "Extracting fencing knowledge from PDFs for RAG enhancement..."
if [ -d "GardeRAG/books" ]; then
    python extract_fencing_knowledge.py --min_length 150 --max_length 500
    echo "Successfully extracted fencing knowledge"
else
    echo "WARNING: GardeRAG/books directory not found, RAG enhancement will be limited"
    mkdir -p GardeRAG/books
fi

# Update environment variables
echo "Updating environment variables..."
cat >> .env << EOF

# Model Settings for Video Analysis
MODEL=ggml-model-Q4_K_M.gguf
CONTEXT_SIZE=4096
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=100
REPEAT_PENALTY=1.05

# LoRA Settings
USE_LORA=true
LORA_MODEL_PATH=MiniCPM-V/finetune/output/output_fencing_lora/merged_model
EOF

# Data preparation check
echo "Checking for training data..."
TRAINING_DATA_PATH="MiniCPM-V/finetune/fencing_book_training_data.json"
if [ ! -f "$TRAINING_DATA_PATH" ]; then
    echo "Preparing training data from fencing PDFs and images..."
    # Extract frames from videos if needed
    if [ ! -d "temp_frames" ] || [ $(ls -1 temp_frames/*.jpg 2>/dev/null | wc -l) -lt 10 ]; then
        echo "Extracting frames from fencing videos for training..."
        mkdir -p temp_frames
        
        # Try to extract frames from videos if available
        if [ -f "fencing.mp4" ] || [ -f "fencing_trimmed.mp4" ]; then
            VIDEO_PATH=$([ -f "fencing.mp4" ] && echo "fencing.mp4" || echo "fencing_trimmed.mp4")
            ffmpeg -i "$VIDEO_PATH" -vf "fps=1" -q:v 2 temp_frames/frame_%03d.jpg
            echo "Extracted frames from $VIDEO_PATH"
        else
            # Generate placeholder images if no video available
            echo "No fencing videos found. Generating placeholder images..."
            python -c "
import os
import numpy as np
from PIL import Image

# Create some placeholder images for training
os.makedirs('temp_frames', exist_ok=True)
for i in range(30):
    img = Image.new('RGB', (640, 480), color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
    img.save(f'temp_frames/frame_{i:03d}.jpg')
    print(f'Created temp_frames/frame_{i:03d}.jpg')
"
        fi
    fi
    
    # Generate training data
    python prepare_fencing_lora_dataset.py \
      --image_dir temp_frames \
      --books_dir GardeRAG/books \
      --output_file "$TRAINING_DATA_PATH" \
      --num_samples 100
    
    echo "Training data prepared at $TRAINING_DATA_PATH"
else
    echo "Training data already exists at $TRAINING_DATA_PATH"
fi

# Verify llama.cpp has video support 
echo "Checking llama.cpp video support..."
if [ -d "llama.cpp" ]; then
    if [ -f "llama.cpp/llama-minicpmv-cli" ]; then
        if llama.cpp/llama-minicpmv-cli --help 2>&1 | grep -q -- "--video"; then
            echo "✅ llama.cpp has video support"
        else
            echo "⚠️ llama-minicpmv-cli exists but may not have video support"
            echo "Rebuilding llama.cpp with video support..."
            
            cd llama.cpp
            make clean
            LLAMA_FFMPEG=1 make -j
            cd ..
            
            if llama.cpp/llama-minicpmv-cli --help 2>&1 | grep -q -- "--video"; then
                echo "✅ Successfully rebuilt llama.cpp with video support"
            else
                echo "❌ Failed to build llama.cpp with video support"
                echo "Please ensure ffmpeg development libraries are installed and rebuild manually:"
                echo "cd llama.cpp && LLAMA_FFMPEG=1 make -j"
                exit 1
            fi
        fi
    else
        echo "❌ llama-minicpmv-cli not found in llama.cpp directory"
        echo "Make sure to build llama.cpp with LLAMA_FFMPEG=1 flag"
        exit 1
    fi
else
    echo "❌ llama.cpp directory not found"
    echo "Please clone the repository and build with video support:"
    echo "git clone -b minicpmv-main https://github.com/OpenBMB/llama.cpp.git"
    echo "cd llama.cpp && LLAMA_FFMPEG=1 make -j"
    exit 1
fi

# Test if LoRA model exists
LORA_MODEL_EXISTS=false
if [ -d "MiniCPM-V/finetune/output/output_fencing_lora/merged_model" ]; then
    # Check for model files
    for file in model.bin pytorch_model.bin model.safetensors consolidated.00.pth; do
        if [ -f "MiniCPM-V/finetune/output/output_fencing_lora/merged_model/$file" ]; then
            LORA_MODEL_EXISTS=true
            break
        fi
    done
fi

# Run the LoRA fine-tuning script if needed
if [ "$LORA_MODEL_EXISTS" = false ]; then
    echo "No merged LoRA model found."
    echo "Do you want to run LoRA fine-tuning now? (y/n)"
    read -r run_finetuning

    if [[ "$run_finetuning" == "y" ]]; then
        # Change to the MiniCPM-V directory
        cd MiniCPM-V/finetune
        
        # Create lora fine-tuning script with improved parameters based on article
        cat > ec2_finetune_lora_custom.sh << 'EOF'
#!/bin/bash
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-6001}

MODEL=${MODEL_PATH:-"openbmb/MiniCPM-V-2_6"}
DATA=${DATA_PATH:-"./fencing_book_training_data.json"}
EVAL_DATA=${EVAL_DATA_PATH:-"./fencing_book_training_data.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/output_fencing_lora"}
LLM_TYPE="qwen2"

# Configure distributed training
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Run fine-tuning
torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --fp16 true \
    --fp16_full_eval true \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm false \
    --use_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --model_max_length 2048 \
    --max_slice_nums 9 \
    --max_steps 1000 \
    --eval_steps 100 \
    --output_dir $OUTPUT_DIR \
    --logging_dir $OUTPUT_DIR \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero3.json \
    --report_to "tensorboard"
EOF
        
        chmod +x ec2_finetune_lora_custom.sh
        
        # Run the fine-tuning
        echo "Starting LoRA fine-tuning..."
        bash ec2_finetune_lora_custom.sh
        
        # Merge the LoRA model with the base model
        echo "Merging LoRA weights with base model..."
        MODEL_PATH=${MODEL_PATH:-"openbmb/MiniCPM-V-2_6"}
        OUTPUT_DIR=${OUTPUT_DIR:-"./output/output_fencing_lora"}
        MAX_STEPS=${MAX_STEPS:-1000}
        
        python merge_lora.py \
          --model_type "$MODEL_PATH" \
          --lora_path "$OUTPUT_DIR/checkpoint-$MAX_STEPS" \
          --merge_path "$OUTPUT_DIR/merged_model"
        
        cd ../..
        
        echo "LoRA fine-tuning and merging completed"
    else
        echo "Skipping LoRA fine-tuning"
    fi
else
    echo "✅ Merged LoRA model found"
fi

# Test if the LoRA model can process video
echo "Testing LoRA model with video analysis..."
if [ -f "fencing_trimmed.mp4" ] || [ -f "fencing.mp4" ]; then
    TEST_VIDEO=$([ -f "fencing_trimmed.mp4" ] && echo "fencing_trimmed.mp4" || echo "fencing.mp4")
    
    # Make test_lora_video.py executable
    chmod +x test_lora_video.py
    
    # Run test with automatic .env update
    python test_lora_video.py --video "$TEST_VIDEO" --update_env
    
    # Check the exit code
    if [ $? -ne 0 ]; then
        echo "⚠️ LoRA model test failed, but the script will continue"
        echo "The .env file has been updated with the appropriate setting"
    else
        echo "✅ LoRA model test successful"
    fi
else
    echo "⚠️ No test video found. Skipping video compatibility test."
    echo "You can run video compatibility test manually with:"
    echo "python test_lora_video.py --video your_video.mp4 --update_env"
fi

# Create a simple test script
echo "Creating test script for quick verification..."
cat > test_fencing_analysis.sh << 'EOF'
#!/bin/bash
# Simple test script for fencing video analysis

if [ ! -f ".env" ]; then
    echo "Error: .env file not found"
    exit 1
fi

# Load environment variables
source .env

# Check if a video file is provided
if [ $# -lt 1 ]; then
    # Use default video if available
    if [ -f "fencing_trimmed.mp4" ]; then
        VIDEO_FILE="fencing_trimmed.mp4"
    elif [ -f "fencing.mp4" ]; then
        VIDEO_FILE="fencing.mp4"
    else
        echo "Error: No video file provided and no default video found"
        echo "Usage: $0 <video_file>"
        exit 1
    fi
else
    VIDEO_FILE="$1"
fi

# Verify the video file exists
if [ ! -f "$VIDEO_FILE" ]; then
    echo "Error: Video file not found: $VIDEO_FILE"
    exit 1
fi

# Get model path
if [ "${USE_LORA}" = "true" ]; then
    MODEL_DIR="$LORA_MODEL_PATH"
    echo "Using LoRA fine-tuned model from $MODEL_DIR"
    
    # Find model file
    MODEL_FILE=""
    for file in model.bin pytorch_model.bin model.safetensors consolidated.00.pth; do
        if [ -f "$MODEL_DIR/$file" ]; then
            MODEL_FILE="$MODEL_DIR/$file"
            break
        fi
    done
    
    # Find mmproj file
    MMPROJ_FILE=""
    for file in mmproj-model-f16.gguf mm_projector.bin mm_projector/model.safetensors; do
        if [ -f "$MODEL_DIR/$file" ]; then
            MMPROJ_FILE="$MODEL_DIR/$file"
            break
        fi
    done
    
    if [ -z "$MODEL_FILE" ] || [ -z "$MMPROJ_FILE" ]; then
        echo "Error: Could not find model files in $MODEL_DIR"
        exit 1
    fi
else
    MODEL_FILE="models/MiniCPM-V-2_6-gguf/$MODEL"
    MMPROJ_FILE="models/MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf"
    echo "Using base model"
fi

# Run analysis
echo "Analyzing video: $VIDEO_FILE"
cd llama.cpp
./llama-minicpmv-cli \
    -m "../$MODEL_FILE" \
    --mmproj "../$MMPROJ_FILE" \
    -c ${CONTEXT_SIZE:-4096} \
    --temp ${TEMPERATURE:-0.7} \
    --top-p ${TOP_P:-0.8} \
    --top-k ${TOP_K:-100} \
    --repeat-penalty ${REPEAT_PENALTY:-1.05} \
    --video "../$VIDEO_FILE" \
    -p "You are an expert fencing coach analyzing this video. Describe the fencing techniques shown, the quality of execution, and provide coaching advice."

# Return to original directory
cd ..
EOF

chmod +x test_fencing_analysis.sh

echo "===== Setup completed successfully! ====="
echo ""
echo "You can now:"
echo "1. Run the web app: python web_app.py"
echo "2. Quick test the model: ./test_fencing_analysis.sh [video_file]"
echo "3. For more details, see LORA_RAG_INTEGRATION.md" 