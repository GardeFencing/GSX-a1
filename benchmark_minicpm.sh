#!/bin/bash

# Define paths to models and video file
MODEL_Q2="./MiniCPM-V-2_6-gguf/ggml-model-Q2_K.gguf"
MODEL_Q4="./MiniCPM-V-2_6-gguf/ggml-model-Q4_K_M.gguf"
MMPROJ="./MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf"
VIDEO="./fencing_trimmed.mp4"
MINICPMV_PATH="./llama.cpp/llama-minicpmv-cli"
PROMPT="What is happening in this video?"
RUNS=3  # Number of times to run each model

# Function to run a model multiple times and compute the average real time
benchmark_model() {
    local model_name=$1
    local model_path=$2
    local total_time=0
    local output_file="benchmark_${model_name}.txt"

    echo "Running benchmark for $model_name ($RUNS runs)..."
    echo "" > "$output_file"

    for ((i=1; i<=RUNS; i++)); do
        echo "Run #$i for $model_name..."
        real_time=$({ time $MINICPMV_PATH \
            -m "$model_path" \
            --mmproj "$MMPROJ" \
            -c 4096 \
            --temp 0.7 \
            --top-p 0.8 \
            --top-k 100 \
            --repeat-penalty 1.05 \
            --video "$VIDEO" \
            -p "$PROMPT"; } 2>&1 | grep "real" | awk '{print $2}')
        
        echo "Run #$i: $real_time" | tee -a "$output_file"
        
        # Convert time format (min:sec) to seconds for calculation
        min=$(echo $real_time | awk -F'm' '{print $1}')
        sec=$(echo $real_time | awk -F'm' '{print $2}' | sed 's/s//')
        total_time=$(echo "$total_time + ($min * 60) + $sec" | bc)
    done

    # Calculate average time
    avg_time=$(echo "scale=2; $total_time / $RUNS" | bc)
    echo "Average Real Time for $model_name: $avg_time seconds" | tee -a "$output_file"
    
    # Return the average
    echo $avg_time
}

# Run benchmarks and store averages
avg_q2=$(benchmark_model "Q2_K" "$MODEL_Q2")
avg_q4=$(benchmark_model "Q4_K_M" "$MODEL_Q4")

# Print comparison results
echo
echo "Benchmark Results:"
echo "------------------"
echo "Q2_K Average Real Time: $avg_q2 seconds"
echo "Q4_K_M Average Real Time: $avg_q4 seconds"

# Determine the faster model
echo
echo "Final Comparison:"
if (( $(echo "$avg_q2 < $avg_q4" | bc -l) )); then
    echo "Q2_K is faster!"
else
    echo "Q4_K_M is faster!"
fi

