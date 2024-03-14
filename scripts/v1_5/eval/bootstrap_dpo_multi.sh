#!/bin/bash

# Check if three arguments are passed
if [ "$#" -ne 10 ]; then
    echo "Usage: $0 <model_base> <model_path> <question_path> <base_answer_path> <image_folder> <num_samples> <subset> <N> <temperature> <start_gpu>"
    exit 1
fi

# Assign the command line arguments to variables
model_base=$1
model_path=$2
question_path=$3
base_answer_path=$4
image_folder=$5
num_samples=$6
subset=$7
N=$8
temperature=$9
GS=${10}

# Loop over each chunk/process
for (( chunk_id=0; chunk_id<N; chunk_id++ ))
do
    # Define the answer path for each chunk
    answer_path="${base_answer_path}_${chunk_id}.jsonl"
    if [ -f "$answer_path" ]; then
        rm "$answer_path"
    fi
    gpu=$(expr $chunk_id + $GS)
    echo "$gpu"
    # Run the Python program in the background
    CUDA_VISIBLE_DEVICES="$gpu" python llava/eval/bootstrap_dpo.py --model-base "$model_base" --model-path "$model_path" --question-file "$question_path" --answers-file "$answer_path" --num-chunks "$N" --chunk-idx "$chunk_id" --image-folder "$image_folder" --num-samples "$num_samples" --subset "$subset" --temperature "$temperature" &

    # Uncomment below if you need a slight delay between starting each process
    # sleep 0.1
done

# Wait for all background processes to finish
wait

merged_file="${base_answer_path}_merged.jsonl"
if [ -f "$merged_file" ]; then
    rm "$merged_file"
fi
# Merge all the JSONL files into one
#cat "${base_answer_path}"_*.jsonl > "${base_answer_path}_merged.jsonl"
for ((i=0; i<N; i++)); do
  input_file="${base_answer_path}_${i}.jsonl"
  cat "$input_file" >> "${base_answer_path}_merged.jsonl"
done
# remove the unmerged files
for (( chunk_id=0; chunk_id<N; chunk_id++ ))
do
    # Define the answer path for each chunk
    answer_path="${base_answer_path}_${chunk_id}.jsonl"
    if [ -f "$answer_path" ]; then
        rm "$answer_path"
    fi
done
