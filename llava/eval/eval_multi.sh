#!/bin/bash

# Check if four arguments are passed
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <model_path> <answers_root> <num_chunks> <temperature>"
  exit 1
fi

scenes=("01-Illegal_Activitiy" "02-HateSpeech" "03-Malware_Generation" "04-Physical_Harm" "05-EconomicHarm" "06-Fraud" "07-Sex" "08-Political_Lobbying" "09-Privacy_Violence" "10-Legal_Opinion" "11-Financial_Advice" "12-Health_Consultation" "13-Gov_Decision")

# Assign the command line arguments to variables
model_path=$1
answers_root=$2
N=$3
temperature=$4

# Check if the answers_root directory exists
if [ ! -d "$answers_root" ]; then
  # Directory does not exist, so create it
  mkdir "$answers_root"
fi

for scene in "${scenes[@]}"; do
  answer_scene_path="${answers_root}/${scene}"
  if [ ! -d "$answer_scene_path" ]; then
    # Directory does not exist, so create it
    mkdir "$answer_scene_path"
  fi

  # Loop over each chunk/process
  for ((chunk_id = 0; chunk_id < N; chunk_id++)); do
    # Define the answer path for each chunk
    answer_path="${answer_scene_path}/${chunk_id}.json"
    if [ -f "$answer_path" ]; then
      rm "$answer_path"
    fi

    # Run the Python program in the background
    CUDA_VISIBLE_DEVICES="$chunk_id" python llava/eval/robustness_eval.py --model-path "$model_path" --scene "$scene" --answers_file "$answer_path" --num-chunks "$N" --chunk-idx "$chunk_id" --temperature "$temperature" &

    # Uncomment below if you need a slight delay between starting each process
    # sleep 0.1
  done

  # Wait for all background processes to finish
  wait
  cd $answer_scene_path
  merged_file="merged.json"
  if [ -f "$merged_file" ]; then
    rm "$merged_file"
  fi

  # Merge all the JSON files into one
  python ~/polite_llava/scripts/concatenate_json.py *.json
  cd ~/polite_llava
  # Remove the unmerged files
  for ((chunk_id = 0; chunk_id < N; chunk_id++)); do
    answer_path="${answer_scene_path}/${chunk_id}.json"
    if [ -f "$answer_path" ]; then
      rm "$answer_path"
    fi
  done
done
