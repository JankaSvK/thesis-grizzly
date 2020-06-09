#!/bin/bash

echo "$@" >> executed_preprocessings.txt

ssh_output="$(ssh -t gpulab "sbatch scripts/preprocess_images_kwargs.sh $@")"
batch_job_id=$(echo "$ssh_output" | grep -o -E '[0-9]+')

echo "Batch Job ID: $batch_job_id"

while true
do
	squeue_output="$(ssh -t gpulab squeue 2>/dev/null)"
	test_output="$(echo "$squeue_output" | grep "$batch_job_id")"
	
	if [ -z "$test_output" ] 
	then
		break
	fi	
	sleep 10
done

