#!/bin/bash

while IFS= read -r line
do 
	sh run_annotate_images_on_gpulab.sh $line
done
