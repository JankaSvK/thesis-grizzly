#!/bin/bash

while read line
do 
	./run_preprocessing_images_on_gpulab.sh $line < /dev/null
done
