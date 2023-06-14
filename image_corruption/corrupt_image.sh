#!/bin/bash
# modify the BASE_DATASET_FOLDER and DATASET to your own path
BASE_DATASET_FOLDER=/home/.data/datasets
DATASET=nlvr/images_test
echo "FOLDER = ${BASE_DATASET_FOLDER}/${DATASET}"

python corruption.py \
  --method blank_image \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 1 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method gaussian_noise \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method shot_noise \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method impulse_noise \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method defocus_blur \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method glass_blur \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method motion_blur \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method zoom_blur \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method gaussian_blur \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method snow \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method frost \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method fog \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method brightness \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method contrast \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method elastic_transform \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method pixelate \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method jpeg_compression \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method speckle_noise \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method gaussian_blur \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method spatter \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
python corruption.py \
  --method saturate \
  --image_root_folder ${BASE_DATASET_FOLDER}/${DATASET} \
  --severity_begin 1 \
  --severity_end 5 \
  --batch_size 100 \
  --num_workers 10
