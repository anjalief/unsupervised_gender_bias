#!/bin/bash
#################################################################################
# This scripts loads the pre-trained models and pre-processed data and outputs
# results over test sets
# Models were trained with CUDA version 9.0 and TITAN X GPU
#################################################################################

source /projects/tir1/users/anjalief/anaconda3/bin/activate test_env

################################## TO FILL IN ############################################################
TOP_DIR="/projects/tir3/users/anjalief/adversarial_gender/unsupervised_gender_bias_models" # the path to unzipped tarball of saved models

# SUFFIX="facebook_wiki"  # Flip comment to change which data set to run
SUFFIX="facebook_congress"
####################################################################################################


# Paths to data
DATA_DIR="${TOP_DIR}/${SUFFIX}"

# Intermediate suffixes
MATCHED_SUFFIX="matched_${SUFFIX}"
SUBS_SUFFIX="subs_name2"
EXTRA_SUFFIX="withtopics"
NO_SUFFIX="notopics"


##################################### Run Models (write test predictions to files) ######################################################
# Matched, no demotion
python train.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${MATCHED_SUFFIX}.${NO_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.${SUBS_SUFFIX}.${SUFFIX}.txt --save_dir ${DATA_DIR}/matched_notopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_matched_notopics.model --gpu 0 --batch_size 32  --write_attention --epochs 5 --lr 0.0001 --load
python analysis_scripts/write_metrics.py --attention_file ${DATA_DIR}/matched_notopics_${SUBS_SUFFIX}/test.${SUBS_SUFFIX}.${SUFFIX}.rt_gender_${SUFFIX}_matched_notopics.model_attention.txt

# No matching, no demotion
python train.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${SUFFIX}.${NO_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.${SUBS_SUFFIX}.${SUFFIX}.txt --save_dir ${DATA_DIR}/baseline_notopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_baseline_notopics.model --gpu 0 --batch_size 32 --write_attention --epochs 5 --lr 0.0001 --load
python analysis_scripts/write_metrics.py --attention_file ${DATA_DIR}/baseline_notopics_${SUBS_SUFFIX}/test.${SUBS_SUFFIX}.${SUFFIX}.rt_gender_${SUFFIX}_baseline_notopics.model_attention.txt

# Matching, demotion
python train_ganlike_multiple_decoders.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${MATCHED_SUFFIX}.${EXTRA_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.${SUBS_SUFFIX}.${SUFFIX}.txt --save_dir ${DATA_DIR}/matched_withtopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_matched_withtopics.model --gpu 0 --batch_size 32  --write_attention --c_steps 3 --t_steps 10 --epochs 3 --lr 0.0001 --load
python analysis_scripts/write_metrics.py --attention_file ${DATA_DIR}/matched_withtopics_${SUBS_SUFFIX}/test.${SUBS_SUFFIX}.${SUFFIX}.rt_gender_${SUFFIX}_matched_withtopics.model_attention.txt

# No matching, demotion
# This command will not run for the congress data. We have excluded the file train.subs_name2.facebook_congress.withtopics.txt from this directory because of its size. We have provided the outputted _attention files in baseline_withtopics_subs_name2
python train_ganlike_multiple_decoders.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${SUFFIX}.${EXTRA_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.${SUBS_SUFFIX}.${SUFFIX}.txt --save_dir ${DATA_DIR}/baseline_withtopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_baseline_withtopics.model --gpu 0 --batch_size 32  --write_attention --c_steps 3 --t_steps 10 --epochs 3 --lr 0.00099999 --load
python analysis_scripts/write_metrics.py --attention_file ${DATA_DIR}/baseline_withtopics_${SUBS_SUFFIX}/test.${SUBS_SUFFIX}.${SUFFIX}.rt_gender_${SUFFIX}_baseline_withtopics.model_attention.txt


# ##################################### Run Models on microagressions data ######################################################
python train.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${MATCHED_SUFFIX}.${NO_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.micro.txt --save_dir ${DATA_DIR}/matched_notopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_matched_notopics.model --gpu 0 --batch_size 32  --write_attention --epochs 5 --lr 0.0001 --load
python analysis_scripts/write_metrics.py --attention_file ${DATA_DIR}/matched_notopics_${SUBS_SUFFIX}/test.micro.rt_gender_${SUFFIX}_matched_notopics.model_attention.txt

python train.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${SUFFIX}.${NO_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.micro.txt --save_dir ${DATA_DIR}/baseline_notopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_baseline_notopics.model --gpu 0 --batch_size 32 --write_attention --epochs 5 --lr 0.0001 --load
python analysis_scripts/write_metrics.py --attention_file ${DATA_DIR}/baseline_notopics_${SUBS_SUFFIX}/test.micro.rt_gender_${SUFFIX}_baseline_notopics.model_attention.txt

python train_ganlike_multiple_decoders.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${MATCHED_SUFFIX}.${EXTRA_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.micro.txt --save_dir ${DATA_DIR}/matched_withtopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_matched_withtopics.model --gpu 0 --batch_size 32  --write_attention --c_steps 3 --t_steps 10 --epochs 3 --lr 0.0001 --load
python analysis_scripts/write_metrics.py --attention_file ${DATA_DIR}/matched_withtopics_${SUBS_SUFFIX}/test.micro.rt_gender_${SUFFIX}_matched_withtopics.model_attention.txt

python train_ganlike_multiple_decoders.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${SUFFIX}.${EXTRA_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.micro.txt --save_dir ${DATA_DIR}/baseline_withtopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_baseline_withtopics.model --gpu 0 --batch_size 32  --write_attention --c_steps 3 --t_steps 10 --epochs 3 --lr 0.0001 --load
python analysis_scripts/write_metrics.py --attention_file ${DATA_DIR}/baseline_withtopics_${SUBS_SUFFIX}/test.micro.rt_gender_${SUFFIX}_baseline_withtopics.model_attention.txt
