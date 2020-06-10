#!/bin/bash
source /projects/tir1/users/anjalief/anaconda3/bin/activate test_env

################################## TO FILL IN ############################################################
RT_GENDER_DATA="/projects/tir3/users/anjalief/gender_bias_test/rt_gender" # the path to the rt_gender data set, which can be downloaded here: https://nlp.stanford.edu/robvoigt/rtgender/
TOP_DIR="/projects/tir3/users/anjalief/gender_bias_test" # Path to store script outputs

SUFFIX="facebook_wiki"  # Flip comment to change which data set to run
# SUFFIX="facebook_congress"
####################################################################################################


# Paths to data
RESPONSE_FILE="${RT_GENDER_DATA}/${SUFFIX}_responses.csv"
POST_FILE="${RT_GENDER_DATA}/${SUFFIX}_posts.csv"
DATA_DIR="${TOP_DIR}/rt_gender_clean_${SUFFIX}"

# Intermediate suffixes
TOK_RESPONSE_FILE="${RT_GENDER_DATA}/${SUFFIX}_responses.tok.tsv"
ODDS_COL="op_name"
MATCHED_SUFFIX="matched_${SUFFIX}"
SUBS_SUFFIX="subs_name2"
EXTRA_SUFFIX="withtopics"
NO_SUFFIX="notopics"


mkdir -p ${DATA_DIR}

run_cmd () {
  echo $cmd
  $cmd
}

cd preprocessing_scripts

# # This was the script used to create data splits. It is not necessary to run it, can run 'make_manual_data_splits.py' to re-create paper splits
# # cmd="python make_clean_data_splits.py --response_file ${TOK_RESPONSE_FILE}  --output_dir ${DATA_DIR}"
# # run_cmd

# Tokenize the rt_gender response data. Split the RtGender corpus into the same data splits as the paper
# python preprocess_rtgender.py --input_file ${RESPONSE_FILE} --output_dir ${RT_GENDER_DATA} --header_to_tokenize 'response_text'
# python make_manual_data_splits.py --response_tsv ${TOK_RESPONSE_FILE} --train_ids ../../${SUFFIX}_data/train_op_ids.txt --test_ids ../../${SUFFIX}_data/test_op_ids.txt --valid_ids ../../${SUFFIX}_data/valid_op_ids.txt --outdirname ${DATA_DIR}


########################################### Do propensity matching ##################################
# cmd="python write_train_op_posts.py --posts_csv ${POST_FILE} --train_ids ../../${SUFFIX}_data/train_op_ids.txt --outfilename ${DATA_DIR}/train_op_posts.tsv"
# run_cmd

# cmd="python write_train_op_posts.py --posts_csv ${POST_FILE} --train_ids ../../${SUFFIX}_data/valid_op_ids.txt --outfilename ${DATA_DIR}/valid_op_posts.tsv"
# run_cmd

cd ..
# mkdir -p ${DATA_DIR}/op_posts_matching
# cmd="python train.py --data RT_GENDER_OP_POSTS --base_path ${DATA_DIR} --train_file train_op_posts.tsv --valid_file valid_op_posts.tsv --save_dir ${DATA_DIR}/op_posts_matching --model RNN --model_name rt_gender_op_posts_rnn.model --gpu 0 --batch_size 32 --write_attention"
# run_cmd

cd preprocessing_scripts

# Can use this to examine data imbalance before propensity matching
# # cmd="python write_op_log_odds.py --raw_data ${DATA_DIR}/train_op_posts.tsv --outfile ${DATA_DIR}/op_posts_matching/nonmatched_log_odds.txt"
# # run_cmd

# Use prediction model to estimate propensity scores
# cmd="python match_propensity_scores.py --propensity_score_file ${DATA_DIR}/op_posts_matching/train.rt_gender_op_posts_rnn.model_attention.txt  --outfile ${DATA_DIR}/op_posts_matching/greedy_rnn_scores_best.csv  --match_type greedy --max_match_dist 0.001"
# run_cmd

# Can use this to examine data imbalance after propensity matching
# # cmd="python write_op_log_odds.py --raw_data ${DATA_DIR}/train_op_posts.tsv --outfile ${DATA_DIR}/op_posts_matching/greedy_rnn_best_log_odds.txt --match_scores ${DATA_DIR}/op_posts_matching/greedy_rnn_scores_best.csv --posts_outfile ${DATA_DIR}/op_posts_matching/train_op_posts_greedy_rnn_matched.tsv"
# # run_cmd

# cmd="python filter_comments_by_matches.py --response_data ${TOK_RESPONSE_FILE} --match_scores ${DATA_DIR}/op_posts_matching/greedy_rnn_scores_best.csv --suffix ${MATCHED_SUFFIX} --outdirname ${DATA_DIR}"
# run_cmd
######################################### Done matching #############################################




########################### Do word-level subsititions ###########################################
# cmd="python do_name_subs.py --training_file ${DATA_DIR}/train.${MATCHED_SUFFIX}.txt --subs_file ./new_subs.txt --output_file ${DATA_DIR}/train.${SUBS_SUFFIX}_${MATCHED_SUFFIX}.txt"
# run_cmd

# cmd="python do_name_subs.py --training_file ${DATA_DIR}/train.txt --subs_file ./new_subs.txt --output_file ${DATA_DIR}/train.${SUBS_SUFFIX}.txt"
# run_cmd

# cmd="python do_name_subs.py --training_file ${DATA_DIR}/test.txt --subs_file ./new_subs.txt --output_file ${DATA_DIR}/test.${SUBS_SUFFIX}.txt"
# run_cmd

# cmd="python do_name_subs.py --training_file ${DATA_DIR}/valid.txt --subs_file ./new_subs.txt --output_file ${DATA_DIR}/valid.${SUBS_SUFFIX}.txt"
# run_cmd


########################### Add log odds to training data ###########################################

# cmd="python add_log_odds_feats.py --input_file ${DATA_DIR}/train.${SUBS_SUFFIX}_${MATCHED_SUFFIX}.txt --output_file ${DATA_DIR}/train.${SUBS_SUFFIX}.${MATCHED_SUFFIX}.${EXTRA_SUFFIX}.txt --odds_column ${ODDS_COL} --add_log_odds"
# run_cmd

cmd="python add_log_odds_feats.py --input_file ${DATA_DIR}/train.${SUBS_SUFFIX}.txt --output_file ${DATA_DIR}/train.${SUBS_SUFFIX}.${SUFFIX}.${EXTRA_SUFFIX}.txt --odds_column ${ODDS_COL} --add_log_odds"
run_cmd

# cmd="python add_log_odds_feats.py --input_file ${DATA_DIR}/train.${SUBS_SUFFIX}_${MATCHED_SUFFIX}.txt --output_file ${DATA_DIR}/train.${SUBS_SUFFIX}.${MATCHED_SUFFIX}.${NO_SUFFIX}.txt"
# run_cmd

# cmd="python add_log_odds_feats.py --input_file ${DATA_DIR}/train.${SUBS_SUFFIX}.txt --output_file ${DATA_DIR}/train.${SUBS_SUFFIX}.${SUFFIX}.${NO_SUFFIX}.txt"
# run_cmd

# cmd="python add_log_odds_feats.py --input_file ${DATA_DIR}/test.${SUBS_SUFFIX}.txt --output_file ${DATA_DIR}/test.${SUBS_SUFFIX}.${SUFFIX}.txt"
# run_cmd

# cmd="python add_log_odds_feats.py --input_file ${DATA_DIR}/valid.${SUBS_SUFFIX}.txt --output_file ${DATA_DIR}/valid.${SUBS_SUFFIX}.${SUFFIX}.txt"
# run_cmd



##################################### Run Models ######################################################
cd ..

# mkdir -p ${DATA_DIR}/matched_notopics_${SUBS_SUFFIX}
# cmd="python train.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${MATCHED_SUFFIX}.${NO_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.${SUBS_SUFFIX}.${SUFFIX}.txt --save_dir ${DATA_DIR}/matched_notopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_matched_notopics.model --gpu 0 --batch_size 32  --write_attention --epochs 5 --lr 0.0001 "
# run_cmd

# mkdir -p ${DATA_DIR}/baseline_notopics_${SUBS_SUFFIX}
# cmd="python train.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${SUFFIX}.${NO_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.${SUBS_SUFFIX}.${SUFFIX}.txt --save_dir ${DATA_DIR}/baseline_notopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_baseline_notopics.model --gpu 0 --batch_size 32 --write_attention --epochs 5 --lr 0.0001"
# run_cmd

# mkdir -p ${DATA_DIR}/matched_withtopics_${SUBS_SUFFIX}
# cmd="python train_ganlike_multiple_decoders.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${MATCHED_SUFFIX}.${EXTRA_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.${SUBS_SUFFIX}.${SUFFIX}.txt --save_dir ${DATA_DIR}/matched_withtopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_matched_withtopics.model --gpu 0 --batch_size 32  --write_attention --c_steps 3 --t_steps 10 --epochs 3 --lr 0.0001"
# run_cmd


# Only for the Wiki model, we changed lr to 0.00099999 to stop it from nanning out
mkdir -p ${DATA_DIR}/baseline_withtopics_${SUBS_SUFFIX}
cmd="python train_ganlike_multiple_decoders.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${SUFFIX}.${EXTRA_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.${SUBS_SUFFIX}.${SUFFIX}.txt --save_dir ${DATA_DIR}/baseline_withtopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_baseline_withtopics.model --gpu 0 --batch_size 32  --write_attention --c_steps 3 --t_steps 10 --epochs 3 --lr 0.00099999"
run_cmd


##################################### Run Models on micro data ######################################################
# cp ../../facebook_wiki_data/test.micro.txt ${DATA_DIR}
# cmd="python train.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${MATCHED_SUFFIX}.${NO_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.micro.txt --save_dir ${DATA_DIR}/matched_notopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_matched_notopics.model --gpu 0 --batch_size 32  --write_attention --epochs 5 --lr 0.0001 --load"
# run_cmd

# cmd="python train.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${SUFFIX}.${NO_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.micro.txt --save_dir ${DATA_DIR}/baseline_notopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_baseline_notopics.model --gpu 0 --batch_size 32 --write_attention --epochs 5 --lr 0.0001 --load"
# run_cmd

# cmd="python train_ganlike_multiple_decoders.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${MATCHED_SUFFIX}.${EXTRA_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.micro.txt --save_dir ${DATA_DIR}/matched_withtopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_matched_withtopics.model --gpu 0 --batch_size 32  --write_attention --c_steps 3 --t_steps 10 --epochs 3 --lr 0.0001 --load"
# run_cmd


# cmd="python train_ganlike_multiple_decoders.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${SUFFIX}.${EXTRA_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.micro.txt --save_dir ${DATA_DIR}/baseline_withtopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_baseline_withtopics.model --gpu 0 --batch_size 32  --write_attention --c_steps 3 --t_steps 10 --epochs 3 --lr 0.00099999 --load"
# run_cmd


########################## Example scripts to print results############################
# python analysis_scripts/write_metrics.py --attention_file ${DATA_DIR}/matched_notopics_${SUBS_SUFFIX}/test.micro.rt_gender_${SUFFIX}_matched_notopics.model_attention.txt
