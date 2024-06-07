pip install ray "ray[tune]" transformers datasets
pip install accelerate -U
pip install sentencepiece

# rm -r results
# rm -r ray_results
workdir="/openbayes/home/"       # remember the `/` at the end

model_path="${workdir}flan-t5-small/"
train_file="${workdir}data/mydata/mytrain.csv"
validation_file="${workdir}data/mydata/myvalidation.csv"
test_file="${workdir}data/mydata/mytest.csv"
output_dir="${workdir}results"
best_model_dir="${workdir}best_model"
local_dir="${workdir}ray_results"
traindata_process_save_path="${workdir}data/mydata/preprocessed/train.pkl"
validationdata_process_save_path="${workdir}data/mydata/preprocessed/valid.pkl"
testdata_process_save_path="${workdir}data/mydata/preprocessed/test.pkl"
rm -r best_model results huggingface ray_results data/mydata/preprocessed



#####! test 20 samples
# model_path="${workdir}flan-t5-small/"
# train_file="${workdir}data/mydata/mytrain_20.csv"
# validation_file="${workdir}data/mydata/myvalidation_20.csv"
# test_file="${workdir}data/mydata/mytest_20.csv"
# output_dir="${workdir}results_20"
# best_model_dir="${workdir}best_model_20"
# local_dir="${workdir}ray_results_20"
# traindata_process_save_path="${workdir}data/mydata/preprocessed/train_20.pkl"
# validationdata_process_save_path="${workdir}data/mydata/preprocessed/valid_20.pkl"
# testdata_process_save_path="${workdir}data/mydata/preprocessed/test_20.pkl"
rm -r best_model_20 results_20 huggingface ray_results_20 data/mydata/preprocessed

# echo "$model_path"

python Q1_2.py --model_path ${model_path} --train_file $train_file --validation_file $validation_file --test_file $test_file --output_dir $output_dir --bestmodel_dir $best_model_dir --local_dir $local_dir --traindata_process_save_path $traindata_process_save_path --valdata_process_save_path $validationdata_process_save_path --testdata_process_save_path $testdata_process_save_path