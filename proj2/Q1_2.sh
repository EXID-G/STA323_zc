# pip install ray "ray[tune]" transformers datasets
# pip install accelerate -U

# rm -r /home/pod/shared-nvme/data/cachefile
# rm -r results
# rm -r ray_results

workdir = "/openbayes/home/"       # remember the `/` at the end
model_path = "${workdir}flan-t5-small/"
train_file = "${workdir}data/mydata/mytrain.csv"
validation_file = "${workdir}data/mydata/myvalid.csv"
test_file = "${workdir}data/mydata/mytest.csv"
output_dir = "${workdir}results"
best_model_dir = "${workdir}best_model"
local_dir = "${workdir}ray_results"



python Q1_2.py --model_path $model_path --train_file $train_file --validation_file $validation_file --test_file $test_file --output_dir $output_dir --best_model_dir $best_model_dir --local_dir $local_dir