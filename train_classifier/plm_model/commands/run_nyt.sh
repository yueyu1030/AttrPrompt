task=nyt
gpu=$1
n_gpu=2

for seed in 0 ; do
for train_seed in 0  ; do # 128 0 21 42 87 100
for model_type in bert-base-uncased ; do #
for n_label in 100; do # the number of training examples per class
for prefix in simprompt attrprompt ; do

seed=${seed}
train_seed=${train_seed}
max_seq_len=128
max_seq_len_test=128

eval_batch_size=256
steps=100
#####################
gen_model=gpt3
gpt_model="gpt-3.5-turbo"
data_dir="./dataset/${task}"
gen_model=${gen_model}_${prefix}_${n_label}
train_file="train_${gpt_model}_${prefix}_n${n_label}.jsonl"
lr=2e-5 # 2e-5
batch_size=32
epochs=8

output_dir=${task}/model
mkdir -p ${output_dir}
mkdir -p ${task}/cache

train_cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main_semi.py --do_train --do_eval --task=${task} \
	--train_file=${train_file} --dev_file=valid.jsonl --test_file=test.jsonl \
	--unlabel_file=unlabeled.json --tokenizer=${model_type} \
	--gen_model=${gen_model} --data_dir=${data_dir} --seed=${seed} --train_seed=${train_seed} \
	--cache_dir="${task}/cache" --output_dir=${output_dir}  \
	--gpu=${gpu} --n_gpu=${n_gpu} --num_train_epochs=${epochs} --weight_decay=${weight_decay} --learning_rate=${lr}  \
	--batch_size=${batch_size} --eval_batch_size=${eval_batch_size} \
	--max_seq_len=${max_seq_len} --max_seq_len_test=${max_seq_len_test} --auto_load=1 \
	--max_steps=${steps} --model_type=${model_type}"
echo $train_cmd 
eval $train_cmd

done
done
done

done 
done
