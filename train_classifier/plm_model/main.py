
import argparse
import os
from utils import load_and_cache_examples, load_and_cache_unlabeled_examples, init_logger, load_tokenizer
from trainer_semi import Trainer
import torch 
import numpy as np 
import random 
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, Subset
import json
import pickle 
from eval import inference_cal, inference_conf, inference_uncertainty, load_pred_data, save_data, create_new_dataset, plot_data
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import copy 

def set_seed(args):
    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if args.n_gpu > 0  and torch.cuda.is_available():
        # print('yes')
        # assert 0
        torch.cuda.manual_seed_all(args.train_seed)
        torch.cuda.manual_seed(args.train_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset, num_labels, train_size  = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset, num_labels,  dev_size = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset, num_labels, test_size = load_and_cache_examples(args, tokenizer, mode="test")
    # unlabeled_dataset, unlabeled_size = load_and_cache_unlabeled_examples(args, tokenizer, mode = 'unlabeled', train_size = train_size)
    contra_datasets = []
    if args.do_extra_eval:        
        dataset_lst = args.extra_dataset.split(",")
        for data in dataset_lst:
            dataset, _, _ = load_and_cache_examples(args, tokenizer, mode = 'contrast', size = -1, contra_name = data)
            contra_datasets.append(dataset)
        print(len(contra_datasets))
   
    print('number of labels:', num_labels)
    print('train_size:', train_size)
    print('dev_size:', dev_size)
    print('test_size:', test_size)
    # print('unlabel_size:', unlabeled_size)
    # if args.train_full_labels:
    #     train_dataset = unlabeled_dataset
    #     # unlabeled_dataset = 
    trainer = Trainer(args, train_dataset=train_dataset, dev_dataset=dev_dataset,test_dataset=test_dataset, contra_datasets = contra_datasets, \
            unlabeled = None, \
            num_labels = num_labels, data_size = train_size
            )
    if args.load_prev == 1:
        print(f"Load ckpt from {args.prev_ckpt}")
        trainer.load_model(path= args.prev_ckpt)
   

    trainer.init_model()

    # trainer.meta_train(n_sample = len(train_dataset))
    trainer.train(n_sample = len(train_dataset))
    exit()
    train_pred, train_feat, train_label, unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo = trainer.inference()
    save_data(train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo, mutual_info_bald = None, n_iter = args.round, dataset = args.task, n_labels = args.label_per_class)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='clean', type=str, help="which method to use")
    parser.add_argument("--gpu", default='0,1,2,3', type=str, help="which gpu to use")
    parser.add_argument("--n_gpu", default=1, type=int, help="which gpu to use")

    parser.add_argument("--seed", default=0, type=int, help="which seed to use")
    parser.add_argument("--train_seed", default=0, type=int, help="which seed to use")
    parser.add_argument("--task", default="agnews", type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="../datasets", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    parser.add_argument("--eval_dir", default="./eval", type=str, help="Evaluation script, result directory")
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--dev_file", default="dev.tsv", type=str, help="dev file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--unlabel_file", default="unlabeled.tsv", type=str, help="Test file")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_extra_eval", action="store_true", help="Whether to run extra eval on the test set.")

    parser.add_argument("--extra_dataset", default="", type=str, help="Whether to run extra eval on the test set.")

    parser.add_argument("--dev_labels", default=100, type=int, help="number of labels for dev set")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.",)

    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X updates steps.")
    parser.add_argument('--self_train_logging_steps', type=int, default=20, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")
    parser.add_argument("--model_type", default="bert-base-uncased", type=str)
    parser.add_argument("--tokenizer", default="bert-base-uncased", type=str)

    parser.add_argument("--auto_load", default=1, type=int, help="Auto loading the model or not")
    parser.add_argument("--add_sep_token", action="store_true", help="Add [SEP] token at the end of the sentence")

    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=100, type=int, help="Training steps for initialization.")
    parser.add_argument("--weight_decay", default=1e-3, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--self_training_batch_size", default=32, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--eval_batch_size", default=256, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--max_seq_len_test", default=128, type=int, help="The maximum total input sequence length after tokenization.")

    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate_st", default=1e-5, type=float, help="The initial learning rate for Adam.")

    parser.add_argument('--self_training_max_step', type = int, default = 10000, help = 'the maximum step (usually after the first epoch) for self training')    
    parser.add_argument('--soft_label', type = int, default = 1, help = 'whether soft label (0 for hard, 1 for soft)')
    parser.add_argument('--self_training_update_period', type = int, default = 100, help = 'update period')
    parser.add_argument("--self_training_eps", default=0.6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--self_training_power", default=2, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--self_training_weight", default=0.5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--gce_loss_q", default=0.5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--sample_labels", default=32, type=int, help="The initial learning rate for Adam.")


    parser.add_argument('--label_per_class', type = int, default = 10, help = 'virtual adversarial training.')
    parser.add_argument('--num_unlabeled', type = int, default = 100, help = 'virtual adversarial training.')
    parser.add_argument('--num_unlabeled_add', type = int, default = 100, help = 'virtual adversarial training.')

    parser.add_argument('--dr_model', type = str, default = 'coco_dr', help = 'virtual adversarial training.')
    parser.add_argument('--dr_N', type = int, default = 10, help = 'virtual adversarial training.')

    parser.add_argument('--dr_dataset', type = str, default = 'news', help = 'virtual adversarial training.')
    parser.add_argument('--filter', type = int, default = 0, help = 'Whether use filter.')
    parser.add_argument('--template', type = int, default = 0, help = 'Prompt Template.')
    parser.add_argument('--round', type = int, default = 0, help = 'Prompt Template.')
    parser.add_argument('--load_prev', type = int, default = 0, help = 'whether load model from prev_round.')
    parser.add_argument('--prev_ckpt', type = str, default = '', help = 'dir of prev ckpt.')
    parser.add_argument('--word_template', type = int, default = 0, help = 'dir of prev ckpt.')
    parser.add_argument('--n_labels', type = int, default = 3000, help = 'number of unlabeled data of prev ckpt.')
    parser.add_argument('--train_full_labels', type = int, default = 0, help = 'whether train with full labels or not')
    parser.add_argument('--bitfit', type = int, default = 0, help = 'whether train with bitfit or not')
    parser.add_argument('--multi_label', type = int, default = 0, help = 'whether train with multi label or not')

# semi_method
    args = parser.parse_args()
    if 'dsp' in args.model_type:
        args.model_name_or_path = 'allenai/'+args.model_type
        args.tokenizer =  'allenai/'+args.model_type
    elif 'biobert' in args.model_type:
        args.model_name_or_path = 'dmis-lab/biobert-v1.1'
        args.tokenizer =  'dmis-lab/biobert-v1.1'
    elif 'codebert' in args.model_type:
        args.model_name_or_path = 'microsoft/codebert-base'
        args.tokenizer = 'microsoft/codebert-base'
    elif 'scibert' in args.model_type:
        args.model_name_or_path = 'allenai/scibert_scivocab_uncased'
        args.tokenizer = 'allenai/scibert_scivocab_uncased'
    elif 'cocolm' in args.model_type:
        args.model_name_or_path = 'microsoft/cocolm-large'
        args.tokenizer =  'microsoft/cocolm-large'
    elif 'deberta-base' in args.model_type:
        args.model_name_or_path = 'microsoft/deberta-v3-base'
        args.tokenizer = 'microsoft/deberta-v3-base'
    elif 'deberta-large' in args.model_type:
        args.model_name_or_path = 'microsoft/deberta-v3-large'
        args.tokenizer =  'microsoft/deberta-v3-large'
    elif 'deberta-xl' in args.model_type:
        args.model_name_or_path = 'microsoft/deberta-v2-xlarge'
        args.tokenizer =  'microsoft/deberta-v2-xlarge'
    elif 'tinybert' in args.model_type:
        args.model_name_or_path = 'huawei-noah/TinyBERT_General_4L_312D'
        args.tokenizer =  'huawei-noah/TinyBERT_General_4L_312D'
    elif 'distilbert' in args.model_type:
        args.model_name_or_path = 'distilbert-base-uncased'
        args.tokenizer =  'distilbert-base-uncased'
    else:
        args.model_name_or_path = args.model_type
        args.tokenizer = args.model_type
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    main(args)