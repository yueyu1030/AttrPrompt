import os
import logging
from tqdm import tqdm, trange
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset, TensorDataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoModelForSequenceClassification
import copy
import math
import os
import random 
from sklearn.metrics import f1_score
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from collections import Counter
from sklearn.metrics import confusion_matrix, ndcg_score, jaccard_score

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0  and torch.cuda.is_available():
        # print('yes')
        # assert 0
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def compute_metrics_rel(key, prediction):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == 0 and guess == 0:
            pass
        elif gold == 0 and guess != 0:
            guessed_by_relation[guess] += 1
        elif gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        elif gold != 0 and guess != 0:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return {'p':prec_micro, 'r':recall_micro, 'f':f1_micro}


def acc_and_f1(preds, labels, average='macro'):
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average='macro')
    #macro_recall = recall_score(y_true=labels, y_pred = preds, average = 'macro')
    #micro_recall = recall_score(y_true=labels, y_pred = preds, average = 'micro')
    #print(acc, macro_recall, micro_recall)

    return {
        "acc": acc,
        "f1": f1
    }

class Trainer(object):
    def __init__(self, args, train_dataset = None, dev_dataset = None, test_dataset = None, unlabeled = None, contra_datasets= [], \
                num_labels = 10, data_size = 100, n_gpu = 1):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.unlabeled = unlabeled
        self.contra_datasets = contra_datasets
        self.data_size = data_size

        self.num_labels = num_labels
        self.config_class = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=self.num_labels)
        self.n_gpu = 1
        # self.devices = "cuda"

    def reinit(self):
        self.load_model()
        self.init_model()

    def init_model(self):
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and self.n_gpu > 0 else "cpu"
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
    
    def load_model(self, path = None):
        print("load Model")
        if path is None:
            logger.info("No ckpt path, load from original ckpt!")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name_or_path,
                config=self.config_class,
                cache_dir=self.args.cache_dir if self.args.cache_dir else None,
            )
        else:
            print(f"Loading from {path}!")
            logger.info(f"Loading from {path}!")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path,
                config=self.config_class,
                cache_dir=self.args.cache_dir if self.args.cache_dir else None,
            )

    def save_prediction_test(self, test_preds, test_labels):
        output_dir = os.path.join(
            self.args.output_dir, self.args.dr_model, self.args.model_type
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.args.multi_label:
            with open(f"{output_dir}/reciprocal_ranks.npy", 'wb') as f:
                np.save(f, test_labels)
            
            with open(f"{output_dir}/precision_5_per_example.npy", 'wb') as f:
                np.save(f, test_preds)
        else:
            with open(f"{output_dir}/test_label.npy", 'wb') as f:
                np.save(f, test_labels)
            
            with open(f"{output_dir}/test_pred.npy", 'wb') as f:
                np.save(f, test_preds)


    def save_prediction(self, loss, preds, labels, test_preds, test_labels):
        output_dir = os.path.join(
            self.args.output_dir, self.args.dr_model
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f"{output_dir}/train_pred.npy", 'wb') as f:
            np.save(f, preds)
    
        with open(f"{output_dir}/train_loss.npy", 'wb') as f:
            np.save(f, loss)
        
        with open(f"{output_dir}/train_label.npy", 'wb') as f:
            np.save(f, labels)

        with open(f"{output_dir}/test_label.npy", 'wb') as f:
            np.save(f, test_labels)
        
        with open(f"{output_dir}/test_pred.npy", 'wb') as f:
            np.save(f, test_preds)


    def save_model(self, stage = 0):
        # {self.args.model_type}_{self.args.al_method}
        output_dir = os.path.join(
            self.args.output_dir,  "checkpoint-{}".format(len(self.train_dataset)), self.args.model_type, "iter-{}".format(stage), f"seed{self.args.train_seed}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        # tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        # torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
        logger.info("Saving model checkpoint to %s", output_dir)

    def evaluate(self, mode, dataset = None, global_step=-1, return_preds = False):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'contra':
            dataset = dataset
        elif mode == 'unlabeled':
            dataset = self.unlabeled
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        # logger.info("  Batch size = %d", self.args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                          }
                if 'distilbert' in self.args.model_type:
                    del inputs['token_type_ids']
                if self.args.multi_label:
                    lbls = inputs["labels"]
                    del inputs["labels"]

                outputs = self.model(**inputs)
                if self.args.multi_label:
                    inputs["labels"] = lbls
                    logits = outputs[0] 
                    logits = F.sigmoid(logits)  
                    tmp_eval_loss = F.binary_cross_entropy(logits, lbls.float())
                    # print(logits.shape)   
                    # print(F.sigmoid(logits))    
                    # print(batch[3])  
                    # f1_score(y_true, y_pred, average = None) 
                    # exit()
                else: 
                    tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss 
        }
        def calculate_mrr(labels, predictions):
            sorted_indices = np.argsort(predictions, axis=1)[:, ::-1]
            ranks = np.argmax(labels[np.arange(len(labels))[:, None], sorted_indices], axis=1) + 1
            reciprocal_ranks = 1.0 / ranks
            mrr = np.mean(reciprocal_ranks)
            return mrr, reciprocal_ranks

        def calculate_precision_at_k(labels, predictions, k):
            sorted_indices = np.argsort(predictions, axis=1)[:, ::-1]
            top_k_labels = labels[np.arange(len(labels))[:, None], sorted_indices[:, :k]]
            precision = np.mean(top_k_labels)
            precision_per_example = np.mean(top_k_labels , axis=1)
            return precision, precision_per_example

        if self.args.multi_label:
            preds_probs = preds                     # probability
            preds_binary = np.zeros(preds.shape) 
            preds_binary[np.where(preds >= 0.5)] = 1 # binary prediction 
            ndcg_1 = ndcg_score(out_label_ids, preds_probs, k = 1)
            ndcg_3 = ndcg_score(out_label_ids, preds_probs, k = 3)
            ndcg_5 = ndcg_score(out_label_ids, preds_probs, k = 5)
            f1_macro = f1_score(out_label_ids, preds_binary, average='macro')
            f1_micro = f1_score(out_label_ids, preds_binary, average='micro')
            jaccard_sample = jaccard_score(out_label_ids, preds_binary, average='samples')
            jaccard_macro = jaccard_score(out_label_ids, preds_binary, average='macro')
            precision_1, precision_1_per_example = calculate_precision_at_k(out_label_ids, preds_probs, k = 1)
            precision_3, precision_3_per_example = calculate_precision_at_k(out_label_ids, preds_probs, k = 3)
            precision_5, precision_5_per_example = calculate_precision_at_k(out_label_ids, preds_probs, k = 5)
            # print(precision_5_per_example.shape)
            mrr, reciprocal_ranks = calculate_mrr(out_label_ids, preds_probs)
            # print(reciprocal_ranks)

            print("=========================")
            print("NDCG@1", ndcg_1, "NDCG@3", ndcg_3, "NDCG@5", ndcg_5, "F1 Macro", f1_macro, "F1 Micro", f1_micro, "Jaccard Sample", jaccard_sample, "Jaccard Macro", jaccard_macro)
            print("P@1",precision_1, "P@3",precision_3, "P@5",precision_5,  "MRR", mrr)
            print("=========================")
            if return_preds:
                 return results["loss"], f1_macro, ndcg_3, precision_5_per_example, reciprocal_ranks
            else:
                return results["loss"], f1_macro, ndcg_3
        else:
            preds_probs = np.exp(preds) / np.sum(np.exp(preds), axis = -1, keepdims = True) 
            preds = np.argmax(preds, axis=1) 
            

            if mode == 'unlabeled':
                return preds, preds_probs, out_label_ids
            
            result = compute_metrics(preds, out_label_ids)
            result.update(result)
            logger.info("***** Eval results *****")

            # print('Accu: %.4f'%(result["acc"]))
            if return_preds:
                print("=================")
                print("Confusion Matrix:")
                print(confusion_matrix(out_label_ids, preds))
                print("====================")
                return results["loss"], result["acc"], result["f1"], preds_probs, out_label_ids
            else:
                return results["loss"], result["acc"], result["f1"]
    
    def train(self, n_sample = 20):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        training_steps = max(self.args.max_steps, int(self.args.num_train_epochs) * len(train_dataloader))
        criterion = nn.CrossEntropyLoss(reduction = 'mean') if not self.args.multi_label else nn.BCEWithLogitsLoss()

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(training_steps * 0.06 / self.args.gradient_accumulation_steps), num_training_steps = training_steps)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size )
        logger.info("  Real batch size = %d", self.args.batch_size * self.args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", training_steps)
        global_step = 0
        tr_loss = 0.0

        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args)
        best_dev = -np.float('inf')
        for _ in train_iterator:
            global_step = 0
            tr_loss = 0.0
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            local_step = 0
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                          }
                if 'distilbert' in self.args.model_type:
                    del inputs['token_type_ids']
                if self.args.multi_label:
                    del inputs["labels"]
               
                outputs = self.model(**inputs)
                if  self.args.multi_label:
                    logits = outputs[0]
                    loss = criterion(input = logits, target = batch[3].float())
                else:
                    loss = outputs[0]
                
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps           
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    local_step += 1
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    epoch_iterator.set_description("iteration:%d, Loss:%.3f, best dev:%.3f" % (_, tr_loss/global_step, 100*best_dev))
                if 0 < training_steps < global_step:
                    epoch_iterator.close()
                    break

            loss_dev, acc_dev, f1_dev = self.evaluate('dev', global_step)
            
            loss_test, acc_test, f1_test = 0 ,0, 0
            loss_test, acc_test, f1_test = self.evaluate('test', global_step)
            if acc_dev > best_dev:
                logger.info("Best model updated!")
                self.best_model = copy.deepcopy(self.model.state_dict())
                best_dev = acc_dev
            print(f'Dev: Loss: {loss_dev}, Acc: {acc_dev}, F1: {f1_dev}', f'Test: Loss: {loss_test}, Acc: {acc_test}, F1: {f1_test}')
        result_dict = {'seed': self.args.train_seed}
        loss_test, acc_test, acc_f1, preds_probs, out_label_ids = self.evaluate('test', global_step, return_preds = True)
        result_dict['acc'] = acc_f1
        result_dict['lr'] = self.args.learning_rate
        result_dict['bsz'] = self.args.batch_size
        result_dict['model'] = self.args.dr_model
        print(f'Test: Loss: {loss_test}, Acc: {acc_test}, F1: {acc_f1}')
        self.save_prediction_test(preds_probs, out_label_ids)
        self.save_model(stage = n_sample)
        return global_step, tr_loss / global_step
  