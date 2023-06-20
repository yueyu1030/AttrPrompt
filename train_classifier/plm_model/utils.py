import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

import re
import json
import logging
import copy
import csv,os
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    # tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_doc(x, word_freq):
    stop_words = set(stopwords.words('english'))
    clean_docs = []
    most_commons = dict(word_freq.most_common(min(len(word_freq), 50000)))
    for doc_content in x:
        doc_words = []
        cleaned = clean_str(doc_content.strip())
        for word in cleaned.split():
            if word not in stop_words and word_freq[word] >= 5:
                if word in most_commons:
                    doc_words.append(word)
                else:
                    doc_words.append("<UNK>")
        doc_str = ' '.join(doc_words).strip()
        clean_docs.append(doc_str)
    return clean_docs

class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, 
                 e1_mask = None, e2_mask = None, keys=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        self.keys=keys

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class Processor(object):
    """Processor for the text data set """
    def __init__(self, args):
        self.args = args
        if self.args.task in ['agnews']:
            self.num_label = 4
        elif self.args.task in ['amazon', 'YelpReviewPolarity','imdb','youtube', 'amazon-polarity', 'SST-2', 'sst2', 'elec', 'qnli', 'yelp']:
            self.num_label = 2
        elif self.args.task in ['nyt']:
            self.num_label = 26
        elif self.args.task in ['amazon-product']:
            self.num_label = 23
        elif self.args.task in ['reddit']:
            self.num_label = 45
        elif self.args.task in ['stackexchange']:
            self.num_label = 50
        elif self.args.task in ['arxiv']:
            self.num_label = 98
        #for i in range(self.num_label):
        self.relation_labels = [x for x in range(self.num_label)]
        self.label2id = {x:x for x in range(self.num_label)}
        self.id2label = {x:x for x in range(self.num_label)}


    def read_data(self, filename):
        path = filename
        with open(path, 'r') as f:
            data = f #json.load(f)
            for x in data:
                yield json.loads(x)
        # return data

    def _create_examples(self, data, set_type):
        examples = []
        for i, d in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            try:
                text_a = d["text"].strip().replace('\\n', ' ').replace('\\', ' ').strip("\n\t\"")
            except:
                text_a = d["text_a"].strip().replace('\\n', ' ').replace('\\', ' ').strip("\n\t\"")
            label = d["_id"] 
            if 'text_b' in d:
                text_b = d["text_b"]
            else:
                text_b = ''

            if i % 5000 == 0:
                logger.info(d)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b = text_b, label=label))
        return examples

    def get_examples(self, mode, file_to_read = None):
        """
        Args:
            mode: train, dev, test
        """
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
        elif mode == 'unlabeled':
            file_to_read = self.args.unlabel_file
        elif mode == 'contrast':
            file_to_read = file_to_read

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        # if mode == 'contrast':
        #     return self._create_examples(self.read_data(os.path.join(self.args.data_dir, file_to_read)), mode)
        # else:
        return self._create_examples(self.read_data(os.path.join(self.args.data_dir, file_to_read)), mode)

def load_and_cache_examples(args, tokenizer, mode, size = -1, contra_name = None):
    processor = Processor(args)
    if mode in ["test"]:
        cached_features_file = os.path.join(
            args.cache_dir,
            'cached_{}_{}_{}_{}'.format(
                mode,
                args.task,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_len_test,
            )
        )
    elif mode in ["dev", "valid"]:
        cached_features_file = os.path.join(
            args.cache_dir,
            'cached_{}_{}_{}_{}'.format(
                mode,
                args.task,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_len_test,
            )
        )
    elif mode in ["contrast"]:
        cached_features_file = os.path.join(
            args.cache_dir,
            'cached_{}_{}_{}_{}_{}'.format(
                mode,
                args.task,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_len_test,
                contra_name[:-5]
            )
        )
    else:
        cached_features_file = os.path.join(
            args.cache_dir,
            'cached_{}_{}_{}_{}'.format(
                mode,
                args.task,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_len,
            )
        )

    if os.path.exists(cached_features_file) and args.auto_load:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.cache_dir)
        if mode == "train":
            examples = processor.get_examples("train")
            max_seq_len = args.max_seq_len
        elif mode == "dev":
            examples = processor.get_examples("dev")
            max_seq_len = args.max_seq_len
        elif mode == "test":
            examples = processor.get_examples("test")
            max_seq_len = args.max_seq_len_test
        elif mode == 'contrast':
            examples = processor.get_examples("contrast", file_to_read = contra_name)
            if "sst2" in contra_name[:-5] or 'mr' in contra_name[:-5] or 'rt' in contra_name[:-5]:
                max_seq_len = 128
            elif 'amazon' in contra_name[:-5]:
                max_seq_len = 128
            elif "yelp" in contra_name[:-5] or 'imdb' in contra_name[:-5]:
                max_seq_len = 400
            else:
                max_seq_len = args.max_seq_len

        else:
            raise Exception("For mode, Only train, dev, test is available")
        features = convert_examples_to_features(examples, max_seq_len, tokenizer, add_sep_token=args.add_sep_token, multi_label=True if  args.multi_label else False)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
   
    # Convert to Tensors and build dataset
    if size > 0:
        import random 
        random.shuffle(features)
        features = features[:size]
    else:
        size = len(features)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if args.multi_label:
        def generate_list(id_list, length):
            result = [0] * length  # Initialize the list with zeros
            for id in id_list:
                if id < length:
                    result[id] = 1  # Set the corresponding index to 1
            return result
        label_lst = [generate_list(f.label_id, 98) for f in features]
        all_label_ids = torch.tensor(label_lst, dtype=torch.long)
    else:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_ids = torch.tensor([ _ for _,f in enumerate(features)], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids, all_ids)
    return dataset, processor.num_label, size

def load_and_cache_unlabeled_examples(args, tokenizer, mode, train_size = 100, size = -1):
    processor = Processor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.cache_dir,
        'cached_{}_{}_{}_{}_unlabel'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
        )
    )

    if os.path.exists(cached_features_file) and args.auto_load:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.cache_dir)

        assert mode == "unlabeled"
        examples = processor.get_examples("unlabeled")
        
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    if size > 0:
        features = features[:size]
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_ids = torch.tensor([_+train_size for _ ,f in enumerate(features)], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_label_ids, all_ids)

    return dataset, len(features)

def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 add_sep_token=False,
                                 mask_padding_with_zero=True,
                                 multi_label=False,
                                ):
    features = []
    sample_per_example = 3
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        #print(example.text_a)
        tokens_a = tokenizer.tokenize(example.text_a)
        if example.text_b != "":
            tokens_b = tokenizer.tokenize(example.text_b)
        else:
            tokens_b = ''

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        if add_sep_token:
            special_tokens_count = 2
        else:
            special_tokens_count = 1
        if tokens_b != '':
            if len(tokens_a) > max_seq_len - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_len - special_tokens_count)]
        else:
            cutoff_len = int((max_seq_len - special_tokens_count) * 0.7)
            if len(tokens_a) > cutoff_len:
                tokens_a = tokens_a[:cutoff_len]
        tokens = tokens_a
        if tokens_b != '':
            tokens += [tokenizer.sep_token]
            tokens += tokens_b
        if add_sep_token:
            sep_token = tokenizer.sep_token
            tokens += [sep_token]

        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]

        token_type_ids = [sequence_a_segment_id] * len(tokens)
        cls_token = tokenizer.cls_token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        #tokens[0] = "$"
        #tokens[1] = "<e2>"
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)


        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        label_id = int(example.label) if not multi_label else list(example.label)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            if multi_label:
                logger.info(f"label: {label_id}")
            else:
                logger.info("label: %s (id = %d)" % (example.label, label_id))
        features.append(
            InputFeatures(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            label_id=label_id,
                          )
            )

    return features


