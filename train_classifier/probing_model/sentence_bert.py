import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import numpy as np
from openTSNE import TSNE
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import argparse
import os 
#Mean Pooling - Take attention mask into account for correct averaging

def load_sentences(file_path):
    texts = []
    ids = []

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)            
            texts.append(data['text'])
            ids.append(data['_id'])
    return texts,  np.array(ids)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def tokenization(args, sentences, tokenizer, model, batch_size = 128, normalize = False):
    # Tokenize sentences
    sentence_embeddings = []
    num_iter = len(sentences)//batch_size if len(sentences) % batch_size == 0 else (len(sentences)//batch_size + 1)

    for i in trange(num_iter):
        encoded_input = tokenizer(sentences[i*batch_size:(i+1)*batch_size], max_length = 200, padding=True, truncation=True, return_tensors='pt').to(f"cuda:{gpuid}")

        # Compute token embeddings
        if args.sentence_embedding_model in ['bert', 'simcse']:
            with torch.no_grad():
                embeddings = model(**encoded_input, output_hidden_states=True, return_dict=True).pooler_output
                sentence_embeddings.append(embeddings)
        elif args.sentence_embedding_model in ['sentencebert']:
            with torch.no_grad():
                model_output = model(**encoded_input)
                    # Perform pooling
                embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                sentence_embeddings.append(embeddings)
        else:
            with torch.no_grad():
                embeddings = model(**encoded_input, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1].squeeze(1) # the embedding of the [CLS] token after the final layer
                sentence_embeddings.append(embeddings)

    sentence_embeddings = torch.cat(sentence_embeddings, dim = 0)

    # Normalize embeddings, 
    if normalize:
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1).squeeze(1)

    print("Sentence embeddings Shape:", sentence_embeddings.shape)
    return sentence_embeddings.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default=0)
    parser.add_argument("--method", type=str, default=0)
    parser.add_argument("--sentence_embedding_model", type=str, default='bert')
    parser.add_argument("--file_name", type=str, default='train.jsonl') # the file name that need to be embed

    args = parser.parse_args()
    gpuid = args.gpu
    dataset = args.dataset
    model = args.model

    print("file name", args.file_name)

    # Load model from HuggingFace Hub
    if args.sentence_embedding_model == 'simcse':
        embedding_model = 'princeton-nlp/sup-simcse-bert-base-uncased'
    elif  args.sentence_embedding_model == 'cocodr':
        embedding_model = "OpenMatch/cocodr-base-msmarco"
    elif  args.sentence_embedding_model == 'sentencebert':
        embedding_model = "sentence-transformers/all-mpnet-base-v2"
    elif  args.sentence_embedding_model == 'bert':
        embedding_model = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model).to(f"cuda:{gpuid}")
    sentences, labels = load_sentences(f"{args.file_name}")
    embeddings = tokenization(args, sentences, tokenizer, model, batch_size = 256, normalize = False)
    os.makedirs(f"./{args.dataset}/{args.sentence_embedding_model}", exist_ok= True)
    np.save(f"./{args.dataset}/{args.sentence_embedding_model}/embeddings_{args.method}.npy", embeddings)
    np.save(f"./{args.dataset}/{args.sentence_embedding_model}/labels_{args.method}.npy", labels)
    
