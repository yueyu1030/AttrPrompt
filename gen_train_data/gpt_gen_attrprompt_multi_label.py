import openai
import asyncio
from typing import List, Dict, Any
import argparse
import os 
import re 
import time
from utils import load_attributes
import numpy as np
import json 
import random 

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

api_key = 'YOUR_OPENAI_API_KEY' # change this to your id

parser = argparse.ArgumentParser("")
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--temperature", default=1, type=float)
parser.add_argument("--top_p", default=0.95, type=float)
parser.add_argument("--n_sample", default=10, type=int, help="number of generated examples per class")
parser.add_argument("--batch_size", default=10, type=int, help="number of generated examples per batch")

parser.add_argument("--dataset", default='agnews', type=str, help="which model to use")

parser.add_argument("--model_name", default='gpt-3.5-turbo', type=str, help="which model to use")
parser.add_argument("--model_type", default='completion', type=str, help="which model type to use")
parser.add_argument("--max_tokens", default=500, type=int, help="the maximum number of tokens for generation")
parser.add_argument("--output_dir", default='.', type=str, help="the folder for saving the generated text")

args = parser.parse_args()
args.api_key = api_key

if args.dataset in ['arxiv']:
    args.domain = 'scientific paper'
    args.attributes = ["length", "subtopics", "technique", "style", "similar"]
    args.metadata = ""
else:
    raise NotImplementedError

async def dispatch_openai_requests(
    messages_list: List[List[Dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> List[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

def call_api_async(msg_lst, model, temperature, max_tokens):
    print("===================================")
    print(f"call APIs, {len(msg_lst)} in total, t= {temperature}.")
    l = len(msg_lst)

    response = asyncio.run(
        dispatch_openai_requests(
            messages_list = msg_lst,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
        )
    )

    ans = [x['choices'][0]['message']['content'] for x in response]
    print(f"API returns {len(ans)} in total.")
    print("===================================")
    return ans 

def main(args):

    with open(f"../datasets/{args.dataset}/label.txt", 'r') as f:
        label_names = [x.lower().replace(" ", "_").strip('\n') for x in f.readlines()]
        id2label = {x: i for (i, x) in enumerate(label_names)}
    print(label_names)
    model = args.model_name
    openai.api_key = args.api_key

    # Call the function to clear the terminal screen
    # clearScreen()

    # Enter a while loop to continually prompt the user for input and generate responses from the OpenAI API
    # while True:
    # Check this url for more info about the parameters
    # https://platform.openai.com/docs/api-reference/completions/create
    
    attr_dict = {}
    for attr in args.attributes:
        if 'subtopics' in attr:
            attr_name = 'subtopics_filter'
        else:
            attr_name = attr 
        if 'subtopics' in attr:
            attr_dict['subtopics'] =  json.load(open(f"../datasets/{args.dataset}/{args.model_type}/{args.model_name}/{attr_name}/subtopics.json", "r"))
            print(len(attr_dict['subtopics']))
        else:
            attr_dict[attr] = load_attributes(attr_name = attr_name, model = model, dataset = args.dataset, method = args.model_type, classes = label_names)
            print(attr, len(attr_dict[attr]))

    for i,subtopic in enumerate(attr_dict['subtopics']):
        print(f"Subclass {i}: {subtopic}.")
        sent_cnt = 0
        attempt = 0
        prompt_lst = []
        attr_lst = []
        examples = []
       
        random.seed(i + 1234)
        while sent_cnt < args.n_sample:
            subtopic_dict = attr_dict['subtopics'][subtopic]
            labels = subtopic_dict["lbl"]
            label_ids = [id2label[x] for x in labels] # topic id
            
            style = random.sample(attr_dict["style"], 1)[0]
            length = random.sample(attr_dict["length"], 1)[0]
            technique = [random.sample(attr_dict["technique"][x], 1)[0] for x in labels]
            random.shuffle(technique)
            random.shuffle(labels)
            attr_lst.append(subtopic_dict)
            prompt_input = f"Write an abstract of {labels} paper in arXiv, following the requirements below: \n \
                        1. the paper abstract should focus on '{re.sub(' ', '_', subtopic)}';\n \
                        2. should be in length between {length} words and {int(length) + 50} words;\n \
                        3. the paper should use the techniques relevant to {technique};\n \
                        4. the style of the paper should be '{style}'"            
            if i % 200 == 0 and attempt == 0:
                print(prompt_input)
            prompt_lst.append(
                    [{"role": "user", "content": prompt_input}]
            )
            if len(prompt_lst) == args.batch_size:
                try:
                    attempt += 1
                    return_msg = call_api_async(prompt_lst, model, args.temperature, args.max_tokens)
                    assert len(return_msg) == len(attr_lst)
                    valid = 0
                    tmp = []
                    for (msg, attr) in zip(return_msg, attr_lst):
                        if "I apologize"  in msg or  "sorry"  in msg or  "Sorry" in msg or "an AI language model" in msg: # invalid contents
                            continue
                        else:
                            valid += 1
                            example = {"_id": label_ids, "label": labels, "text": clean_str(msg)}
                            example.update(attr)
                            examples.append( example)
                            tmp.append(example)
                    sent_cnt += valid 
                    prompt_lst = []
                    attr_lst = []
                    print(f"Subclass {i}: {re.sub(' ', '_', subtopic)}, Attempt: {attempt}, Sent cnt: {sent_cnt}. ")
                    prefix = f"gen_examples/{re.sub(' ', '_', subtopic)}/train_p{args.top_p}_{i}_{attempt}.jsonl"
                    os.makedirs(f"{args.output_dir}/gen_examples/{re.sub(' ', '_', subtopic)}", exist_ok= True)
                    f = open(f"{args.output_dir}/{prefix}", 'w')
                    for e in tmp:
                        f.write(json.dumps(e) + "\n")
                    f.close()

                except openai.error.RateLimitError:
                    print("Rate Limit Error! Attempt:", attempt)
                    prompt_lst = []
                    attr_lst = []
                    time.sleep(10)
                    continue
                except  openai.error.APIError:
                    print("API Error! Attempt:", attempt)
                    prompt_lst = []
                    attr_lst = []
                    time.sleep(5)
                    continue
                except openai.error.APIConnectionError:
                    print("APIConnectionError", attempt)
                    prompt_lst = []
                    attr_lst = []
                    time.sleep(5)
                    continue 
                except openai.error.InvalidRequestError:
                    print("API Error! Invalid Request:", attempt)
                    prompt_lst = []
                    attr_lst = []
                    continue
            if sent_cnt > args.n_sample or attempt >= 20:
                break
      

if __name__ == '__main__':
    main(args)

