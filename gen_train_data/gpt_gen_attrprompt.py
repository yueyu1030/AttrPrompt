import openai
import asyncio
from typing import List, Dict, Any
import argparse
import os 
from tqdm import trange, tqdm
import re 
import time
from utils import load_attributes, load_entity
import numpy as np
import json 

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

api_key = 'YOUR_OPENAI_API_KEY' # change this to your id

parser = argparse.ArgumentParser("")
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--temperature", default=1, type=float, help="which seed to use")
parser.add_argument("--top_p", default=0.95, type=float, help="which seed to use")
parser.add_argument("--n_sample", default=10, type=int, help="the number of examples generated for each class")
parser.add_argument("--batch_size", default=20, type=int, help="")

parser.add_argument("--dataset", default='agnews', type=str, help="which model to use")

parser.add_argument("--model_name", default='gpt-3.5-turbo', type=str, help="which model to use")
parser.add_argument("--model_type", default='completion', type=str, help="which model type to use")
parser.add_argument("--max_tokens", default=2048, type=int, help="which seed to use")
parser.add_argument("--output_dir", default='.', type=str, help="the folder for saving the generated text")

args = parser.parse_args()
args.api_key = api_key
# args.prompt = "Please generate 10 news about business."
# class_name ='business'
# args.prompt = "Please generate 10 book, novel, or publication with descriptions."
# class_name ='WrittenWork'
if args.dataset in ['nyt-fine']:
    args.domain = 'news'
    args.background = ""
    args.attributes = ["length", "location", "subtopics",  "style", "similar"]
elif args.dataset in ['agnews']:
    args.domain = 'news'
    args.attributes = ["length", "location", "subtopics",  "style"]
elif args.dataset in ['sst2']:
    args.domain = 'movie review'
    args.attributes = ["length", "genre", "subtopics",  "style", "location"]
elif args.dataset in ['yelp']:
    args.domain = 'restaurant review'
    args.attributes = ["length", "cuisine", "subtopics",  "style"]
elif args.dataset in ['wos']:
    args.domain = 'scientific paper'
elif args.dataset in ['amazon-product']:
    args.domain = 'review'
    args.attributes = ["length", "brands", "product_name", "experience", "similar", 'style']
elif args.dataset in ['reddit']:
    args.domain = 'web forum'
    args.attributes = ["length", "experience", "resource", "similar", 'style']
elif args.dataset in ['stackexchange']:
    args.domain = 'web forum'
    args.attributes = ["length", "scenario", "depth", "similar", 'style']
else:
    raise NotImplementedError

def gen_example(attr_dict):
    lengths = {}
    for x in attr_dict:
        lengths[x] = len(attr_dict[x])
    while True:
        return_dict = {}
        for z in lengths:
            
            idx_z = np.random.randint(low = 0, high = lengths[z], dtype = int)
            return_dict[z] = idx_z 
            # lst.append(return_dict)
      
        yield return_dict

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
    print(label_names)
    model = args.model_name
    openai.api_key = args.api_key
    
    attr_dict = {}
    for attr in args.attributes:
        if 'subtopics' in attr:
            attr_name = 'subtopics_filter'
        elif 'product_name' in attr:
            attr_name = 'product_name_filter'
        elif 'experience' in attr:
            attr_name = 'experience_filter'
        elif 'resource' in attr:
            attr_name = 'resource_filter'
        elif 'scenario' in attr:
            attr_name = 'scenario_filter'
        else:
            attr_name = attr 
        attr_dict[attr] = load_attributes(attr_name = attr_name, model = model, dataset = args.dataset, method = args.model_type, classes = label_names)

    for i, class_name in tqdm(enumerate(label_names)):
        print(i, class_name)
        print(f"Prompt, Give a synthetic sample of {args.domain} about {re.sub('_', ' ', class_name)} following the requirements below")
        sent_cnt = 0
        attempt = 0
        attr_dict_cls = {}
        for attr in attr_dict:
            if attr in ['subtopics', 'similar', 'brands', 'product_name', 'product_name_filter', "experience", "resource", "scenario", "scenario_filter"]:
                attr_dict_cls[attr] = attr_dict[attr][class_name]
            else:
                attr_dict_cls[attr] = attr_dict[attr]
        prompt_lst = []
        attr_lst = []
        examples = []
        if "similar" in attr_dict:
            similar_label = ",".join(attr_dict["similar"][class_name])
        for return_dict in gen_example(attr_dict_cls):
            prompt_tmp = {x: attr_dict_cls[x][return_dict[x]] for x in return_dict}
            attr_lst.append(return_dict)
            if args.dataset ==  'amazon-product':

                prompt_input = f"Write a review for  {re.sub('_', ' ', class_name)} product in Amazon, following the requirements below: \n \
                            1. the review should be about the product of '{prompt_tmp['product_name']}';\n \
                            2. the brand of the {re.sub('_', ' ', class_name)} product should be '{prompt_tmp['brands']}'; \n \
                            3. should be in length between {prompt_tmp['length']} words and {int(prompt_tmp['length']) + 50} words;\n \
                            4. should describe the usage experience: {prompt_tmp['experience']};\n \
                            5. the review should be focus on '{prompt_tmp['style']}';\n \
                            6. the review must be relevant to {re.sub('_', ' ', class_name)} and irrelevant to: {similar_label}."
            elif args.dataset ==  'sst2':
                prompt_input = f"Write a {re.sub('_', ' ', class_name)} review for a movie, following the requirements below: \n \
                            1. the overall rating should be {re.sub('_', ' ', class_name)};\n \
                            2. the review should discuss about a {prompt_tmp['genre']} movie;  \n \
                            3. the review should focus on '{prompt_tmp['subtopics']}'; \n \
                            4. should be in length between {prompt_tmp['length']} words and {int(prompt_tmp['length']) + 50} words;\n \
                            5. the style of the review should be '{prompt_tmp['style']}'"
            elif args.dataset ==  'yelp':
                prompt_input = f"Write a {re.sub('_', ' ', class_name)} review for a restaurant, following the requirements below: \n \
                            1. the overall review should be {re.sub('_', ' ', class_name)}';\n \
                            2. should be a '{prompt_tmp['cuisine']}' restaurant';  \n \
                            3. should focus on '{prompt_tmp['subtopics']}'; \n \
                            4. should be in length between {prompt_tmp['length']} words and {int(prompt_tmp['length']) + 50} words;\n \
                            5. the style of the review should be '{prompt_tmp['style']}'"
            elif args.dataset == 'reddit':
                prompt_input = f"Give a synthetic sample of post in reddit on {re.sub('_', ' ', class_name)} community following the requirements below: \n\
                            1. should focus on '{prompt_tmp['experience']}';\n \
                            2. should be in length between {prompt_tmp['length']} words and {int(prompt_tmp['length']) + 50} words;\n \
                            3. The writing style of the post should be '{prompt_tmp['style']}';\n \
                            4. should mention the resource of {prompt_tmp['resource']}; \n \
                            5. The post must be relevant to {re.sub('_', ' ', class_name)} community and irrelevant to the following community: {similar_label}."

            elif args.dataset == 'nyt-fine':
                prompt_input = f"Give a synthetic sample of news in NYT on {re.sub('_', ' ', class_name)} following the requirements below: \n\
                            1. should focus on '{prompt_tmp['subtopics']}';\n \
                            2. should be in length between {prompt_tmp['length']} words and {int(prompt_tmp['length']) + 50} words;\n \
                            3. The writing style of the news should be '{prompt_tmp['style']}';\n \
                            4. The location of the news should be in {prompt_tmp['location']}; \n \
                            5. The news must be relevant to {re.sub('_', ' ', class_name)} and irrelevant to: {similar_label}."
            elif args.dataset == 'agnews':
                prompt_input = f"Give a synthetic sample of news on {re.sub('_', ' ', class_name)} following the requirements below: \n\
                            1. should focus on '{prompt_tmp['subtopics']}';\n \
                            2. should be in length between {prompt_tmp['length']} words and {int(prompt_tmp['length']) + 50} words;\n \
                            3. The writing style of the news should be '{prompt_tmp['style']}';\n \
                            4. The location of the news should be in {prompt_tmp['location']}; \n"

            elif args.dataset == 'stackexchange':
                prompt_input = f"Give a synthetic sample of question post in {args.dataset} on {re.sub('_', ' ', class_name)} following the requirements below: \n\
                            1. should focus on the scenario of '{prompt_tmp['scenario']}';\n \
                            2. should be in length between {prompt_tmp['length']} words and {int(prompt_tmp['length']) + 50} words;\n \
                            3. The question should be in {prompt_tmp['depth']}; \n \
                            4. The writing style of the question should be '{prompt_tmp['style']}';\n."

            if attempt == 0 and len(prompt_lst) == 0:
                print(f"Prompt Input: {prompt_input}")

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
                        if "I apologize"  in msg or  "sorry"  in msg or  "Sorry" in msg or  "an AI language model" in msg or "I cannot perform" in msg:
                            continue
                        else:
                            valid += 1
                            example = {"_id": i, "text": clean_str(msg)}
                            example.update(attr)
                            examples.append( example)
                            tmp.append(example)
                    sent_cnt += valid 
                    prompt_lst = []
                    attr_lst = []
                    print(f"CLass {i}: {class_name}, Attempt: {attempt}, Sent cnt: {sent_cnt}. ")
                    prefix = f"gen_examples/{class_name}/train_p{args.top_p}_{i}_{attempt}.jsonl"
                    os.makedirs(f"{args.output_dir}/gen_examples/{class_name}", exist_ok= True)
                    f = open(f"{args.output_dir}/{prefix}", 'w')
                    for e in tmp:
                        f.write(json.dumps(e) + "\n")
                    f.close()

                except openai.error.RateLimitError:
                    print("Rate Limit Error! Attempt:", attempt)
                    prompt_lst = []
                    attr_lst = []
                    time.sleep(15)
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
                    print("InvalidRequestError! Invalid Request:", attempt)
                    prompt_lst = []
                    attr_lst = []
                    continue
                except openai.error.Timeout:
                    print("Timeout Error! Invalid Request:", attempt)
                    prompt_lst = []
                    attr_lst = []
                    continue
            if sent_cnt >= args.n_sample or attempt > 200:
                break
       

if __name__ == '__main__':
    main(args)

