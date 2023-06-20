import openai
import asyncio
from typing import List, Dict, Any
import argparse
import os 
from tqdm import trange, tqdm
import re 
import time
import json 
from utils import load_attributes, load_entity

# filter irrelevant subtopics & entities

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def split_list(lst, k):
    """Split a list into chunks of length k"""
    return [lst[i:i+k] for i in range(0, len(lst), k)]

api_key = 'YOUR_OPENAI_API_KEY' # change this to your id

parser = argparse.ArgumentParser("")
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--temperature", default=0.7, type=float)
parser.add_argument("--top_p", default=0.95, type=float)
parser.add_argument("--dataset", default='agnews', type=str, help="which dataset to use")
parser.add_argument("--attribute", default='entities', type=str, help="which attribute to use")


parser.add_argument("--model_name", default='gpt-3.5-turbo', type=str, help="which model to use")
parser.add_argument("--model_type", default='completion', type=str, help="which model type to use")
parser.add_argument("--max_tokens", default=128, type=int, help="maximum tokens")
parser.add_argument("--output_dir", default='.', type=str, help="the folder for saving the generated text")
parser.add_argument("--filter", default=1, type=int, help="the folder for saving the generated text")

args = parser.parse_args()
args.api_key = api_key
if args.dataset in ['nyt-fine']:
    args.domain = 'News'
    args.data_source = "NYT"
    args.metadata = 'location'
elif args.dataset in ['wos']:
    args.domain = 'Paper'
    args.data_source = "Web of Science"
elif args.dataset in ['amazon-product']:
    args.domain = 'product Review'
    args.data_source = "Amazon"
    args.metadata="product_name"
elif args.dataset in ['arxiv']:
    args.domain = 'paper'
    args.data_source = "arxiv"
    args.metadata = "subtopics"
elif args.dataset in ['stackexchange', 'reddit']:
    args.domain = 'Web Forum'
    args.data_source = args.dataset
    args.metadata="experience"
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

def call_api_async(msg_lst, model):
    print("===================================")
    print(f"call APIs, {len(msg_lst)} in total.")
    l = len(msg_lst)

    response = asyncio.run(
        dispatch_openai_requests(
            messages_list = msg_lst,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
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
    prompt = args.prompt
    openai.api_key = args.api_key
    return_dict = {} 

    # Call the function to clear the terminal screen
    # clearScreen()

    attributes = load_attributes(attr_name = args.attribute, model = args.model_name, dataset = args.dataset, method = args.model_type, classes = label_names)

    similar_keywords = load_attributes(attr_name = 'similar', model = args.model_name, dataset = args.dataset, method = args.model_type, classes = label_names)
    # print(attributes, similar_keywords)
    # exit()
    i = 0
    max_reject = 5
    for c in tqdm(label_names):
        i += 1        
        similar_keyword = ",".join(similar_keywords[c])
        args.prompt = [re.sub("_", " ", f"Consider {x}. Is it relevant to the following categories: {similar_keyword}? Return 1 for yes and 0 for no.") for x in attributes[c]]
        
        print(f"Consider {attributes[c][0]}. Is it relevant to the following categories: {similar_keyword}? Return 1 for yes (related to any of these categories) and 0 for no. You only need to return one number.")
        n_reject = 0
        args.prompt = split_list(args.prompt, 25)
        for prompt in args.prompt:
            msg_lst = [
                        [{"role": "user", "content": p}] for p in prompt
                    ] 
            succeed = 0
            while succeed == 0:
                try:
                    return_msg = call_api_async(msg_lst, model)
                    print(attributes[c], return_msg)
                    assert len(return_msg) == len(prompt) and len(prompt) == len(attributes[c])
                    prefix = f"{args.attribute}_filter/{c}.jsonl"
                    os.makedirs(f"{args.output_dir}/{args.attribute}_filter/", exist_ok= True)
                    f = open(f"{args.output_dir}/{prefix}", 'w')
                    
                    for msg, x, attr in zip(return_msg, prompt, attributes[c]):                        
                        msg = msg.lower()
                        if n_reject < max_reject and ('yes' in msg or '1' in msg):
                            n_reject += 1
                            print(f"Removing {attr} for {c}!")
                        else:
                            f.write(attr + '\n')
                            
                    succeed = 1
                    # exit()
                except openai.error.RateLimitError:
                    print("Rate Limit Error!")
                    time.sleep(10)
                except  openai.error.APIError:
                    print("API Error!")
                    time.sleep(5)
                except openai.error.InvalidRequestError:
                    print("API Error! Invalid Request")

if __name__ == '__main__':
    main(args)

