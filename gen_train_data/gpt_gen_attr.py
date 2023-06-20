import openai
import asyncio
from typing import List, Dict, Any
import argparse
import os 
from tqdm import trange, tqdm
import re 
import time
import json 
from utils import load_attributes


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

api_key = 'YOUR_OPENAI_API_KEY' # change this to your id

parser = argparse.ArgumentParser("")
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--temperature", default=1, type=float)
parser.add_argument("--top_p", default=0.95, type=float)
parser.add_argument("--n_attribute", default=10, type=int)
parser.add_argument("--dataset", default='agnews', type=str)
parser.add_argument("--attribute", default='subtopics', type=str) 

parser.add_argument("--model_name", default='gpt-3.5-turbo', type=str, help="which model to use")
parser.add_argument("--model_type", default='attrprompt', type=str, help="which model type to use")
parser.add_argument("--max_tokens", default=2048, type=int)
parser.add_argument("--output_dir", default='.', type=str, help="the folder for saving the generated text")

args = parser.parse_args()
args.api_key = api_key
if args.dataset in ['nyt-fine']:
    args.domain = 'News'
    args.data_source = "NYT"
elif args.dataset in ['arxiv']:
    args.domain = 'Papers'
    args.data_source = "arxiv"
elif args.dataset in ['amazon-product']:
    args.domain = 'product'
    args.data_source = "Amazon"
elif args.dataset in ['stackexchange', 'reddit']:
    args.domain = 'Web Post'
    
    args.data_source = args.dataset
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
            temperature=1.0,
            max_tokens=400,
            top_p=1.0,
        )
    )
    ans = [x['choices'][0]['message']['content'] for x in response]
    print(f"API returns {len(ans)} in total.")
    print("===================================")
    return ans 

def main(args):

    with open(f"./datasets/{args.dataset}/label.txt", 'r') as f:
        label_names = [x.lower().replace(" ", "_").strip('\n') for x in f.readlines()]
    print(label_names)
    model = args.model_name
    openai.api_key = args.api_key

    ###
    if args.attribute == 'similar': # find the similar topics
        args.prompt = [re.sub("_", " ", f"Give 5 most relevant classes related to {x} {args.domain} on {args.data_source} from the classes: {json.dumps(label_names)}. Write each class in a line. Do not include {x}.") for x in label_names]
        print(f"Generate a {args.domain} using model {model}, prompt: {args.prompt}")
    else:
        similar_keywords = load_attributes(attr_name = args.attribute, model = args.model_name, dataset = args.dataset, method = args.model_type, classes = label_names)
        args.prompt = [re.sub("_", " ", f"List {args.n_sample} diverse {args.attribute} related to {label_names[i]} on {args.data_source}. These {args.attribute} should be unrelated to  {similar_keywords[x]}.") for (i,x) in enumerate(label_names)]

    msg_lst = [
                [{"role": "user", "content": p}] for p in args.prompt
            ]
    success = 0 
    while success == 0:
        try:
            return_msg = call_api_async(msg_lst, model)
            success = 1
        except openai.error.RateLimitError or openai.error.RateLimitError:
            print("Not succeed. Try again.")
            continue 

    assert len(return_msg) == len(label_names)
    for msg, x in zip(return_msg, label_names):
        class_name = re.sub(" ", "_", x)
        prefix = f"{args.attribute}/{class_name}.jsonl"
        os.makedirs(f"{args.output_dir}/{args.attribute}/", exist_ok= True)
        f = open(f"{args.output_dir}/{prefix}", 'w')
        f.write(msg + '\n')

if __name__ == '__main__':
    main(args)

