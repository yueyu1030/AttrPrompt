import openai
import asyncio
from typing import List, Dict, Any
import argparse
import os 
from tqdm import trange, tqdm
import re 
import time
import random 
import numpy as np 
import json 

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

api_key = 'YOUR_OPENAI_API_KEY' # change this to your id

parser = argparse.ArgumentParser("")
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--temperature", default=1, type=float)
parser.add_argument("--top_p", default=0.95, type=float)
parser.add_argument("--n_sample", default=10, type=int)
parser.add_argument("--dataset", default='agnews', type=str, help="which dataset to use")

parser.add_argument("--model_name", default='gpt-3.5-turbo', type=str, help="which model to use")
parser.add_argument("--model_type", default='completion', type=str, help="which model type to use")
parser.add_argument("--max_tokens", default=2048, type=int, help="max tokens to use")
parser.add_argument("--output_dir", default='.', type=str, help="the folder for saving the generated text")

args = parser.parse_args()
args.api_key = api_key

if  args.dataset in ['arxiv']:
    args.domain = 'scientific paper abstract'
    args.prefix = ""
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


def main(args):

    with open(f"../datasets/{args.dataset}/label.txt", 'r') as f:
        label_names = [x.lower().strip('\n') for x in f.readlines()]
    print(label_names)
    openai.api_key = args.api_key

    # Call the function to clear the terminal screen
    # clearScreen()

    # Enter a while loop to continually prompt the user for input and generate responses from the OpenAI API
    # while True:
    # Check this url for more info about the parameters
    # https://platform.openai.com/docs/api-reference/completions/create
    attempt = 0
    for i, class_name in tqdm(enumerate(label_names)):
        print(i, class_name)
        j = 0
        tot_cnt = 0
        print(f"Generate a {args.domain} using prompt: {args.prompt}, t= {args.temperature}")
        candidate_class = [ (cls_id, x) for cls_id, x in enumerate(label_names) if x != class_name]
        while tot_cnt < args.n_sample:            
            random.shuffle(candidate_class)
            num_add_topics = min(4, np.random.zipf(2.5, 1)[0]) - 1
            if num_add_topics == 0:
                class_name_gen = class_name
                class_idx = [i]
            else:
                class_name_gen = class_name + ", " +  ",".join([x[1] for x in candidate_class[:num_add_topics]])
                
                class_idx = [i] + [x[0] for x in candidate_class[:num_add_topics]]
            args.prompt = re.sub("_", " ", f"Suppose you are a writer for {args.domain}. Give a synthetic {args.domain} about {args.prefix} {class_name_gen}.")
            if j % 10 == 0:
                print("Prompt: ", args.prompt, "\tClass Name: ", class_name_gen, "\tClass idx: ", class_idx)

            prefix = f"{class_name}/train_t{args.temperature}_p{args.top_p}_{i}_{j}.jsonl"
            j += 1
            try:
                os.makedirs(f"{args.output_dir}/{class_name}/", exist_ok= True)
                f = open(f"{args.output_dir}/{prefix}", 'w')
                
                response = asyncio.run(
                    dispatch_openai_requests(
                        messages_list=[
                            [{"role": "user", "content": args.prompt}],
                        ] * 25,
                        model="gpt-3.5-turbo",
                        temperature=args.temperature,
                        max_tokens=300,
                        top_p=1.0,
                    )
                )
            except openai.error.RateLimitError:
                print(f"Error for class {i}")
                time.sleep(10)
                continue
            except  openai.error.APIError:
                print("API Error! Attempt:")
                time.sleep(10)
                # exit()
                continue
            except openai.error.InvalidRequestError:
                print("InvalidRequestError!")
                # exit()
                time.sleep(5)
                continue
            except openai.error.APIConnectionError:
                print("APIConnectionError", attempt)
                # exit()
                time.sleep(5)
                continue 
            ans = [clean_str(x['choices'][0]['message']['content']) for x in response]
            for a in ans:
                example = {"_id": class_idx, "label": class_name_gen, "text": a}
                f.write(json.dumps(example) + '\n')
            tot_cnt += len(ans)
            print(f"total generated examples: {tot_cnt}")
           

if __name__ == '__main__':
    main(args)

