import openai
import asyncio
from typing import List, Dict, Any
import argparse
import os 
from tqdm import tqdm
import re 
import time
import json 

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

api_key = 'YOUR_OPENAI_API_KEY' # change this to your id

parser = argparse.ArgumentParser("")
parser.add_argument("--temperature", default=1, type=float, help="which seed to use")
parser.add_argument("--top_p", default=1.0, type=float, help="top_p for sampling")
parser.add_argument("--n_sample", default=10, type=int, help="number of examples to be generated")
parser.add_argument("--dataset", default='agnews', type=str, help="which model to use")

parser.add_argument("--model_name", default='gpt-3.5-turbo', type=str, help="which model to use")
parser.add_argument("--max_tokens", default=512, type=int, help="which seed to use")
parser.add_argument("--output_dir", default='.', type=str, help="the folder for saving the generated text")

args = parser.parse_args()
args.api_key = api_key

### Prompt Format ###
if args.dataset in ['nyt-fine', 'agnews']:
    args.domain = 'news'
    args.prefix = ''
elif args.dataset in ['yelp']:
    args.domain = 'restaurant review'
    args.prefix = ''
elif args.dataset in ['sst2']:
    args.domain = 'movie review'
    args.prefix = ''
elif args.dataset in ['amazon-product']:
    args.domain = 'review'
    args.prefix = 'a product of'
elif args.dataset in ['stackexchange', 'reddit']:
    args.prefix = ''
    args.domain = f"posts in {args.dataset}" 
else:
    raise NotImplementedError
#####################

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

    for i, class_name in tqdm(enumerate(label_names)):
        print(i, class_name)
        if args.dataset in ['sst2', 'yelp']:
            args.prompt = re.sub("_", " ", f"Suppose you are a writer for {args.domain}. Please give an example of a synthetic {class_name} {args.domain}.")
        else:
            args.prompt = re.sub("_", " ", f"Suppose you are a writer for {args.domain}. Please give an example of a synthetic {args.domain} about {args.prefix} {class_name}.")
        example_cnt = 0 
        while example_cnt < args.n_sample:
            prefix = f"{class_name}/train_p{args.top_p}_{i}_{j}.jsonl"
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
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                    )
                )
            except openai.error.RateLimitError:
                print(f"RateLimitError for class {i}.")
                time.sleep(10)
                continue
            except  openai.error.APIError:
                print("APIError for class {i}.")
                time.sleep(10)
                continue
            except openai.error.InvalidRequestError:
                print("InvalidRequestError!")
                continue

            ans = [clean_str(x['choices'][0]['message']['content']) for x in response]
            for text in ans:
                example_cnt += 1
                data = {"_id": i, "text": text}
                f.write(json.dumps(data) + '\n')

if __name__ == '__main__':
    main(args)

