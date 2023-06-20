import numpy as np 
import json 
from thefuzz import fuzz
from collections import Counter


def load_class_name(dataset):
    with open(f"../datasets/{dataset}/label.txt", 'r') as f:
        qtext = [x.lower().replace(" ", "_").strip('\n') for x in f.readlines()]
        id2label = {x: i for (i, x) in enumerate(qtext)}
    return qtext, id2label

field = "subtopics"
dataset = "arxiv"
labels, id2label = load_class_name(dataset = dataset)

keys_label = {} 
keys = {}
for label in labels:
    keys_label[label] = label 
    keys[label] = [label]

for label in labels:
    with open(f"../datasets/{dataset}/attrprompt/gpt-3.5-turbo/{field}/{label}.jsonl", "r") as f:
        for lines in f:
            text = lines.replace("\n", "")
            text = text.lstrip('-').lstrip('0123456789.').strip("\"\',()[]").strip().lower()
            if text == "":
                continue
            if text in keys and label not in keys[text]:
                keys[text].append(label)
            else:
                keys[text] = [label]

remove_keyword_lst = []
for label in labels:
    with open(f"../datasets/{dataset}/attrprompt/gpt-3.5-turbo/{field}/{label}.jsonl", "r") as f:
        for lines in f:
            text = lines.replace("\n", "")
            text = text.lstrip('-').lstrip('0123456789.').strip("\"\',()[]").strip().lower()
            if text == "":
                continue
            for x in keys_label:
                if label != x:
                    if fuzz.ratio(text, x) >= 90 or x + " " in (text + " ") or text in x:
                        print(text, "orig", label, "overlap", x, fuzz.ratio(text, x), text in x)
                        if fuzz.ratio(text, x) >= 90 or (x + " ") in (text + " "):
                            remove_keyword_lst.append(x)
                            remove_keyword_lst.append(text)
                        else:
                            if x not in keys[text]:
                                keys[text] += [x]

for x in remove_keyword_lst:
    if x in keys:
        del keys[x]
for label in labels:
    if label in keys:
        del keys[label]

z = Counter()
for x in keys:
    for y in keys[x]:
        z[y] += 1


for y in keys:
    lbls = list(set(keys[y]))
    idxs = list(set([id2label[z] for z in keys[y]]))
    keys[y] = {"lbl": lbls, "lbl_idx": idxs, "n_lbl": len(idxs)}

with open(f"../datasets/{dataset}/attrprompt/gpt-3.5-turbo/subtopics_filter/subtopics.json", 'w') as f:
    json.dump(keys, f, indent=2)

# if len(keys[x]) > 2:
#     print(x, keys[x])
# for y in keys:
#     if fuzz.ratio(x, y) >= 85 and fuzz.ratio(x, y) <= 95:
#         print(x, y)
# continue 
# fuzz.ratio("this is a test", "this is a test!")