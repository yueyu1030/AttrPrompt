import json 
import re 
from collections import defaultdict

def remove_numbered_lists(text):
    # Use regular expression to match numbered lists
    pattern = r'^\d+\.\s+'
    regex = re.compile(pattern, re.MULTILINE)
    
    # Remove numbered lists from the text
    text = regex.sub('', text)
    
    return text

def load_attributes(attr_name = "", model = "", dataset = "", method = "", classes = None):
    '''
        return a dictionary for class-dependent features, and a list for generic features   
    '''
    if attr_name in ["subtopics_filter", "subtopics", "keywords", "similar", "brands", "product_name", 'product_name_filter', \
        'resource',  'resource_filter', 'experience_filter', 'scenario', "scenario_filter", "technique"]: #'experience',
        assert classes is not None 
        return_dict = {} 
        for c in classes:
            return_dict[c] = []
            with open(f"../datasets/{dataset}/{method}/{model}/{attr_name}/{c}.jsonl", 'r') as f:
                for lines in f:
                    clean_text = remove_numbered_lists(lines.lstrip('0123456789.').lstrip('-()').strip("\"\'\n").strip())
                    if clean_text != "":
                        return_dict[c].append(clean_text)
            if attr_name not in "similar":
                return_dict[c].append(f"Others {attr_name} for {c}.")
        return return_dict

    else:
        lst = []
        with open(f"../datasets/{dataset}/{method}/{model}/{attr_name}/{attr_name}.txt", 'r') as f:
            for lines in f:
                lst.append(lines.strip("\n"))
        return lst 


if __name__ == "__main__":
    x=  load_attributes(attr_name = "subtopics", model = "gpt-3.5-turbo", dataset = "nyt-fine", method = "divgen", classes = ['abortion', "american_football"])
