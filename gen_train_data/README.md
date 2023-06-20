# Generating training data with LLMs

This folder contains the code for generating training data using LLMs.


# Setup
You need to first have an OpenAI API key (see [here](https://openai.com/blog/openai-api)).


# Multi-class Classification
## Generating training data with simple prompts (SimPrompt)
Example Command: 
```
python gpt_gen_simprompt.py --dataset=nyt --temperature=1.0 --top_p=1.0 --n_sample=346 --model_name=gpt-3.5-turbo
```
Please specify the `output_folder`, which will be the place for saving the generated training data.

## Generating training data with attributed prompts (AttrPrompt)
### Prompting LLM for Attribute Type Generation
We recommend to use the [Online ChatGPT Platform](https://chat.openai.com/) for this step since it only need one forward step. An example of the forward process is listed as follows:
```
>  What do you think are important attributes to generate some diverse news from NYT under a specific topic, such as Football, Federal Budget. Examples: subtopics, writing style...

> Important attributes of news from the New York Times under a specific topic could include:

Subtopics: exploring different angles and subtopics within the main topic can lead to a wider range of news stories.

Writing style: varying the writing style can help to make the news stories more interesting and engaging for readers. 

Sources: using a variety of sources, such as experts, eyewitnesses, and affected individuals, can provide different perspectives and add diversity to the news stories.

Geographical locations: covering the topic from different geographical locations can help to show how it affects different communities and provide a global perspective.
...
```
In this way, we will include `subtopics`, `writing style`, `location` as specific attributes. The generated subtopics for each dataset can be found at the `datasets/${dataset_name}` folder. 

### Generate values for each attribute
Use the following scripts for generating the most similar classes
```
python gpt_gen_attr.py --dataset=${dataset} --attribute=similar --n_attribute=5 --output_dir=../datasets/attrprompt/gpt-3.5-turbo
```
Where 
- `dataset` is the name of dataset, e.g. nyt;
- `n_attribute` is the number of attribute values, we use 5 in our work;
- `attribute` is the attribute name, e.g. subtopics.
- `output_dir` is the directory for generated attributes. Note that there is a folder for each class. 

Then, with the similar classes, we generate the attributed value using the following scripts:
```
python gpt_gen_attr.py --dataset=${dataset} --attribute=${attribute} --n_attribute=${n_attribute} --output_dir=../datasets/attrprompt/gpt-3.5-turbo
```
Where 
- `dataset` is the name of dataset, e.g. nyt;
- `n_attribute` is the number of attribute values, e.g. 10;
- `attribute` is the attribute name, e.g. subtopics.
- `output_dir` is the directory for generated attributes. Note that there is a folder for each class. 

### Filtering Attributes Using LLMs
Use the following commands to filter value for the class-dependent attributes.
```
python filter.py --dataset=${dataset} --attribute=${attribute} --output_dir=../datasets/attrprompt/gpt-3.5-turbo
```

### Generate training data with attributes
Use the following commands to generate training data with attributes.
```
python gpt_gen_attrprompt.py --dataset=${dataset} --batch_size=20 --n_sample=500 --top_p=1.0 --temperature=1.0 --output_dir=../datasets/attrprompt/gpt-3.5-turbo
```

# Multi-label Classification
## Generating training data with simple prompts (SimPrompt)
Example Command: 
```
python gpt_gen_simprompt_multi_label.py --dataset=arxiv --temperature=1.0 --top_p=1.0 --n_sample=280 --model_name=gpt-3.5-turbo
```
Please specify the `output_folder`, which will be the place for saving the generated training data.

## Generating training data with attributed prompts (AttrPrompt)
After generate the subtopics, instead of filtering the ambiguous subtopics, we use the following commands to merge the subtopics as
```
python merge_subtopics_multi_label.py
```

Then, use tfollowing commands to generate training data with  attributes.
```
python gpt_gen_attrprompt_multi_label.py --dataset=${dataset} --batch_size=20 --n_sample=500 --top_p=1.0 --temperature=1.0 --output_dir=../datasets/attrprompt/gpt-3.5-turbo
```