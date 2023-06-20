# Classifier Training
This directory contains the code for training the classification model with the generated training data.

## Fine-tuning Pretrained Language Models
The code in in the `plm_model` folder. Use the following commands
```
bash commands/run_nyt.sh ${gpu}
```
where `gpu` is the id of gpu used for training.
In this script, `data_dir` is the directory for loading the train/test data. 

## Linear Probing
In this case, you need to first generate the embeddings using the commands as follows
```
python sentence_bert.py --gpu=${gpu} --dataset=${dataset} --method=${method} --sentence_embedding_model=${sentence_embedding_model}
--file_name=${file_name}
```
where
- `gpu` is the id of gpu used for inference;
- `dataset` is the dataset used for inference;
- `method` is the dataset generation methods (e.g. simprompt or attrprompt);
- `sentence_embedding_model` is the embedding model (e.g. SentenceBERT);
- `file_name` is the file name that contain the documents.

The generated embeddings will be saved in `./{args.dataset}/{args.sentence_embedding_model}` folder.

Then, use the following commands for linear probing:
```
python logistic_regression.py --dataset=${dataset} --sentence_embedding_model=${sentence_embedding_model} --embedding_folder=${embedding_folder}
```
where the `embedding_folder` is the folder for saving the embeddings generated from the last step.