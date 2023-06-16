import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import argparse
import os 

parser = argparse.ArgumentParser("")
parser.add_argument("--dataset", type=str, default=0)
parser.add_argument("--sentence_embedding_model", type=str, default=0)
parser.add_argument("--embedding_folder", type=str, default='')

args = parser.parse_args()

dataset = args.dataset
embedding_model = args.sentence_embedding_model

# Load embeddings
embeddings_train = np.load(os.path.join(args.embedding_folder, 'embeddings_train.npy'))
embeddings_test = np.load(os.path.join(args.embedding_folder, 'embeddings_test.npy'))

# Assuming you have an array of labels saved as a .npy file
labels_train = np.load(os.path.join(args.embedding_folder, 'labels_train.npy'))
labels_test = np.load(os.path.join(args.embedding_folder, 'labels_test.npy'))


# Train a logistic regression model
lr = LogisticRegression(max_iter=1300, verbose = 1)
lr.fit(embeddings_train, labels_train)

# Evaluate the model on the test set
y_pred = lr.predict(embeddings_test)
print(y_pred.shape)
accuracy = (y_pred == labels_test).mean()
f1 = f1_score(labels_test, y_pred, average='macro')
print(f"Dataset:{dataset}, embedding {embedding_model}, Train Test accuracy:", accuracy, "Test F1:", f1)
