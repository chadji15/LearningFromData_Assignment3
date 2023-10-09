import random as python_random
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, pipeline
from pprint import pprint
import tensorflow as tf
import matplotlib.pyplot as plt

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)

models_list = [
    'bert-base-uncased',
    'bert-large-uncased',
    'distilbert-base-uncased',
    'roberta-base'
]

zero_shot_models = [
    'facebook/bart-large-mnli'  # Zero-shot
]


def read_corpus(corpus_file):
    '''Read in review data set and returns docs and labels'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip()
            documents.append(" ".join(tokens.split()[3:]).strip())
            # 6-class problem: books, camera, dvd, health, music, software
            labels.append(tokens.split()[0])
    return documents, labels


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='../data/train.txt', type=str,
                        help="Input file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", type=str, default='../data/dev.txt',
                        help="Separate dev set to read in (default dev.txt)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    parser.add_argument('--model', default='distilbert-base-uncased', choices=models_list + zero_shot_models +['all'],
                        help="Which pre-trained model to use")
    parser.add_argument('--epochs', type=int, default=1, help='Epochs to fine-tune model for')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size: best in powers of 2')
    parser.add_argument('--fig_path', type=str, help='Save location for confusion matrix')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='Learning rate for fine tuning')
    args = parser.parse_args()
    return args


def train_model(lm, tokens_train, Y_train_bin, num_labels, epochs=1, batch_size=8, learning_rate=5e-5):
    print("Loading model....")
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=num_labels)
    #loss_function = CategoricalCrossentropy(from_logits=True)
    optim = Adam(learning_rate=learning_rate)
    print("Training model....")
    model.compile( optimizer=optim, metrics=['accuracy'])
    model.fit(tokens_train, Y_train_bin, verbose=1, epochs=epochs,
              batch_size=batch_size)
    print("Done!")
    return model


def evaluate_model(lm, tokens_dev, Y_dev_bin, labels, figpath):
    print("Evaluating model....")
    pred = lm.predict(tokens_dev)["logits"]
    # Get predictions using the trained model
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_dev_bin, axis=1)

    report = classification_report(Y_test, Y_pred, target_names=labels, digits=3)
    print(report)
    cm = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig(figpath)

def evaluate_zero_shot(model, tokens_dev, Y_test, labels, figpath):
    output = model(tokens_dev, labels, multilabel=False)
    Y_pred = [pred["labels"][0] for pred in output]

    report = classification_report(Y_test, Y_pred, digits=3)
    print(report)
    cm = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(figpath)

def main():

    lm = args.model

    # Read in the data and embeddings
    print("Execution args:")
    pprint(args)
    print("..........................\n")
    print("Loading data...")
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)
    labels = encoder.classes_

    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_train = tokenizer(X_train, padding=True, max_length=100,
                             truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=100,
                           truncation=True, return_tensors="np").data

    if not args.fig_path:
        args.fig_path = f"{args.model}_confusion_matrix.png".replace('/', '')

    if args.model in zero_shot_models:
        model = pipeline('zero-shot-classification', model=args.model)
        evaluate_zero_shot(model,X_dev,Y_dev, labels, args.fig_path)
    else:
        model = train_model(lm, tokens_train, Y_train_bin,  len(labels),
                            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)

        evaluate_model(model, tokens_dev, Y_dev_bin, labels,args.fig_path)


if __name__ == "__main__":
    args = create_arg_parser()
    if args.model == 'all':
        for m in models_list + zero_shot_models:
            args.model = m
            args.fig_path = None
            main()
    else:
        main()
