import random as python_random
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from pprint import pprint
import tensorflow as tf

# import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)

models_list = [
    'bert-base-uncased',
    'bert-large-uncased',
    'distilbert-base-uncased',
    'roberta-base',
    'typeform/distilbert-base-uncased-mnli',  # zero-shot
    'facebook/bart-large-mnli',  # Zero-shot
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
    parser.add_argument('--model', default='bert-base-uncased', choices=models_list + ['all'],
                        help="Which pre-trained model to use")
    parser.add_argument('--epochs', type=int, default=1, help='Epochs to fine-tune model for')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size: best in powers of 2')

    args = parser.parse_args()
    return args


def train_model(lm, tokens_train, Y_train_bin, tokens_dev, Y_dev_bin, num_labels, epochs=1, batch_size=8):
    print("Loading model....")
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=num_labels)
    loss_function = CategoricalCrossentropy(from_logits=True)
    optim = Adam(learning_rate=5e-5)
    print("Training model....")
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    model.fit(tokens_train, Y_train_bin, verbose=1, epochs=epochs,
              batch_size=batch_size, validation_data=(tokens_dev, Y_dev_bin))
    print("Done!")
    return model


def evaluate_model(lm, tokens_dev, Y_dev_bin, labels):
    print("Evaluating model....")
    Y_pred = lm.predict(tokens_dev)["logits"]
    # Get predictions using the trained model
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_dev_bin, axis=1)

    report = classification_report(Y_test, Y_pred, labels=labels, digits=3)
    print(report)
    cm = confusion_matrix(Y_test, Y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()


def main():
    args = create_arg_parser()
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

    model = train_model(lm, tokens_train, Y_train_bin, tokens_dev, Y_dev_bin, len(labels),
                        epochs=args.epochs, batch_size=args.batch_size)
    evaluate_model(model, tokens_dev, Y_dev_bin, labels)


if __name__ == "__main__":
    main()
