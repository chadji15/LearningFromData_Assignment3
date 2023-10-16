# LearningFromData_Assignment3

# Project set-up

---

First download the dataset and put it in the `data` folder. Following that, run the following command to install the 
dependencies (with a Python 3.10 environment):

```python
pip install -R requirements.txt
```

Now split the dataset into separate train, validation and test files by running the following script:

```python
python3 split_train_val_test.py
````
## Classification using LSTMs
The files for this part exist in the `lstm` folder. 

The notebook containing the code and the development process is names `lstm.ipynb`, along with pre-generated output.

`lstm.py` contains the basic template file with the final LSTM model plugged into it, so it can easily be run if needed.

Before you can do that you do need to run the following from the root folder:

```shell
pip install lstm/requirements.txt
```

## 2

## Comparing multiple pre-trained language models

In the script _compare.py_ we implement some experiments that aim to
compare the performance of different pre-trained language models
on the downstream task. The chosen model is loaded and then finetuned, with
the exception of _facebook/bart-large-mnli_. The model is then
evaluated on a different chunk of the dataset and accuracy is reported,
along with the confusion matrix and f1-scores, precision and recall
for all the classes.

### Execution
```
usage: compare.py [-h] [-i TRAIN_FILE] [-d DEV_FILE] [-t TEST_FILE]
                  [--model {bert-base-uncased,bert-large-uncased,distilbert-base-uncased,roberta-base,facebook/bart-large-mnli,all}]
                  [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                  [--fig_path FIG_PATH] [--learning_rate LEARNING_RATE]

options:
  -h, --help            show this help message and exit
  -i TRAIN_FILE, --train_file TRAIN_FILE
                        Input file to learn from (default train.txt)
  -d DEV_FILE, --dev_file DEV_FILE
                        Separate dev set to read in (default dev.txt)
  -t TEST_FILE, --test_file TEST_FILE
                        If added, use trained model to predict on test set
  --model {bert-base-uncased,bert-large-uncased,distilbert-base-uncased,roberta-base,facebook/bart-large-mnli,all}
                        Which pre-trained model to use
  --epochs EPOCHS       Epochs to fine-tune model for
  --batch_size BATCH_SIZE
                        Batch size: best in powers of 2
  --fig_path FIG_PATH   Save location for confusion matrix
  --learning_rate LEARNING_RATE
                        Learning rate for fine tuning
```
