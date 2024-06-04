# Transformer

‘Attention Is All You Need’ implementation in Pytorch.

---
## Project Overview
This project is an implementation of the Transformer model as described in the paper 'Attention Is All You Need' using Pytorch. The Transformer model is designed for sequence-to-sequence tasks, primarily focusing on natural language processing tasks like translation.

## Environment

Driver version : 550.54.15

cuda : 11.6

cudnn : 8.4.0.27

### Using virtual environment

1. Install virtualenv via pip
    
    `pip install virtualenv`
    
2. Create a virtual environment with virtualenv
    
    `virtualenv [example] --python=3.8`
    
3. Run a virtualenv created via source (Linux)
    
    `source [example]/bin/activate`
    
4. Terminate a running virtual environment
    
    `deactivate`
    

---
## Prerequisties
- Python 3.8
- Pytorch
- Other dependencies listed in `requirements.txt`

## Dataset
We use the Multi30k dataset for training.

## Install
Install all required dependencies and download the Multi30k dataset by running:

```bash
bash prepare.sh
```

## Usage

### Run Training and Evaluation
To start training and evaluation with the Multi30k dataset, run:

```bash
python3 main.py
```

### Select best checkpoint
To select the best model checkpoint, run:

```bash
python3 select_best_checkpoint.py --checkpoint-dir ./checkpoint --best-model-path ./best_model.pt
```
### Hyper parameter setting

- N_EPOCH = 1000
- BATCH_SIZE = 512
- NUM_WORKERS = 8
- LEARNING_RATE = 1e-5
- WEIGHT_DECAY = 5e-4
- ADAM_EPS = 5e-9
- SCHEDULER_FACTOR = 0.9
- SCHEDULER_PATIENCE = 10
- WARM_UP_STEP = 100
- DROPOUT_RATE = 0.1
### Best checkpoint

- epoch : 372
- validation loss : 1.77118
- BLEU score : 30.43513

---

## Reference

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [Harvard NLP](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [hyunwoongko/transformer](https://github.com/hyunwoongko/transformer)
- [WikiDocs](https://wikidocs.net/31379)
