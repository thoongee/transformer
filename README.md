# Transformer

This project is an implementation and reproducibility study of the Transformer model as described in the paper 'Attention Is All You Need' using PyTorch.

---
## Project Overview
We aimed to:

- Reproduce the original Transformer performance claims (self-attention effectiveness, translation quality, scalability).
- Validate the model’s performance under limited computational resources.
- Conduct hyperparameter search experiments (batch size, optimizer, dropout).
- Perform an ablation study on the importance of self-attention.
- Extend the Transformer application to a Question Answering (QA) task using DistilBERT.

## Experiments

### 1. Transformer Reproduction
- Successfully reproduced Transformer using Multi30k.
- Achieved **BLEU score** of **30.4** on EN-DE translation — *higher* than the original paper's reported scores (Base: 27.3, Big: 28.4).

### 2. Hyperparameter Search
- **Batch Size**:
  - Best performance at batch size 128 (BLEU 30.9), but batch size 256 recommended for time-efficiency.
- **Optimizer**:
  - Adam optimizer yielded best results (lowest validation loss, BLEU 28.8).
- **Dropout Rate**:
  - Dropout 0.1 produced the best BLEU score (~28.8).

### 3. Ablation Study
- Replacing self-attention with Dense/LSTM/CNN layers drastically degraded translation performance.
- Confirmed self-attention's **critical** role in Transformer.

| Model                | BLEU  | Training Time (sec) |
|----------------------|-------|---------------------|
| w/ Self-attention     | 28.84 | 3218                |
| w/ Dense Layer        | 3.25  | 1484                |
| w/ LSTM Layer         | 3.21  | 1516                |
| w/ CNN Layer          | 3.31  | 1507                |

### 4. Application to Question Answering (QA)
- Extended the project using **DistilBERT** (a distilled Transformer model) for QA tasks.
- Evaluated on SQuAD v1.1 dataset:
  - Exact Match (EM): **80.33%**
  - F1 Score: **89.69%**

## Results Summary

| Task                                    | Result                              |
|-----------------------------------------|-------------------------------------|
| EN-DE Translation BLEU (Transformer)   | 30.4                               |
| Hyperparameter Search (Best BLEU)       | 30.9 (Batch Size 128)               |
| Ablation Study (Self-attention)         | Critical for good BLEU performance  |
| QA Task (DistilBERT on SQuAD)            | EM 80.33%, F1 89.69%                |

---
## Environment

- Driver version : 550.54.15
- cuda : 11.6
- cudnn : 8.4.0.27

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
- **Training dataset**: [Multi30k dataset](https://github.com/multi30k/dataset)  
  (due to computational constraints, instead of the original WMT 2014 dataset)

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
