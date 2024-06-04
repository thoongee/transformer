---
# Transformer

‘Attention Is All You Need’ implementation in Pytorch.


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

## Install

```bash
bash prepare.sh
```

## Run Train ([Multi30k](https://github.com/multi30k/dataset))

```bash
python3 main.py --dropout 0.1 --batch_size 256 --optimizer adam --model attention
```

## **Select best checkpoint**

```bash
python3 select_best_checkpoint.py --checkpoint-dir ./checkpoint --best-model-path ./best_model.pt
```
### Hyper parameter setting

- N_EPOCH = 100
- BATCH_SIZE = 256
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

```bash

```
