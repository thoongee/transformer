# transformer
Transformer(Attention Is All You Need) Implementation in Pytorch.

A detailed implementation description is provided in the post below.
#### [Blog Post](https://cpm0722.github.io/pytorch-implementation/transformer)

---

#### Install
```bash
bash prepare.sh
```

#### Run Train ([Multi30k](https://github.com/multi30k/dataset))
```bash
python3 main.py
```

---

#### Reference

- ##### [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- ##### [Harvard NLP](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- ##### [hyunwoongko/transformer](https://github.com/hyunwoongko/transformer)
- ##### [WikiDocs](https://wikidocs.net/31379)

Transformer(Attention Is All You Need) Implementation in Pytorch

---

**Environment**

Driver version : 550.54.15

cuda : 11.6

cudnn : 8.4.0.27

**Virtual environment**

*# pip를 통해 virtualenv 설치*

`pip install virtualenv`

*#  virtualenv를 이용해 가상환경 생성*

`virtualenv [example] --python=3.8`

*# source를 통해 생성한 가상환경 실행 (Linux)*

`source [example]/bin/activate`

*# 실행 중인 가상환경 종료*

`deactivate`

---

**Install**

```
bash prepare.sh
```

**Run Train ([Multi30k](https://github.com/multi30k/dataset))**

```
python3 main.py
```

**Select best checkpoint**

```
python3 select_best_checkpoint.py --checkpoint-dir ./checkpoint --best-model-path ./best_model.pt
```

---

**Reference**

- https://github.com/cpm0722/transformer_pytorch
- [Blog Post](https://cpm0722.github.io/pytorch-implementation/transformer)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [Harvard NLP](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [hyunwoongko/transformer](https://github.com/hyunwoongko/transformer)
- [WikiDocs](https://wikidocs.net/31379)
