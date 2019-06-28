# ESA: Entity Summarization with Attention
Entity Summarization with Attention
## BACKGROUND AND CONTRIBUTORS
### Background
This [paper](https://arxiv.org/abs/1905.10625) is in procedings of CIKM 2019, we propose a novel tenchniuqe in entity summarization task with attention mechanism.
### Contributors
- **Dongjun Wei** weidongjun@iie.ac.cn
- **Yaxin Liu** liuyaxin@iie.ac.cn
## ENVIRONMENT AND DEPENDENCY
### Environment
- Ubuntu 16.04
- python 3.5+
- pytorch 1.0.1
- java 8
### Dependency
```python
pip install numpy
pip install tqdm
```
## USAGE
### Train
```linux
git clone git@github.com:WeiDongjunGabriel/ESA.git
cd .../ESA
cd model
python main.py
```
we also provide a commandline tool for training the ESA model, you can also run the following command for more details:
```linux
python main.py -h
```
### Test
```linux
cd .../ESA
cd test
sh run.sh
```
