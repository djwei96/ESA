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
for example, if you want to train the model in dbpedia, the commands are as follows:
```linux
python main.py \
    --db_name dbpedia \
    --mode train \
    --transE_dim 100 \
    --pred_embedding_dim 100 \
    --lr 0.0001 \
    --clip 50 \
    --loss_function BCE \
    --regularization False \
    --n_epoch 50 \
    --save_every 2
```
if you want to test the model and generate entity summarization results, the commands are as follows:
```linux
python main.py \
    --db_name dbpedia \
    --model test \
    --use_epoch 48
```
we also provdie a mode called "all" to train and test the model at the same time, the commands are as follows:
```linux
python main.py \
    --db_name dbpedia \
    --mode all \
    --transE_dim 100 \
    --pred_embedding_dim 100 \
    --lr 0.0001 \
    --clip 50 \
    --loss_function BCE \
    --regularization False \
    --n_epoch 50 \
    --save_every 2 \
    --model test \
    --use_epoch 48
```
### Test
```linux
cd .../ESA
cd test
sh run.sh
```
