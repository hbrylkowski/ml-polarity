# Implementation of Sentiment Analysis on movie reviews
This implementation is written with Keras and runnable on gcloud ml-engine

Dataset is 10662 movie reviews - half positive and half negative. 
Goal is to correctly label them. Data is split randomly into three portions in ratio 8:1:1
Biggest is training set, smaller ones are test and validation set.

##Run local
Install dependencies (If you already have tf with GPU support install keras without dependencies)

```bash
$ pip install -r requirements.txt
```
Set up environmental variables
```bash
$ source ./script/local_env.sh
```
Create model
```bash
$ python trainer/create_model.py \
    --train-file $TRAIN_DATA \          
    --eval-file $EVAL_DATA \
    --job-dir $MODEL_DIR
```

## Pretrained word embeddings
You have to provide argument `--word-vectors-path` with path to binary vectors file and `--embedding-size` 
(for GoogleNews it 300)

[Google news vectors download](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

## Tweaks 
You can see all script parameters by invoking:
```
$ python trainer/create_model.py -h
```

## Accuracy
loss: 0.4827 - acc: 0.7702 - batch_size 1


## TODO
- Write guide to predict on saved model
- Write guide how to train this on google cloud
