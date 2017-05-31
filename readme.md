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

##Tweaks 
You can see all script parameters by invoking:
```
$ python trainer/create_model.py -h
```

##TODO
- Write guide how to train this on google cloud
- Add possibility to use pretrained word embeddings 
