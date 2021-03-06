# Sentiment Analysis

## Install PhoBert

PhoBert-base

```shell
$wget https://public.vinai.io/PhoBERT_base_transformers.tar.gz
$tar -xzvf PhoBERT_base_transformers.tar.gz
```

PhoBert-large

```shell
$wget https://public.vinai.io/PhoBERT_large_transformers.tar.gz
$tar -xzvf PhoBERT_large_transformers.tar.gz
```

## Install VnCoreNLP

```shell
!mkdir -p vncorenlp/models/wordsegmenter
!wget -q --show-progress https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
!wget -q --show-progress https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
!wget -q --show-progress https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
!mv VnCoreNLP-1.1.1.jar vncorenlp/ 
!mv vi-vocab vncorenlp/models/wordsegmenter/
!mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
```

## Scripts

```shell
python train.py --fold <fold-id> \
--train_path ./data/train.csv \
--dict_path ./<path-to-phobert>/dict.txt \
--config_path ./<path-to-phobert>/config.json \
--bpe-codes ./<path-to-phobert>/bpe.codes \
--pretrained_path ./<path-to-phobert>/model.bin \
--checkpoint_path ./models \
--rdrsegmenter_path /<absolute-path-to>/VnCoreNLP-1.1.1.jar 
```

Example

```shell
python train.py --fold 0 \
--train_path ./data/train.csv \
--dict_path ./PhoBERT_base_transformers/dict.txt \
--config_path ./PhoBERT_base_transformers/config.json \
--bpe-codes ./PhoBERT_base_transformers/bpe.codes \
--pretrained_path ./PhoBERT_base_transformers/model.bin \
--checkpoint_path ./models \
--rdrsegmenter_path ./vncorenlp/VnCoreNLP-1.1.1.jar 
```
