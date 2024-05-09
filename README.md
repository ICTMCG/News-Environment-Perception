# News Environment Perception

This is the official repository of the paper:

> **Zoom Out and Observe: News Environment Perception for Fake News Detection**
>
> Qiang Sheng, Juan Cao, Xueyao Zhang, Rundong Li, Danding Wang, and Yongchun Zhu
>
> *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022)*
>
> [PDF](https://aclanthology.org/2022.acl-long.311.pdf) / [Poster](https://sheng-qiang.github.io/data/NEP-Poster.pdf) / [Code](https://github.com/ICTMCG/News-Environment-Perception) / [Chinese Video](https://www.bilibili.com/video/BV1MS4y1e7PY) / [Chinese Blog](https://mp.weixin.qq.com/s/aTFeuCYIpSoazeRi52jqew) / [English Blog](https://montrealethics.ai/zoom-out-and-observe-news-environment-perception-for-fake-news-detection/)

# Datasets

The experimental datasets where can be seen in `dataset` folder, including the [Chinese Dataset](https://github.com/ICTMCG/News-Environment-Perception/tree/main/dataset/Chinese), and the [English Dataset](https://github.com/ICTMCG/News-Environment-Perception/tree/main/dataset/English). Note that you can download the datasets only after an ["Application to Use the Datasets for News Environment Perceived Fake News Detection"](https://forms.office.com/r/QrkcNKuMLB) has been submitted.

# Code

## Key Requirements

```
python==3.6.10
torch==1.6.0
transformers==4.0.0
```

## Preparation

### Step 1: Obtain the representations of posts and news environment items

#### Step1.1: Prepare the SimCSE model

Due to the space limit of the GitHub, we upload the SimCSE's training data by [Google Drive](https://drive.google.com/drive/folders/1J8p6ORqOhlpjl2lWAWq43pgUdG1O0L9T?usp=sharing). You need to download the dataset file (i.e., `[dataset]_train.txt`), and move it into the `preprocess/SimCSE/train_SimCSE/data` of this repo. Then,

```
cd preprocess/SimCSE/train_SimCSE

# Configure the dataset.
sh train.sh
```

Of course, you can also prepare the SimCSE model by your custom dataset. 

#### Step1.2: Obtain the texts' representations

```
cd preprocess/SimCSE

# Configure the dataset.
sh run.sh
```

### Step 2: Construct the macro & micro environment

Get the macro environment and rank its internal items by similarites:

```
cd preprocess/NewsEnv

# Configure the specific T days of the macro environment.
sh run.sh
```

### Step 3: Prepare for the specific detectors

This step is for the preparation of the specific detectors. There are six base models in our paper, and the preparation dependencies of them are as follows: 

<table>
   <tr>
       <td colspan="2"><b>Model</b></td>
       <td><b>Input (Tokenization)</b></td>
       <td><b>Special Preparation</b></td>
   </tr>
   <tr>
       <td rowspan="4"><b>Post-Only</b></td>
       <td>Bi-LSTM</td>
      <td>Word Embeddings</td>
      <td>-</td>
   </tr>
   <tr>
      <td>EANN</td>
      <td>Word Embeddings</td>
      <td>Event Adversarial Training</td>
   </tr>
   <tr>
      <td>BERT</td>
      <td>BERT's Tokens</td>
      <td>-</td>
   </tr>
   <tr>
      <td>BERT-Emo</td>
      <td>BERT's Tokens</td>
      <td>Emotion Features</td>
   </tr>
   <tr>
       <td rowspan="2"><b>"Zoom-In"</b></td>
      <td>DeClarE</td>
      <td>Word Embeddings</td>
      <td rowspan="2">Fact-checking Articles</td>
   </tr>
   <tr>
      <td>MAC</td>
      <td>Word Embeddings</td>
   </tr>
</table>

In the table above, there are five preprocess in total: (1) Tokenization by Word Embeddings, (2) Tokenization by BERT, (3) Event Adversarial Training, (4) Emotion Features, and (5) Fact-checking Articles. We will describe the five respectively.

#### Tokenization by Word Embeddings

This tokenization is dependent on the external pretrained word embeddings. In our paper, we use the [sgns.weibo.bigram-char](<https://github.com/Embedding/Chinese-Word-Vectors>) ([Downloading URL](https://pan.baidu.com/s/1FHl_bQkYucvVk-j2KG4dxA)) for Chinese and [glove.840B.300d](https://github.com/stanfordnlp/GloVe) ([Downloading URL](https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip)) for English.

```
cd preprocess/WordEmbeddings

# Configure the dataset and your local word-embeddings filepath. 
sh run.sh
```

#### Tokenization by BERT

```
cd preprocess/BERT

# Configure the dataset and the pretrained model
sh run.sh
```

#### Event Adversarial Training

```
cd preprocess/EANN

# Configure the dataset and the event number
sh run.sh
```

#### Emotion Features

```
cd preprocess/Emotion/code/preprocess

# Configure the dataset
sh run.sh
```

#### Fact-checking Articles

There are two preparation for fact-checking articles:

1. **Retrieve the most relevant articles for every post**. Specifically, we have retrieved every post's Top10 relevant articles that should be published BEFORE the post, whose results are saved in the `preprocess/BM25/data` folder. If you want to learn about more implementation details, just refer to `preprocess/BM25/[dataset].ipynb`.
2. **Tokenize the fact-checking articles by word embeddings**:

```
cd preprocess/WordEmbeddings

# Configure the dataset and your local word-embeddings filepath. Set the data_type as 'article'.
sh run.sh
```

## Training and Inferring

```
cd model

# Configure the dataset and the parameters of the model
sh run.sh
```

After that, the results and classification reports will be saved in `ckpts/[dataset]/[model]`.

# Citation

If you find our dataset and code are helpful, please cite the following ACL 2022 paper:

```
@inproceedings{NEP,
    title = "Zoom Out and Observe: News Environment Perception for Fake News Detection",
    author = "Sheng, Qiang  and
      Cao, Juan  and
      Zhang, Xueyao  and
      Li, Rundong and
      Wang, Danding  and
      Zhu, Yongchun",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
    month = may,
    year = "2022",
    publisher = "Association for Computational Linguistics"
}
```

And as the HuffPost part of the English news environment is based on the News Category Dataset, please cite the following reports as the kaggle page requires:

```
@dataset{misra2018news,
  title={News Category Dataset},
  author={Misra, Rishabh},
  year = {2018},
  month = {06},
  doi = {10.13140/RG.2.2.20331.18729}
}
@book{misra2021sculpting,
  author = {Misra, Rishabh and Grover, Jigyasa},
  year = {2021},
  month = {01},
  pages = {},
  title = {Sculpting Data for ML: The first act of Machine Learning},
  isbn = {9798585463570}
}
```
