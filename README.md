# Variational Deep Semantic Hashing (SIGIR'2017)
The PyTorch implementation of the models and experiments of [Variational Deep Semantic Hashing](http://students.engr.scu.edu/~schaidar/paper/Variational_Deep_Hashing_for_Text_Documents.pdf) (SIGIR 2017).

Author: Suthee Chaidaroon and Yi Fang

# Platform
This project uses python 3.6 and pytorch 0.4

# Available dataset
- reuters, rcv1, ng20, tmc, dbpedia, agnews, yahooanswer

# Prepare dataset
We provide a script to generate the datasets in the preprocess folder. You need to download the raw datasets for TMC. 

# Training and Evaluating the model
To train the unsupervised learning model, run the following command:
```
python train_VDSH.py -d [dataset name] -g [gpu number] -b [number of bits]
```

To train the supervised learning model, run the following command:
```
python train_VDSH_S.py -d [dataset name] -g [gpu number] -b [number of bits]
```
OR
```
python train_VDSH_SP.py -d [dataset name] -g [gpu number] -b [number of bits]
```

# Bibtex
```
@inproceedings{Chaidaroon:2017:VDS:3077136.3080816,
 author = {Chaidaroon, Suthee and Fang, Yi},
 title = {Variational Deep Semantic Hashing for Text Documents},
 booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
 series = {SIGIR '17},
 year = {2017},
 isbn = {978-1-4503-5022-8},
 location = {Shinjuku, Tokyo, Japan},
 pages = {75--84},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3077136.3080816},
 doi = {10.1145/3077136.3080816},
 acmid = {3080816},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {deep learning, semantic hashing, variational autoencoder},
}
```

# 10-405 Steps
## Training
- Download VDSH folder
- Create directory datasets/darknet
- Place compiled csv in datasets/darknet, name it data.csv
- Create directory dataset/darknet
- Run python3 preprocess/create_single_label_dataset.py -d darknet
- Run python3 preprocess/convert_tf_to_tfidf.py -d darknet
- Run python3 train_VDSH.py -d darknet.tfidf

## Visualizing
- If cuda available, install t-sne cuda (OPTIONAL)
 -- https://github.com/CannyLab/tsne-cuda/wiki/Installation
- run python3 tsne_and_visualize.py -m <model_name>
- note: <model_name> should be in directory trained_models/

