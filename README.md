# DESIRE-ME: Domain-Enhanced Supervised Information REtrieval using Mixture-of-Experts

Accepted at ECIR 2024

## Abstract

Open-domain question answering requires retrieval systems able to cope with the diverse and varied nature of questions, providing accurate answers across a broad spectrum of query types and topics. To deal with such topic heterogeneity through a unique model, we propose DESIRE-ME, a neural information retrieval model that leverages the Mixture-of-Experts framework to combine multiple specialized neural models. We rely on Wikipedia data to train an effective neural gating mechanism that classifies the incoming query and that weighs correspondingly the predictions of the different domain-specific experts. This allows DESIRE-ME to specialize adaptively in multiple domains. Through extensive experiments on publicly available datasets, we show that our proposal can effectively generalize domain-enhanced neural models. DESIRE-ME excels in handling open-domain questions adaptively, boosting by up to 12% in NDCG@10 and 23% in P@1, the underlying state-of-the-art dense retrieval model.

## Recreate the dataset

### Wikipedia Dataset

To recreate the dataset we rely on [Wikimedia database dump of the English Wikipedia on December 20, 2018](https://archive.org/details/enwiki-20181220). To recreate the wikipedia files that we need just run the `wikipedia_data_creation.sh` file containing the following comands:

```
WIKIPEDIA_FOLDER=wikipedia_data

mkdir $WIKIPEDIA_FOLDER

echo "cat_id,cat_title,cat_pages,cat_subcats,cat_files" > $WIKIPEDIA_FOLDER/category.csv
category_link=https://archive.org/download/enwiki-20181220/enwiki-20181220-category.sql.gz
wget -P $WIKIPEDIA_FOLDER $category_link
zcat $WIKIPEDIA_FOLDER/enwiki-20181220-category.sql.gz | python3 mysqldump_to_csv.py >> $WIKIPEDIA_FOLDER/category.csv

echo "cl_from,cl_to,cl_sortkey,cl_timestamp,cl_sortkey_prefix,cl_collation,cl_type" > $WIKIPEDIA_FOLDER/categorylinks.csv
categorylinks_link=https://archive.org/download/enwiki-20181220/enwiki-20181220-categorylinks.sql.gz
wget -P $WIKIPEDIA_FOLDER $categorylinks_link
zcat $WIKIPEDIA_FOLDER/enwiki-20181220-categorylinks.sql.gz | python3 mysqldump_to_csv.py >> $WIKIPEDIA_FOLDER/categorylinks.csv

echo "page_id,page_namespace,page_title,page_is_redirect,page_is_new,page_random,page_touched,page_links_updated,page_latest,page_len,page_content_model,page_lang" > $WIKIPEDIA_FOLDER/page.csv
page_link=https://archive.org/download/enwiki-20181220/enwiki-20181220-page.sql.gz
wget -P $WIKIPEDIA_FOLDER $page_link
zcat $WIKIPEDIA_FOLDER/enwiki-20181220-page.sql.gz | python3 mysqldump_to_csv.py >> $WIKIPEDIA_FOLDER/page.csv
```
This will create three files: `category.csv`, `categorylinks.csv` and `page.csv` dataframes from the Wikipedia Dump SQL files.
Alternatively, simply run: `1_pipeline_wikipedia_data_creation.sh` 

### BEIR Datasts
For this part, the code is slightly different for NQ as it has two separate datasets for training: nq-train and nq, and for Climate-FEVER which does not have the training and validation.

<details>
<summary>NQ-train</summary>

```
# NQ-TRAIN
DATA_FOLDER='nq-train'

WIKI_FOLDER="wikipedia_data"
python3  add_wikicategory.py  --wiki_folder  $WIKI_FOLDER  --dataset  $DATA_FOLDER
```
</details>

<details>
<summary>NQ, HotpotQA, FEVER and Climate-FEVER</summary>

```
DATA_FOLDER='nq'  # 'hotpotqa', 'fever' or 'climate-fever'

python3  create_pyserini_data.py  --data_folder  $DATA_FOLDER  --dataset  $DATA_FOLDER

python3  -m  pyserini.index.lucene  \
--collection  JsonCollection  \
--input  "$DATA_FOLDER"_serini_jsonl  \
--index  indexes/"$DATA_FOLDER"_serini_index  \
--generator  DefaultLuceneDocumentGenerator  \
--threads  8  \
--fields  title  \
--storePositions  --storeDocvectors  --storeRaw

python3  -m  pyserini.search.lucene  \
--index  indexes/"$DATA_FOLDER"_serini_index  \
--topics  $DATA_FOLDER/queries.tsv  \
--output  $DATA_FOLDER/run.txt  \
--bm25  \
--k1  0.9  \
--b  0.4  \
--fields  contents=1  title=1  \
--hits  100  \
--batch  100

python3  serini_run_to_json.py  --data_folder  $DATA_FOLDER

WIKI_FOLDER="wikipedia_data"
python3  add_wikicategory.py  --wiki_folder  $WIKI_FOLDER  --dataset  $DATA_FOLDER
```
</details>

For the complete pipeline run: `2_pipeline_pyserini.sh`

## Training and Testing 

To train the models so go to the src folder (`cd src`) folder and run the following: 

```
DATASET=nq-train
MODEL=contriever

python3 1_train.py model=$MODEL dataset=$DATASET testing=$DATASET training.max_epoch=60 training.batch_size=512 training.lr=1e-4

python3 2_create_embedding.py model=$MODEL dataset=$DATASET testing=$DATASET training.batch_size=16

python3 3_test.py model=$MODEL dataset=$DATASET testing=$DATASET model.init.specialized_mode='desireme' 
python3 3_test.py model=$MODEL dataset=$DATASET testing=$DATASET model.init.specialized_mode='rand'
python3 4_beir_eval.py model=$MODEL dataset=$DATASET testing=$DATASET # zero shot model

python3 5_significance_test.py dataset=$DATASET testing=$DATASET
```
Just change the DATASET and MODEL to the one you want to replicate, for `climate-fever` do not run the `python3 1_train.py` command as there is not training required. <br>
The possible DATASET are: `nq-train`, `hotpotqa`, `fever` and `climate-fever`.<br>
The possible MODEL are: `cocodr-base-msmarco`, `cocodr-large-msmarco` and `contriever`.<br>
The configurations for training and everthing must be managed fron the `conf` folder, and you can override them using flags.

To replicate the fine-tuning of  the biencoders, use the following commands:
```
DATASET='nq-train' # hotpotqa, fever
MODEL='cocodr-large-msmarco' # cocodr-base-msmarco, cocodr-large-msmarco

python3 6_train_biencoder.py model=$MODEL dataset=$DATASET testing=$DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.lr=1e-6 training.max_epoch=10 training.batch_size=32
python3 7_create_embedding.py model=$MODEL dataset=$DATASET testing=$DATASET training.batch_size=16 dataset.model_dir='output/'$DATASET'/saved_model_biencoder' testing.embedding_dir='output/'$DATASET'/embedding_biencoder'
python3 8_test_biencoder.py model=$MODEL dataset=$DATASET testing=$DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' testing.embedding_dir='output/'$DATASET'/embedding_biencoder'
```
Again the DATASET and the MODEL must be changed according to what you want to replicate.

For the complete pipeline for all models and datasets run: `3_pipeline_models.sh`.


# Citation
```
@InProceedings{kasela2024desireme,
author={Kasela, Pranav and Pasi, Gabriella and Perego, Raffaele and Tonellotto, Nicola},
editor={Goharian, Nazli and Tonellotto, Nicola and He, Yulan and Lipani, Aldo and McDonald, Graham and Macdonald, Craig and Ounis, Iadh},
title="DESIRE-ME: Domain-Enhanced Supervised Information Retrieval Using Mixture-of-Experts",
booktitle="Advances in Information Retrieval",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="111--125",
isbn="978-3-031-56060-6",
doi="10.1007/978-3-031-56060-6_8"
}


```
