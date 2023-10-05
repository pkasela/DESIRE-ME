
  

# DESIRE-ME: Domain-Enhanced Supervised Information REtrieval using Mixture-of-Experts

  

  

## Abstract

  

Open-domain question answering requires retrieval systems able to cope with the diverse and varied nature of questions, providing accurate answers across a broad spectrum of query types and topics. To deal with such topic heterogeneity through a unique model, we propose DESIRE-ME, a neural information retrieval model that leverages the Mixture-of-Experts framework to combine multiple specialized neural models. We rely on Wikipedia data to train an effective neural gating mechanism that classifies the incoming query and that weighs correspondingly the predictions of the different domain-specific experts. This allows DESIRE-ME to specialize adaptively in multiple domains. Through extensive experiments on publicly available datasets, we show that our proposal can effectively generalize domain-enhanced neural models. DESIRE-ME excels in handling open-domain questions adaptively, boosting by up to 12% in NDCG@10 and 23% in P@1, the underlying state-of-the-art dense retrieval model.

  

  

## Recreate the dataset

  

### Wikipedia Dataset

To recreate the dataset we rely on [Wikimedia database dump of the English Wikipedia on December 20, 2018](https://archive.org/details/enwiki-20181220). To recreate the wikipedia files that we need just run:

```
WIKIPEDIA_FOLDER='wikipedia_data'
mkdir $WIKIPEDIA_FOLDER

category_link='https://archive.org/download/enwiki-20181220/enwiki-20181220-category.sql.gz'
wget -P $WIKIPEDIA_FOLDER $category_link
zcat $WIKIPEDIA_FOLDER/enwiki-20181220-category.sql.gz | python3 mysqldump_to_csv.py >> $WIKIPEDIA_FOLDER/category.csv

categorylinks_link='https://archive.org/download/enwiki-20181220/enwiki-20181220-categorylinks.sql.gz'
wget -P $WIKIPEDIA_FOLDER $categorylinks_link
zcat $WIKIPEDIA_FOLDER/enwiki-20181220-categorylinks.sql.gz | python3 mysqldump_to_csv.py >> $WIKIPEDIA_FOLDER/categorylinks.csv

page_link='https://archive.org/download/enwiki-20181220/enwiki-20181220-page.sql.gz'
wget -P $WIKIPEDIA_FOLDER $page_link
zcat $WIKIPEDIA_FOLDER/enwiki-20181220-page.sql.gz | python3 mysqldump_to_csv.py >> $WIKIPEDIA_FOLDER/page.csv
```

This will create three files: `category.csv`, `categorylinks.csv` and `page.csv` dataframes from the Wikipedia Dump SQL files.

### BEIR Datasts
