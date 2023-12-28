# Just for NQ-TRAIN
DATA_FOLDER='nq-train'

WIKI_FOLDER="wikipedia_data"
python3  add_wikicategory.py  --wiki_folder  $WIKI_FOLDER  --dataset  $DATA_FOLDER

# Get BM25 and labeling for nq, hotpotqa, fever and climate-fever
DATA_FOLDERS='nq hotpotqa fever climate-fever'
for DATA_FOLDER in $DATA_FOLDERS;
do
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
    --hits  1000  \
    --batch  100

    python3  serini_run_to_json.py  --data_folder  $DATA_FOLDER

    WIKI_FOLDER="wikipedia_data"
    python3  add_wikicategory.py  --wiki_folder  $WIKI_FOLDER  --dataset  $DATA_FOLDER
done