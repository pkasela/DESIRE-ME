DATA_FOLDER='hotpotqa'
python create_pyserini_data.py

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "$DATA_FOLDER"_serini_jsonl \
  --index indexes/"$DATA_FOLDER"_serini_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --fields title \
  --storePositions --storeDocvectors --storeRaw


python -m pyserini.search.lucene \
  --index indexes/"$DATA_FOLDER"_serini_index \
  --topics $DATA_FOLDER/queries.tsv \
  --output $DATA_FOLDER/run.txt \
  --bm25 \
  --k1 0.9 \
  --b 0.4 \
  --fields contents=1 title=1 \
  --hits 100 \
  --batch 100

python -m pyserini.search.lucene \
  --index indexes/"$DATA_FOLDER"_serini_index \
  --topics $DATA_FOLDER/dev_queries.tsv \
  --output $DATA_FOLDER/dev_run.txt \
  --bm25 \
  --k1 0.9 \
  --b 0.4 \
  --fields contents=1 title=1 \
  --hits 100 \
  --batch 100


python serini_run_to_json.py