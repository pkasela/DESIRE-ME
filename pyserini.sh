python create_pyserini_data.py

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input hotpotqa_serini_jsonl \
  --index indexes/hotpotqa_serini_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --fields title \
  --storePositions --storeDocvectors --storeRaw


python -m pyserini.search.lucene \
  --index indexes/hotpotqa_serini_index \
  --topics hotpotqa/queries.tsv \
  --output hotpotqa/run.txt \
  --bm25 \
  --k1 0.9 \
  --b 0.4 \
  --fields contents=1 title=1 \
  --hits 100 \
  --batch 100

python serini_run_to_json.py