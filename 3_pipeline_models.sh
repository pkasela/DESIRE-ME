cd src

DATASETS='nq-train hotpotqa fever'
MODELS='cocodr-base-msmarco contriever cocodr-large-msmarco'

for DATASET in $DATASETS;
do
    for MODEL in $MODELS;
    do
        python3 1_train.py model=$MODEL dataset=$DATASET testing=$DATASET training.max_epoch=60 training.batch_size=512 training.lr=1e-4

        python3 2_create_embedding.py model=$MODEL dataset=$DATASET testing=$DATASET training.batch_size=16

        python3 3_test.py model=$MODEL dataset=$DATASET testing=$DATASET model.init.specialized_mode='zeros' 
        python3 3_test.py model=$MODEL dataset=$DATASET testing=$DATASET model.init.specialized_mode='desireme' 
        python3 3_test.py model=$MODEL dataset=$DATASET testing=$DATASET model.init.specialized_mode='rand'
        
        python3 4_significance_test.py dataset=$DATASET testing=$DATASET

        python3 6_train_biencoder.py model=$MODEL dataset=$DATASET testing=$DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.lr=1e-6 training.max_epoch=10 training.batch_size=32
        python3 7_create_embedding.py model=$MODEL dataset=$DATASET testing=$DATASET training.batch_size=16 dataset.model_dir='output/'$DATASET'/saved_model_biencoder' testing.embedding_dir='output/'$DATASET'/embedding_biencoder'
        python3 8_test_biencoder.py model=$MODEL dataset=$DATASET testing=$DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' testing.embedding_dir='output/'$DATASET'/embedding_biencoder'
    done
done

DATASET='climate-fever'

for MODEL in $MODELS;
do
    python3 2_create_embedding.py model=$MODEL dataset=$DATASET testing=$DATASET training.batch_size=16

    python3 3_test.py model=$MODEL dataset=$DATASET testing=$DATASET model.init.specialized_mode='zeros' 
    python3 3_test.py model=$MODEL dataset=$DATASET testing=$DATASET model.init.specialized_mode='weight' 
    python3 3_test.py model=$MODEL dataset=$DATASET testing=$DATASET model.init.specialized_mode='rand' 

    python3 4_significance_test.py dataset=$DATASET testing=$DATASET
done