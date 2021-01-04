SIZE_DEP=75
SIZE_ASSOC=25
TRAIN_FN=parus_first_10m_lines.conll
COL_CTX_D=3
USE_DEPREL=1
MODEL_FN=vectors.bin
VOC_M=main.vocab
VOC_P=proper.vocab
VOC_T=tokens.vocab
VOC_D=ctx_dep.vocab
THREADS=8

echo "MAKING BINARIES"
make
cp ./data/stopwords.assoc ./
cp ./data/mwe.list ./

echo ""
echo "DOWNLOADING TRAINING CONLL DATA"
if [ ! -e parus_first_10m_lines.conll.zip ]; then
  wget https://media.githubusercontent.com/media/parus-proj/conll2vec/master/data/parus_first_10m_lines.conll.zip
fi

echo ""
echo "TRAINSET EXTRACTION AND FITTING"
gzip --decompress --stdout ./parus_first_10m_lines.conll.zip | ./mwe2vec -task fit -fit_input stdin -train $TRAIN_FN

echo ""
echo "BUILDING VOCABULARIES"
./mwe2vec -task vocab -train $TRAIN_FN \
          -vocab_m $VOC_M -vocab_p $VOC_P -vocab_t $VOC_T -vocab_d $VOC_D -col_ctx_d $COL_CTX_D -use_deprel $USE_DEPREL \
          -min-count_m 70 -min-count_p 100 -min-count_t 50 -min-count_d 20

echo ""
echo "TRAINING EMBEDDINGS -- MAIN"
./mwe2vec -task train -train $TRAIN_FN \
          -vocab_m $VOC_M -backup backup.data -vocab_d $VOC_D -vocab_a $VOC_M -col_ctx_d $COL_CTX_D -use_deprel $USE_DEPREL -model $MODEL_FN \
          -sample_w 1e-4 -sample_d 1e-4 -sample_a 1e-4 \
          -size_d $SIZE_DEP -size_a $SIZE_ASSOC -negative 4 -iter 10 -threads $THREADS

echo ""
echo "TRAINING EMBEDDINGS -- PROPER"
./mwe2vec -task train -train $TRAIN_FN \
          -vocab_p $VOC_P -restore backup.data -vocab_d $VOC_D -vocab_a $VOC_M -col_ctx_d $COL_CTX_D -use_deprel $USE_DEPREL -model $MODEL_FN \
          -sample_w 1e-2 -sample_d 1e-2 -sample_a 1e-4 \
          -size_d $SIZE_DEP -size_a $SIZE_ASSOC -negative 4 -iter 10 -threads $THREADS

echo ""
echo "RUN SIMILARITY METER"
./mwe2vec -task sim -model $MODEL_FN -model_fmt bin -size_d $SIZE_DEP -size_a $SIZE_ASSOC

