TRAINROOT=AB_train5k_test10k_lmdb_${1}
#TRAINROOT=A_train10k_test10k_lmdb
SUFFIX=""
CHECK_DIR=/data/jing_liang/CRNN
RES_DIR=result/${TRAINROOT}${SUFFIX}
IMG_H=48
IMG_W=600
EPOCH=100
N_TEST=10000
BATCH=16
GPU=${2}

python test.py \
--trainRoot datasets/${TRAINROOT}/trainB \
--valRoot  datasets/${TRAINROOT}/testB \
--resRoot ${RES_DIR} \
--imgH ${IMG_H} \
--imgW ${IMG_W} \
--nepoch ${EPOCH} \
--expr_dir ${CHECK_DIR}/${TRAINROOT}${SUFFIX} \
--batchSize ${BATCH} \
--ntest ${N_TEST} \
--cuda ${GPU}
