#!/bin/bash
TRAIN="train_crnn.sh"
TEST="test_crnn.sh"
GPU=3
PRETRAINED=-1


bash ${TRAIN} 50 ${GPU}
bash ${TEST} 50 ${GPU}
bash ${TRAIN} 100 ${GPU}
bash ${TEST} 100 ${GPU}
#bash ${TRAIN} 25 ${GPU}
#bash ${TEST} 25 ${GPU}
#bash ${TRAIN} 40 ${GPU}
#bash ${TEST} 40 ${GPU}
#bash ${TRAIN} 90 ${GPU}
#bash ${TEST} 90 ${GPU}
#for((i=25; i>=25; i=i-20));
#do
#bash ${TRAIN} Syn${i}k ${GPU} ${PRETRAINED};
#bash ${TEST} Syn${i}k ${GPU} ${PRETRAINED};
#done
