#!/bin/bash
TRAIN="train_crnn.sh"
TEST="test_crnn.sh"
GPU=4
PRETRAINED=-1

#bash ${TRAIN} Syn${i}k ${GPU} ${PRETRAINED};
#bash ${TEST} Syn25k ${GPU}
bash ${TEST} 10 ${GPU}
bash ${TEST} 50 ${GPU}

