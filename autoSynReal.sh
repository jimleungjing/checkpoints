#!/bin/bash
TRAIN="trainSynReal.sh"
TEST="testSynReal.sh"
GPU=1


bash ${TRAIN} A ${GPU}
bash ${TEST} A ${GPU}
bash ${TRAIN} B ${GPU}
bash ${TEST} B ${GPU}
