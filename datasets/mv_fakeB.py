import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--outputTrainPath', type=str, default='./A2B_V_train10k', help='path to dataset')
parser.add_argument('--fakeBPath', type=str, default='./A_V_train10k', help='path to fake B')
parser.add_argument('--epoch', type=str, default='160', help='test epoch')

opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.outputTrainPath):
    os.mkdir(opt.outputTrainPath)

A_fns = '''find %s -name "*_fake_B.png" | xargs -i cp {} %s'''%(opt.fakeBPath + "/test_" + opt.epoch + "/images", opt.outputTrainPath)       # get filenames of A
subprocess.call(A_fns, shell=True)
# testA_code = subprocess.Popen(testA_fns, shell=True, stdout=subprocess.PIPE)
# tests = [bytes.decode(fn).replace('\n','') for fn in testA_code.stdout.readlines()]
