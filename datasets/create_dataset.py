import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import argparse
import subprocess
import random

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if isinstance(k,str):
                k = k.encode()
            if isinstance(v, str):
                v = v.encode()
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputPath', type=str, default='./AB_train1k_test10k_lmdb_1', help='path to dataset')
    parser.add_argument('--imagePath', type=str, default='./B_test10k', help='path to images')
    parser.add_argument('--percentage', type=int, default=100, help='path to images')
    parser.add_argument('--nSize', type=int, default=0, help='size of images')
    # parser.add_argument('--labelPath', type=str, default='.', help='path to labels')
    
    opt = parser.parse_args()
    print(opt)

    #opt.outputPath = opt.outputPath + "_%d/"%opt.percentage + "testB"
    opt.outputPath = opt.outputPath + "/testB" 
    if not os.path.exists(opt.outputPath):
        subprocess.call("mkdir -p %s"%(opt.outputPath), shell=True)

    imagePathList = []
    labelPathList = []

    IMG_DIR = opt.imagePath
    unique = 0
    for root, dirs, files in os.walk(IMG_DIR, topdown=False):
        if opt.nSize > 0:
            nDatasets = opt.nSize
        else:
            nDatasets = int(opt.percentage / 100 * len(files))
        
        seeds = random.sample(range(0,len(files)), nDatasets)
        for seed in seeds:
            name = files[seed]
            label = os.path.splitext(name)[0].split("_")
            if len(label) == 1:
                unique += 1
            labelPathList.append(label[0])
            imagePathList.append(os.path.join(root, name))
        break
    # test list
    # for i in range(len(imagePathList)):
    #     print(labelPathList[i], "\t", imagePathList[i])
    print(unique)
    print(len(labelPathList))

    createDataset(opt.outputPath, imagePathList, labelPathList)


    
