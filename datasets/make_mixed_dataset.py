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


def make_lmdb_datasets(imagePath, outputPath):
    imagePathList = []
    labelPathList = []

    IMG_DIR = imagePath
    unique = 0
    for root, dirs, files in os.walk(IMG_DIR, topdown=False):
        for name in files:
            label = os.path.splitext(name)[0].split("_")
            if len(label) == 1:
                unique += 1
            labelPathList.append(label[0])
            imagePathList.append(os.path.join(root, name))
    # test list
    # for i in range(len(imagePathList)):
    #     print(labelPathList[i], "\t", imagePathList[i])
    print(unique)
    print(len(labelPathList))
    createDataset(outputPath, imagePathList, labelPathList)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputPath', type=str, default='./B_train5k_test10k_lmdb', help='path to dataset')
    parser.add_argument('--fakeBPath', type=str, default='./A2B_V_train10k', help='path to fake B')
    parser.add_argument('--realBPath', type=str, default='./B_train5k', help='path to real B')
    parser.add_argument('--testBPath', type=str, default='./B_test10k', help='path to test B')
    parser.add_argument('--nSize', type=int, default=10000, help="size of training set")
    parser.add_argument('--mixedWeight', type=float, default=0.5, help="weights for mixed")
    # parser.add_argument('--labelPath', type=str, default='.', help='path to labels')
    
    opt = parser.parse_args()
    print(opt)

    # opt.outputPath += "_%d" % (opt.mixedWeight * 100)
    outputTrainPath = os.path.join(opt.outputPath, "trainB")
    outputTestPath = os.path.join(opt.outputPath, "testB")

    if not os.path.exists(outputTrainPath):
        subprocess.call("mkdir -p %s"%(outputTrainPath), shell=True)

    if not os.path.exists(outputTestPath):
        subprocess.call("mkdir -p %s"%(outputTestPath), shell=True)

    # create mixed training set
    imagePathList = []
    labelPathList = []

    unique = 0
    set_cnt = 0
    # add the fake B
    # for root, dirs, files in os.walk(opt.fakeBPath, topdown=False):
    #     for name in files:
    #         label = os.path.splitext(name)[0].split("_")
    #         if len(label) != 2:
    #             unique += 1
    #         labelPathList.append(label[0])
    #         imagePathList.append(os.path.join(root, name))
    #         set_cnt += 1
    #         if set_cnt >= opt.mixedWeight * opt.nSize:
    #             break
    
    # sample w_mixed * nSize data from the synthesis dataset
    '''
    sizeFake = int(opt.mixedWeight * opt.nSize)
    for root, dirs, files in os.walk(opt.fakeBPath, topdown=False):
        seeds = random.sample(range(0,len(files)), sizeFake)
        for seed in seeds:
            name = files[seed]
            label = os.path.splitext(name)[0].split("_")
            if len(label) != 2:
                unique += 1
            labelPathList.append(label[0])
            imagePathList.append(os.path.join(root,name))
        break

    print("Unique images of fake B:\t",unique)
    print("size of fake B:\t",len(labelPathList))       
    # add the real B
    sizeReal = int((1-opt.mixedWeight) * opt.nSize)
    
    ''' 
    for root, dirs, files in os.walk(opt.realBPath, topdown=False):
        seeds = random.sample(range(0,len(files)), len(files))
        for seed in seeds:
            name = files[seed]
            label = os.path.splitext(name)[0].split("_")
            if len(label) != 2:
                unique += 1
            labelPathList.append(label[0])
            imagePathList.append(os.path.join(root, name))
        break
    
    # test list
    # for i in range(len(imagePathList)):
    #     print(labelPathList[i], "\t", imagePathList[i])
    print("Unique images:\t",unique)
    print("size of training set:\t",len(labelPathList))
    createDataset(outputTrainPath, imagePathList, labelPathList)

    # ================================
    # create the testing set
    # ================================
    imagePathList = []
    labelPathList = []

    unique = 0
    # add the test B
    for root, dirs, files in os.walk(opt.testBPath, topdown=False):
        for name in files:
            label = os.path.splitext(name)[0].split("_")
            if len(label) == 1:
                unique += 1
            labelPathList.append(label[0])
            imagePathList.append(os.path.join(root, name))
    print("Unique images:\t",unique)
    print("size of testing set:\t",len(labelPathList))
    createDataset(outputTestPath, imagePathList, labelPathList)
