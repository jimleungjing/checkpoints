import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', required=True, help='path to dataset')
parser.add_argument('--valRoot', required=True, help='path to dataset')
parser.add_argument('--resRoot', required=True, help='path to result')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--cuda', type=str, default='-1', help='enables cuda')

parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to test')
parser.add_argument('--ntest', type=int, default=100, help='number of samples to test')
opt = parser.parse_args()
print(opt)

model_path = '{0}/netCRNN_{1}.pth'.format(opt.expr_dir, opt.nepoch)

# load the pretrained CRNN model from the specified path
alphabet = opt.alphabet
nclass = len(opt.alphabet) + 1
nc = 1
netCRNN = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
cudnn.benchmark = True

# Set the gpu mode
if torch.cuda.is_available() and opt.cuda != '-1':
    str_ids = opt.cuda.split(",")
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        netCRNN = netCRNN.to(gpu_ids[0])
        netCRNN = torch.nn.DataParallel(netCRNN, device_ids=gpu_ids)
        image = image.to(gpu_ids[0])
image = Variable(image)


print('loading pretrained netCRNN from %s' % model_path)
# if isinstance(netCRNN, torch.nn.DataParallel):
#     netCRNN = netCRNN.module
netCRNN.load_state_dict(torch.load(model_path, map_location=str(device)))






# Load the test data
transformer = dataset.resizeNormalize((opt.imgW, opt.imgH))
test_dataset = dataset.lmdbDataset(
    root=opt.valRoot, transform=transformer)
test_data_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))

# Set the label converter
converter = utils.strLabelConverter(alphabet)



# ===============================================
# Helper Function
# ===============================================
def EditDistance(strA, strB):
    lenA = len(strA)
    lenB = len(strB)
    dp = np.zeros((lenA + 1, lenB + 1))

    # initialize the array dp
    for i in range(lenA + 1):
        dp[i][0] = i
    for j in range(lenB + 1):
        dp[0][j] = j
    # compute the edit distance
    for i in range(lenA):
        for j in range(lenB):
            if strA[i] == strB[j]:
                dp[i+1][j+1] = dp[i][j]
            else:
                insert = dp[i+1][j] + 1
                delete = dp[i][j+1] + 1
                replace = dp[i][j] + 1
                dp[i+1][j+1] = min(insert,delete)
                dp[i+1][j+1] = min(dp[i+1][j+1], replace)
    return dp[lenA][lenB]

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        # if input_image = [B,C,H,W]
        # image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # if input_image = [C, H, W]
        image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

# ===============================================
# Start to test
# ===============================================
print("Start testing!")
netCRNN.eval()
val_iter = iter(test_data_loader)
max_iter = min(opt.ntest, len(test_data_loader))
# image = Image.open(img_path).convert('L')
# image = transformer(image)
# if torch.cuda.is_available():
#     image = image.to(gpu_ids[0])
# image = image.view(1, *image.size())
# image = Variable(image)

i = 0
n_correct = 0
n_dataset = 0
NED = 0

if not os.path.exists(opt.resRoot):
    subprocess.call("mkdir -p %s"%(opt.resRoot), shell=True)
FAIL_DIR=os.path.join(opt.resRoot,"fails")
if not os.path.exists(FAIL_DIR):
    subprocess.call("mkdir -p %s"%(FAIL_DIR), shell=True)


res = open(os.path.join(opt.resRoot, "pred.txt"),"w+")
succ = open(os.path.join(opt.resRoot, "succ.txt"), "w+")
fail = open(os.path.join(opt.resRoot, "fail.txt"), "w+")

for i in range(max_iter):
    data = val_iter.next()
    i += 1
    images, texts = data
    decode_texts = [text[2:-1] for text in texts]
    texts = decode_texts

    utils.loadData(image, images)
    preds = netCRNN(image)
    if len(gpu_ids) > 0:
        preds_chunks = preds.chunk(len(gpu_ids), dim=0)
        preds = torch.cat(preds_chunks, dim = 1)
    
    preds_size = Variable(torch.IntTensor([preds.size(0)] * images.size(0)))
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    
    idx = 0
    for raw_pred, sim_pred, label in zip(raw_preds, sim_preds, texts):
        # print('%-20s => %-20s' % (raw_pred, sim_pred))
        prompt = ""
        edit = 0
        if sim_pred != label.lower():    
            maxLen = max(len(sim_pred), len(label))
            minLen = min(len(sim_pred), len(label))
            for i in range(maxLen):
                if i < minLen:
                    if sim_pred[i] != label[i]:
                        prompt += "^"
                    else:
                        prompt += " "
                else:
                    prompt += "^"
            edit = EditDistance(sim_pred, label)
            NED += (edit / maxLen)
            fail.write("pred:%s\tgt:%s\n" % (sim_pred, label))
            # save the image
            im_fn = "%s/%s" % (FAIL_DIR, label) 
            while os.path.exists(im_fn + ".png"):
                seed = np.random.randint(0,100)
                im_fn += "_%d"%(seed)
            im = tensor2im(image[idx])
            save_image(im, im_fn + ".png")
        else:
            n_correct += 1
            succ.write("pred:%s\tgt:%s\n" % (sim_pred, label))
        n_dataset += 1
        print("pred:%-20s\ngt  :%-20s\nerr :%-20s\nEdit:%f\n" % (sim_pred, label, prompt, edit))
        res.write("pred:%-20s\t\tgt:%-20s\n" % (sim_pred, label))
        idx += 1
    # accuracy = n_correct / float(max_iter * opt.batchSize)

accuracy = n_correct / float(n_dataset)
print('Total accuray: %f\tNED:%f' % (accuracy, NED))
res.write('\nTotal accuray: %f\tNED:%f\n' % (accuracy, NED))
res.close()
succ.close()
fail.close()
