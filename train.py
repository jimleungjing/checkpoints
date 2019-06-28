from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import time

import models.crnn as crnn

parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', required=True, help='path to dataset')
parser.add_argument('--valRoot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=-1, help='number of epochs to train for')
# TODO(meijieru): epoch -> iter
parser.add_argument('--cuda', type=str, default='-1', help='enables cuda')
parser.add_argument('--pretrained', type=int, default=-1, help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=30, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--val_freq', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--save_freq', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and opt.cuda == '-1':
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(
    root=opt.valRoot, transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))

nclass = len(opt.alphabet) + 1
nc = 1
#nc = 3

converter = utils.strLabelConverter(opt.alphabet)
criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda != '-1':
    str_ids = opt.cuda.split(",")
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        crnn.to(gpu_ids[0])
        crnn = torch.nn.DataParallel(crnn, device_ids=gpu_ids)
        image = image.to(gpu_ids[0])
        criterion = criterion.to(gpu_ids[0])
if opt.pretrained > -1:
    model_path = '{0}/netCRNN_{1}.pth'.format(opt.expr_dir, opt.pretrained)
    print('loading pretrained model from %s' % model_path)
    # crnn.load_state_dict(torch.load(opt.pretrained))
    crnn.load_state_dict(torch.load(model_path))
print(crnn)



image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr, weight_decay=0.0005)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


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


def val(net, dataset, criterion, max_iter=20):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    sum_NED = 0
    n_dataset = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        decode_texts = [text[2:-1] for text in cpu_texts]
        t, l = converter.encode(decode_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_chunks = preds.chunk(len(gpu_ids), dim=0)
        preds = torch.cat(preds_chunks, dim = 1)            # [num_gpu * time_step, batch_size / num_gpu, nOut] -> [time_step, batch_size, nOut]

        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        # [max_values,...,], [max_index,...,] = tensor.max(dim=k)
        # preds = [time_step, batch_size, nOut]
        # _, preds = preds.max(2)
        # preds = [time_step, batch_size]
        _, preds = preds.max(2)         
        # preds = preds.squeeze(2)

        # after transpose, preds = [batch_size, time_step]
        preds = preds.transpose(1, 0).contiguous().view(-1)
        # preds = [batch_size * time_step]
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        # len(sim_preds) = batch_size, len(data_loader) = sizeof(datasets) / batch_size
        for pred, target in zip(sim_preds, cpu_texts):
            n_dataset += 1
            gt = target[2:-1].lower()
            if pred == gt:
                n_correct += 1
            else:
                # sum of normalized edit distance
                sum_NED += (EditDistance(pred, gt) / len(gt))

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        # print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
        print('%-20s => %-20s' % (raw_pred, pred))
        gt = gt[2:-1].lower()
        prompt = ""
        ned = 0
        if pred != gt:    
            maxLen = max(len(pred), len(gt))
            minLen = min(len(pred), len(gt))
            for i in range(maxLen):
                if i < minLen:
                    if pred[i] != gt[i]:
                        prompt += "^"
                    else:
                        prompt += " "
                else:
                    prompt += "^"
            ned = EditDistance(pred, gt)
        print("pred:%-20s\ngt  :%-20s\nerr :%-20s" % (pred, gt, prompt))
        print("Edit Distance:\t%f" % (ned))

    # accuracy = n_correct / float(max_iter * opt.batchSize)
    accuracy = n_correct / float(n_dataset)
    print('Test loss: %f, accuray: %f, NED: %f' % (loss_avg.val(), accuracy, sum_NED))
    return accuracy, sum_NED


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    decode_texts = [text[2:-1] for text in cpu_texts]
    t, l = converter.encode(decode_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    
    # it will merge on the dimension 0 when running in multiple GPUs mode, 
    # say we use 4 GPUs,
    # image_size = [batch_size, channels, height, width] = [16, 1, 48, 600]
    # preds(in CRNN) = [Seq_len, batch_size, nOut] = [151, 4, 37]
    # preds(in trainBatch) = [Seq_len * num_gpu, batch_size, nOut] = [604, 4, 37]
    # that is, since we specify the DataParallel on the dimension 0, it will merge each batch to the dimension 0, which will result in an error in the following steps
    preds = crnn(image)
    preds_chunks = preds.chunk(len(gpu_ids), dim=0)
    preds = torch.cat(preds_chunks, dim = 1)            # [num_gpu * time_step, batch_size / num_gpu, nOut] -> [time_step, batch_size, nOut]

    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    # output = [batch_size, time_step, nOut=nclass] if set batch_first true
    # preds_size = Variable(torch.IntTensor([preds.size(1)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

if opt.pretrained > -1:
    epoch_base = opt.pretrained
else:
    epoch_base = 0

# log the loss and accuracy = correct / (max_iter * batch_size)
log = open(os.path.join(opt.expr_dir, "log.txt"), "a+")

for epoch in range(epoch_base + 1, epoch_base + opt.nepoch + 1):
    train_iter = iter(train_loader)
    i = 0
    epoch_start_time = time.time()
    epoch_iters = len(train_loader)
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        i += 1
        if i % 10 == 0:
            print("Iter [%d / %d] Loss: %f" % (i, epoch_iters, cost))
        if i % opt.displayInterval == 0:
            print('Epoch [%d/%d]\tIter [%d/%d]\tAverage Loss: %f' %
                  (epoch, epoch_base + opt.nepoch, i, len(train_loader), loss_avg.val()))
            log.write('Epoch [%d/%d]\tIter [%d/%d]\tAverage Loss: %f\n' %
                  (epoch, epoch_base + opt.nepoch, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()
    # validation
    if epoch % opt.val_freq == 0:
        accuracy, NED = val(crnn, test_dataset, criterion)
        log.write("Epoch:%d\tAccuracy:%f\tNED:%f\n" % (epoch, accuracy, NED))
    # do checkpointing
    if epoch % opt.save_freq == 0:
            torch.save(
                crnn.state_dict(), '{0}/netCRNN_{1}.pth'.format(opt.expr_dir, epoch))
    print('End of epoch %d / %d \t Time Taken: %d sec\n' % (epoch, epoch_base + opt.nepoch, time.time() - epoch_start_time))
