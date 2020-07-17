'''
Minimum working example of GCP bucket integrated example with support for Tensorboard
ImageNet training code adapted from 
    https://github.com/pytorch/examples/tree/master/imagenet

Project originally created for 
Originally created for Stanford Spring 2019 CS341
Jingbo Yang, Ruge Zhao, Meixian Zhu
'''

import argparse
import datetime
import getpass
import os
from pathlib import Path
import pprint as pp
import random
import shutil
import sys
import time
import warnings

import numpy as np
from pytz import timezone
import socket
from sklearn.metrics import confusion_matrix as skcm
import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# Fancy Additions
from utils.gc_storage import GCStorage, GCOpen
from utils.tb_logger import Logger
from utils.tb_plot_helper import plot_simple, plot_confusion_matrix, plot_tsne
from utils.model_builder import EmbeddingModel
from utils.evaluator import Evaluator
from utils.argparser_helper import fix_nested_namespaces, get_experiment_number, namespace_to_dict

from utils.private import GC_BUCKET, CREDENTIAL_PATH, PROJECT_ID

# Global Vars
TIME_ZONE = 'US/Pacific'
EXP_ROOT = 'experiments'
SOURCE_ROOT = os.path.dirname(os.path.abspath(__file__))

best_acc1 = 0
total_steps = 0


CREDENTIAL_PATH = 'bash_scripts/example_credential.json'
LOCAL_CACHE = os.environ['HOME'] + '/gc_cache'

# List of evaluation functions and additional parameters
# See utils/evaluator.py for how thse functions are used
EVAL_LIST = [('topk', (1, )), ('topk', (5, )),
             ('confusion_matrix', ()),
             ('tsne', (10, )), ('tsne', (25, ))]

# Prepare infrastructure
cloudFS = GCStorage.get_CloudFS(PROJECT_ID, GC_BUCKET, CREDENTIAL_PATH, LOCAL_CACHE)
logger = None

MODEL_NAME = 'resnet18'

def parse_argument():

    # Parse arguments
    # model_names = sorted(name for name in models.__dict__
    #     if name.islower() and not name.startswith("__")
    #     and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--exp_name', required=True,
                            help='experiment name')
    parser.add_argument('--train_data', required=True,
                            help='path to train dataset')
    parser.add_argument('--val_data', required=True,
                            help='path to val dataset')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', default=False, action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')

    args = parser.parse_args()
    args = fix_nested_namespaces(args)      # Correclty support arguments that allow for definition of
                                            # --model_args.model_name for more organized argument handling

    us_timezone = timezone(TIME_ZONE)
    date = datetime.datetime.now(us_timezone).strftime("%Y-%m-%d")
    date_short = datetime.datetime.now(us_timezone).strftime("%Y%m%d")
    save_dir = Path(EXP_ROOT) / date
    args.exp_name = getpass.getuser() + '_' + socket.gethostname() + '_' + date_short + '_' + args.exp_name
    exp_num = get_experiment_number(cloudFS, save_dir, args.exp_name)
    args.exp_name = args.exp_name + '_' + str(exp_num)
    
    date_folder = date

    # import pdb; pdb.set_trace()

    # Also recover the original arguments just in case
    arg_text = ' '.join(sys.argv)
    args.orig_command_line = arg_text

    return args, namespace_to_dict(args), date_folder


def main():
    global logger

    args, args_dict, date_folder = parse_argument()
    logger = Logger(cloudFS, Path(f'{EXP_ROOT}/{date_folder}/{args.exp_name}'))

    logger.log_sourcecode(SOURCE_ROOT)

    logger.log_text({'setup:command_line': ' '.join(sys.argv),
                     'setup:parsed_arguments': pp.pformat(args, indent=4)},
                    0)
    logger.log_json(args_dict, 'args')
    logger.log_sourcecode(os.path.dirname(os.path.abspath(__file__)))


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1

    args.gpu = gpu

    if args.gpu is not None:
        logger.log("Use GPU: {} for training".format(args.gpu))

    evaluator = Evaluator(EVAL_LIST)

    # Data loading code
    traindir = args.train_data
    valdir = args.val_data
    num_classes = len(os.listdir(args.val_data))
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))
    logger.log('Dataset defined')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    logger.log('Training dataloader defined')

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    logger.log('Validation dataloader defined')

    # create model
    model = models.__dict__[MODEL_NAME](pretrained=args.pretrained)
    model = EmbeddingModel(model, num_classes)

    if not torch.cuda.is_available():
        logger.log('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    logger.log('Model created defined')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    logger.log('Loss function defined')

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    logger.log('Optimizer defined')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.log("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.log("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.log("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        logger.log('Model in evaluation mode')
        validate(val_loader, model, criterion, evaluator, 0, args)
        return
    else:
        logger.log('Model in training mode')

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, evaluator, optimizer, epoch, args)

        # evaluate on validation set
        validate(val_loader, model, criterion, evaluator, epoch, args)

        if not args.multiprocessing_distributed or \
            (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

            state = {
                'epoch': epoch + 1,
                'arch': MODEL_NAME,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }
            logger.log_model_state(state, epoch + 1, total_steps, torch.save)


def train(train_loader, model, criterion, evaluator, optimizer, epoch, args):
    global total_steps

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in \
            tqdm.tqdm(enumerate(train_loader),
            total=len(train_loader.dataset) / args.batch_size):
        total_steps += 1

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output, embedding = model(images)
        loss = criterion(output, target)

        evaluator.store('train', 'output', output)
        evaluator.store('train', 'target', target)
        evaluator.store('train', 'embedding', embedding)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        end = time.time()

        if i % args.print_freq == 0:
            evaluator.store('train_step', 'output', output)
            evaluator.store('train_step', 'target', target)
            evaluator.store('train_step', 'embedding', embedding)

            scalar_result, image_result = evaluator.evaluate('train_step')
            evaluator.report(epoch, i, total_steps,
                                scalar_result, image_result)
            evaluator.clear('train_step')

    scalar_result, image_result = evaluator.evaluate('train')
    evaluator.report(epoch, i, total_steps,
                        scalar_result, image_result)
    evaluator.clear('train')


def validate(val_loader, model, criterion, evaluator, epoch, args):
    global total_steps

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in \
                tqdm.tqdm(enumerate(val_loader),
                total=len(val_loader.dataset) / args.batch_size):
            total_steps += 1

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output, embedding = model(images)
            loss = criterion(output, target)
            
            evaluator.store('validation', 'output', output)
            evaluator.store('validation', 'target', target)
            evaluator.store('validation', 'embedding', embedding)

            # measure elapsed time
            end = time.time()

            # if i % args.print_freq == 0:
            #    progress.display(i)

    scalar_result, image_result = evaluator.evaluate('validation')
    evaluator.report(epoch, i, total_steps,
                        scalar_result, image_result)
    evaluator.clear('validation')


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
