'''

## Change directory paths on line 53 and 55

### Experiments for MNIST MLP:

python main.py --dataset=mnist --arch=mlp  --save_dir=- --epochs=30 --wdecay=0.0005  --sch=sch1 --init=proposed --lr=1. --normalization=wn --L=2


### Experiments for CIFAR-10 Resnet

python main.py --arch=resnet --resblocks=9 --epochs=182 --wdecay=0.0002 --sch=he_sch --init=he --lr=1 --save_dir=- --normalization=wn


### Important arguments:
--init: option- proposed, he
--lr: float factor that multiplies the learning rate specified in the learning rate schedule
--normalization: options- wn (for traditional WN), swn (for proposed WN)
--arch: architecture option- resnet, mlp
--resblocks: integer specofying the number of resblocks to be used if using resnet (use 9 for resnet-56 and 83 for resnet-500)
--L: integer specifying the depth of MLP if --arch is mlp
--wdecay: float specifying the value of weight decay coefficient
--epochs: integer specifying the number of epochs
'''


import argparse
import time
import math
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle as pkl

from models import ResNet56, ResNet, vgg11, MLPNet,WResNet_model,ResNet_model

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from utils import progress_bar, get_channels_norm
from torch.optim import SGD
import collections
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
import glob
import tqdm

parser = argparse.ArgumentParser(description='CNN experiments')

# Directories
parser.add_argument('--data', type=str, default='path/to/data/',
                    help='location of the data corpus')
parser.add_argument('--root_dir', type=str, default='path/where/results/are/stored/',
                    help='root dir path to save the log and the final model')
parser.add_argument('--save_dir', type=str, default='default/',
                    help='dir path (inside root_dir) to save the log and the final model')

# Hyperparams
parser.add_argument('--normalization', type=str, default='bn',
                    help='type of normalization (swn, wn, bn)')
parser.add_argument('--lr', type=float, default=1.,
                    help='learning rate factor that gets multiplied to the hard coded LR schedule in the code')
parser.add_argument('--m', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--epochs', type=int, default=182,
                    help='upper epoch limit')
parser.add_argument('--init', type=str, default="proposed")
parser.add_argument('--wdecay', type=float, default=0.,
                    help='weight decay applied to all weights')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset name (cifar10)')
parser.add_argument('--sch', type=str, default='sch1',
                    help='LR schedule')
parser.add_argument('--arch', type=str, default='resnet',
                    help='arch name (resnet, vgg11)')
parser.add_argument('--resblocks', type=int, default=9,
                    help='number of resblocks if using resnet architecture')
parser.add_argument('--L', type=int, default=2,
                    help='num of layers in MLP')



parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--dropout', type=float, default=0.,
                    help='dropout')
parser.add_argument('--opt', type=str, default="sgd",
                    help='optimizer(sgd)')
parser.add_argument('--bs', type=int, default=100, metavar='N',
                    help='batch size')
parser.add_argument('--mbs', type=int, default=100, metavar='N',
                    help='minibatch size')
parser.add_argument('--noaffine', type=bool, default=False,
                    help='not use affine of BN T/F')
parser.add_argument('--datasize', type=int, default=45000,
                    help='dataset size')
parser.add_argument('--nesterov', type=bool, default=False,
                    help='Use nesterov momentum T/F')
parser.add_argument('--resume', type=bool, default=False,
                    help='resume experiment ')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--cluster', action='store_true', help='do not show the progress bar for batch job')


args = parser.parse_args()




if args.arch == 'resnet' or args.arch == 'wresnet':
    if args.dataset=='cifar10':
        if args.sch=='he_sch':
            lr_sch = [[91, 0.1], [136, 0.01], [182, 0.001], [1000000000000, 0.001]]
        elif args.sch=='const':
            lr_sch = [[1000000000000, 0.1]]

    elif args.dataset=='cifar100':
        if args.sch=='sch1':
            lr_sch = [[60, 0.1], [120, 0.01], [160, 0.001], [1000000000000, 0.001]]
        elif args.sch=='const':
            lr_sch = [[1000000000000, 0.1]]
    mom_sch = [[99999999, 1.]]
elif args.arch == 'vgg11':
    if args.sch=='sch1':
        lr_sch = [[25, 0.1], [50, 0.05], [75, 0.025],\
    [100, 0.012], [125, 0.005], [150, 0.0025], [175, 0.0012], [1000000000000, 0.0012/2.]]
    elif args.sch=='const':
            lr_sch = [[1000000000000, 0.1]]

    mom_sch = [[99999999, 1.]]
elif args.arch == 'mlp':
    if args.sch=='sch1':
        lr_sch = [[10, 0.1], [20, 0.05], [30, 0.025], [1000000000000, 0.005]]
        mom_sch = [[99999999, 1.]]


if args.arch=='resnet':
    arch = 'resnet' + str(6*args.resblocks+2)
else:
    arch = args.arch
if args.save_dir[0]=='-':
    tail = args.save_dir[1:]
    args.save_dir = args.dataset + '_' + args.opt + '_bs' +str(args.bs) + '_mbs' + str(args.mbs) + '_m' + str(args.m)+ '_lr{:.3f}'.format(args.lr*lr_sch[0][1]) + '_wd' + \
            str(args.wdecay) + '_' + arch + '_' + 'ep' + str(args.epochs) + '_normalization-' + args.normalization \
            + '_L{}'.format(int(args.L)) + '_init-' + args.init 
    args.save_dir = args.save_dir + tail
log_dir = os.path.join(args.root_dir, args.save_dir) + '/'


# Set the random seed manually for reproducibility.
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run without --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################

print('==> Preparing data..')
if args.dataset=='cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    ind = np.random.permutation(range(50000))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(ind[:args.datasize])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(ind[args.datasize:])
    trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.mbs, #shuffle=True, num_workers=2)
                                              sampler=train_sampler, num_workers=2)

    trainset_hes = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)
    train_sampler_hes = torch.utils.data.sampler.SubsetRandomSampler(ind[:25000])
    trainloader_hessian = torch.utils.data.DataLoader(trainset_hes, batch_size=500, shuffle=False,#shuffle=True, num_workers=2)
                                              sampler=train_sampler_hes, num_workers=2)


    validloader = torch.utils.data.DataLoader(trainset, batch_size=args.mbs, #shuffle=True, num_workers=2)
                                              sampler=valid_sampler, num_workers=2)


    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.mbs, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    nb_classes = len(classes)
elif args.dataset=='cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    ind = np.random.permutation(range(50000))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(ind[:args.datasize])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(ind[args.datasize:])
    trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.mbs, #shuffle=True, num_workers=2)
                                              sampler=train_sampler, num_workers=2)

    trainset_hes = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=transform_train)
    train_sampler_hes = torch.utils.data.sampler.SubsetRandomSampler(ind[:25000])
    trainloader_hessian = torch.utils.data.DataLoader(trainset_hes, batch_size=500, shuffle=False,#shuffle=True, num_workers=2)
                                              sampler=train_sampler_hes, num_workers=2)


    validloader = torch.utils.data.DataLoader(trainset, batch_size=args.mbs, #shuffle=True, num_workers=2)
                                              sampler=valid_sampler, num_workers=2)


    testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.mbs, shuffle=False, num_workers=2)

    nb_classes = 100
elif args.dataset=='mnist':

        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        train_set = torchvision.datasets.MNIST(root=args.data, train=True, transform=trans, download=True)

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(args.datasize))
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(45000, 50000))

        test_set = torchvision.datasets.MNIST(root=args.data, train=False, transform=trans)

        trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.mbs, #shuffle=True, num_workers=2)
                                                  sampler=train_sampler, num_workers=2)

        train_set_hes = torchvision.datasets.MNIST(root=args.data, train=False, transform=trans, download=True)
        train_sampler_hes = torch.utils.data.sampler.SubsetRandomSampler(range(45000))
        trainloader_hessian = torch.utils.data.DataLoader(train_set_hes, batch_size=500,  #shuffle=True, num_workers=2)
                                              sampler=train_sampler_hes, num_workers=2)


        validloader = torch.utils.data.DataLoader(train_set, batch_size=args.mbs, #shuffle=True, num_workers=2)
                                                  sampler=valid_sampler, num_workers=2)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=args.mbs, shuffle=False, num_workers=2)

        nb_classes = 10
###############################################################################
# Build the model
###############################################################################
args.save_dir = os.path.join(args.root_dir, args.save_dir)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

writer = SummaryWriter(log_dir=log_dir)


print('==> Building model..')
start_epoch=1
if args.arch == 'resnet':
    model = ResNet_model(dropout=args.dropout, normalization= args.normalization, num_classes=nb_classes, dataset=args.dataset, n=args.resblocks)
elif args.arch=='wresnet':
    model = WResNet_model(num_classes=nb_classes, depth=16, multiplier=4, bn=True)
elif args.arch == 'vgg11':
    model = vgg11(dropout=args.dropout, normalization= args.normalization, dataset=args.dataset, affine=not args.noaffine)
elif args.arch == 'mlp':
    model = MLPNet(nhiddens = [500]*args.L, dropout=args.dropout, normalization= args.normalization)
# or generally do : ResNet(n=9, nb_filters=16, num_classes=nb_classes, dropout=args.dropout)
nb = 0
if args.init == 'he':
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nb += 1
            print ('Update init of ', m)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d) and not args.noaffine:
            print ('Update init of ', m)
            m.weight.data.fill_(1)
            m.bias.data.zero_()
if args.init == 'proposed':
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nb += 1
            print ('Update init of ', m)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(1./ n))
        elif isinstance(m, nn.Linear):
            nb += 1
            print ('Update init of ', m)
            m.weight.data.normal_(0, math.sqrt(1./ m.weight.data.size()[1]))
        elif isinstance(m, nn.BatchNorm2d) and not args.noaffine:
            print ('Update init of ', m)
            m.weight.data.fill_(1)
            m.bias.data.zero_()
print( 'Number of Conv layers: ', (nb))

best_acc=0
lr_list = []
train_loss_list = []
train_acc_list = []
valid_acc_list = []


if args.cuda:
    model.cuda()
total_params = sum(np.prod(x.size()) if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Args:', args)
print( 'Model total parameters:', total_params)
with open(args.save_dir + '/log.txt', 'a') as f:
    f.write(str(args) + ',total_params=' + str(total_params) + '\n')

criterion = nn.CrossEntropyLoss()


###############################################################################
# Training code
###############################################################################

def test(epoch, loader, valid=False,train_loss=None):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        if not args.cluster:
            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint.
    acc = 100.*float(correct)/float(total)
    if valid and acc > best_acc:
        print('Saving best model..')
        state = {
            'net': model,
            'acc': acc,
            'epoch': epoch,
        }
        with open(args.save_dir + '/best_model.pt', 'wb') as f:
                torch.save(state, f)
        best_acc = acc
    return acc

global_iters=0
last_load_epoch=0
def train(epoch):
    global trainloader
    global optimizer
    global args
    global model, global_iters,last_load_epoch
    # Turn on training mode which enables dropout.
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_list = []
    total_loss_list=[]

    for lr_ in lr_sch:
        if epoch<= lr_[0]:
            lr = lr_[1]
            break

    optimizer.zero_grad()
    if not hasattr(train, 'nb_samples_seen'):
        train.nb_samples_seen = 0

    iters=0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()


        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        loss_list.append(loss.data.cpu().numpy() )

        if train.nb_samples_seen+args.mbs==args.bs:
            global_iters+=1
            iters+=1

            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr*lr

            for name, variable in model.named_parameters():
                g = variable.grad.data
                g.mul_(1./(1+train.nb_samples_seen/float(args.mbs)))

            optimizer.step()

            optimizer.zero_grad()
            train.nb_samples_seen = 0
            total_loss_list.extend(loss_list)
            loss_list=[]
        else:
            train.nb_samples_seen += args.mbs



        
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if not args.cluster:
            progress_bar(batch_idx, len(trainloader), 'Epoch {:3d} | Loss: {:3f} | Acc: {:3f}'
                         .format (epoch, train_loss/(batch_idx+1), 100.*float(correct)/float(total)))
    return sum(total_loss_list)/float(len(total_loss_list)), 100.*correct/total, iters
# Loop over epochs.

if args.opt=='sgd':
    optimizer = SGD(model.parameters(), lr=args.lr*lr_sch[0][1], momentum=args.m, weight_decay=args.wdecay, nesterov=args.nesterov)
else:
    print('Optimizer not supported')
    exit(0)

valid_acc=0
best_val_acc=0
for epoch in range(start_epoch, args.epochs+1):
    epoch_start_time = time.time()



    loss, train_acc, iters = train(epoch)


    train_loss_list.append(loss)
    train_acc_list.append(train_acc)
    valid_acc = test(epoch, validloader, valid=True, train_loss=loss)
    valid_acc_list.append(valid_acc)
    with open(args.save_dir + "/valid_acc.pkl", "wb") as f:
        pkl.dump(valid_acc_list, f)


    with open(args.save_dir + "/train_loss.pkl", "wb") as f:
        pkl.dump(train_loss_list, f)

    with open(args.save_dir + "/train_acc.pkl", "wb") as f:
        pkl.dump(train_acc_list, f)


    if best_val_acc<valid_acc:
        best_val_acc = valid_acc
    writer.add_scalar('/SGD:best_val_acc', best_val_acc, epoch)


    lr_list.append(optimizer.param_groups[0]['lr'])
    writer.add_scalar('/SGD:LR', lr_list[-1], epoch)

    for param_group in optimizer.param_groups:
        wdecay_ = param_group['weight_decay']
        break

    status = 'Epoch {}/{} | Loss {:3.4f} | Acc {:3.2f} | val-acc {:3.2f} | LR {:4.4f} | BS {} | momentum {} | wdecay {}'.\
        format( epoch,args.epochs, loss, train_acc, valid_acc, lr_list[-1], args.bs, args.m, wdecay_)
    print (status)
    writer.add_scalar('/SGD:loss', loss, epoch)
    writer.add_scalar('/SGD:Acc', train_acc, epoch)
    writer.add_scalar('/SGD:val_acc', valid_acc, epoch)
    with open(args.save_dir + '/log.txt', 'a') as f:
        f.write(status + '\n')

    with open(args.save_dir + "/LR_list.pkl", "wb") as f:
            pkl.dump(lr_list, f)



    print('-' * 89)


# Load the best saved model.
with open(args.save_dir + '/best_model.pt', 'rb') as f:
    best_state = torch.load(f)
model = best_state['net']




# Run on test data.
test_acc = test(epoch, testloader)
writer.add_scalar('/SGD:test_acc', test_acc, 0)
print('=' * 89)
print('| End of training | test acc {}'.format(test_acc))


print('=' * 89)
