import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import random_split
import utils




import tensorboard.backend.event_processing.event_accumulator as ea



from ed import *
from torch.distributions.beta import Beta

import numpy as np
import scipy
from scipy import optimize
import matplotlib.pyplot as plt
#from tensorboard_logger import configure, log_value

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

##from prettytable import PrettyTable

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--aug-lr', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--no-nesterov', dest='nesterov', default=True,  action='store_false', help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq_train', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--print-freq_test', default=20, type=int,
                    help='print frequency (default: 10)')

parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dir', type=str, default="TMP", help='Results dir name')



parser.add_argument('--cont', action='store_true', default=False,
                    help='just load checkpoint')

parser.add_argument('--cont-src', default="", type=str,
                    help='source to load weights from')

parser.add_argument('--name', type=str, default="", help='Name of event file')

parser.add_argument('--arch', default="ED-FC", type=str,
                    help='architecture')

parser.add_argument('-D', '--tD', action='store_true', default=False,
                    help='transposable activation')

parser.add_argument('-W', '--tW', action='store_true', default=False,
                    help='transposable weights')

parser.add_argument('--single',  action='store_true', default=False,
                    help='single version of transposed net')

parser.add_argument('--no-skips', dest= 'skips', action='store_false', default=True,
                    help='use skips')

parser.add_argument('--no-cosine', dest= 'cosine', action='store_false', default=True,
                    help='dont use cosine annealing')

parser.add_argument('--layers', '-L', default=4, type=int,
                    help='number of layers')


parser.add_argument('--start-epoch', default=0, type=int,
                    help='epoch to start from')

parser.add_argument('-w', '--widening', default=2, type=int,
                    help='widening factor')

parser.add_argument('--first', default=None, type=int,
                    help='first hidden layer')

parser.add_argument('--strides', default=1, type=int,
                    help='strides')

parser.add_argument('-a', '--act', default="relu", type=str,
                    help='activation')

parser.add_argument('--dataset', default="mnist", type=str,
                    help='activation')

parser.add_argument('--sigma', default=1.25, type=float, help='momentum')


parser.add_argument('--epsilon', default=1e-5, type=float, help='epsilon for finite diff')
parser.add_argument('--mage', action='store_true', default=False, help="Daniel's scheme")
parser.add_argument('--per-batch', action='store_true', default=False, help="Calc V per batch")
parser.add_argument('--finite-diff', action='store_true', default=False, help="use finite diff")
parser.add_argument('--fwd-mode', action='store_true', default=False, help="use forward mode")
parser.add_argument('--resample', action='store_true', default=False, help="sampe new delta for each minibatch")

parser.add_argument('--directional', action='store_true', default=False, help="use backprop with directional derivative")


parser.add_argument('--epoch-scale', default=1, type=int, help="epoch multiplier")
parser.add_argument('--dont-normalize', dest="normalize_v", action='store_false', default=True, help="normalize V")
parser.add_argument('--double', action='store_true', default=False, help="double precision")

parser.add_argument('--replicates', default=0, type=int, help="replicates")

parser.add_argument('--sparsity', default=0.0, type=float, help='deltas sparsity')


parser.add_argument('--ndirections', default=1, type=int, help='number of random directions to average on')

parser.add_argument('--rundir', type=str, default=".", help='Results dir name')


parser.set_defaults(augment=True)



def already_exists(path, epochs, delete = True):
    

    if not os.path.exists(path):
        return False
    dirslist = os.listdir(path)
    for event in dirslist:
        event_acc = ea.EventAccumulator(f"{path}/{event}")
        event_acc.Reload()

        print(f"{path}/{event}")
        if 'test/loss' in event_acc.scalars.Keys():
            if len(event_acc.scalars.Items('test/loss')) >= epochs:
                return True
        if delete:
            os.remove(f"{path}/{event}")
        
    return False



def parse_txt(cmd):
    return parser.parse_args(cmd.replace("  "," ").split(" "))

def count_parameters(model):
    return None
    # table = PrettyTable(["Modules", "Parameters"])
    # total_params = 0
    # for name, parameter in model.named_parameters():
    #     if not parameter.requires_grad: continue
    #     param = parameter.numel()
    #     table.add_row([name, param])
    #     total_params+=param
    # print(table)
    # print(f"Total Trainable Params: {total_params}")
    return total_params

def auto_name(args):
    if args.first is None:
        txt = f"{args.dataset}_{args.arch}_{args.layers}x{args.widening}_B{args.batch_size}_V2_"
    else:
        txt = f"{args.dataset}_{args.arch}_{args.layers}-{args.first}x{args.widening}_B{args.batch_size}_V2_"

    loglr = round(np.log(args.lr)/np.log(10))
    lead = round(args.lr / (10**loglr))

    lr = f"{lead}E{loglr}"
    txt = txt + f"LR{lr}_"

    if not args.cosine:
        txt = txt + "noCosine_"

    if args.double:
        txt = txt + "fp32_"

    if args.resample:
        txt = txt + "resampleV2_"

    if args.ndirections > 1:
        txt = txt + f"{args.ndirections}-Directions_"

    if args.sparsity > 0.0:
        txt = txt + f"sparse{int(args.sparsity*100)}_"

    if args.finite_diff or args.fwd_mode:
        if args.finite_diff:
            txt = txt + "finiteDiff_"
        else:
            txt = txt + "fwdModeV6_"
        if args.mage:
            txt = txt + "mageV3_"##v2: fixed normalization
            if args.per_batch:
                txt = txt + "PerBatchV4_"
            if args.normalize_v:
                txt = txt + "Normalized_"

            if not args.replicates == 0:
                txt = txt + f"Avg{args.replicates}_"
    else:
        if args.directional:
            txt = txt + "directionalV2_"
            if args.mage:
                txt = txt + "mage_" 
                if args.normalize_v:
                    txt = txt + "Normalized_"
    if len(args.name) > 0:
        txt = txt + args.name
    else:
        txt = txt[:-1]

        


    return txt


def get_model(args):


    utils.set_seed(args.seed)


    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data loading code




    print('==> Preparing data..')

    kwargs = {'num_workers': 1, 'pin_memory': True}

    if args.dataset == "cifar10":


        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

        dataset = 'cifar10'
        input_size = 32
        input_channels = 3
        n_classes = 10
    else:
        normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        n_classes = 10
        dataset = 'mnist'
        input_size = 28
        input_channels = 1

    transform=transforms.Compose([
        transforms.ToTensor()])
        ##normalize
        ##])

    aname = auto_name(args)
    save_path = os.path.join("./logs", aname + "/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ds_train = datasets.__dict__[dataset.upper()](root='../data', train=True, download=True, transform=transform)
    ds_val = datasets.__dict__[dataset.upper()](root='../data', train=False, transform=transform)


    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=True, **kwargs)


    # create model
    args.arch = args.arch.lower()
    if args.arch == "ed-fc":
        model = ED_FC(input_size, args.layers, args.tW, args.tD, args.skips, args.act, args.widening, lamb = 0.0)
    elif args.arch == "ed-conv":
        model = ED_Conv(input_size, args.layers, args.tW, args.tD, args.skips, args.act, args.widening, lamb = 0.0, input_channels = input_channels, strides = args.strides)
    elif args.arch == "dbl-fc":
        model = dbl_FC(input_size, args.layers, (not args.single))
    elif args.arch == "fc":
        model = PlainFC(input_size, args.layers, args.act, args.widening, lamb = 0.0, classes = n_classes, first_layer = args.first)
    else:
        raise ValueError("wrong arch: %f" % args.arch)
    # get the number of model parameters
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.parameters()])))

    count_parameters(model)

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    if args.double:
        model = model.double()

    if args.cont:
        if len(args.cont_src)  == 0:
            re = True
            retxt = ""
            for retxt in ["","500","200","100"]:
                try:
                    load_path = os.path.join("./logs", aname + "/")
                    state_dict = torch.load(load_path + f"ED{retxt}.pt")
                    model.load_state_dict(state_dict)
                    model.eval()
                    re = False
                    print("loaded %s" % (load_path + f"ED{retxt}.pt"))
                    break
                except:
                    pass
        else:
            load_path = args.cont_src
            state_dict = torch.load(load_path)
            model.load_state_dict(state_dict)
            model.eval()
            re = False
            print("loaded %s" % load_path)
                

    return model, dl_train, dl_val, device, aname, save_path

def steps_lr(optimizer, args, epoch):
    if epoch < args.epochs//5:
        lr = args.lr
    elif epoch < 2*args.epochs//5:
        lr = args.lr * 3e-1
    elif epoch < 3*args.epochs//5:
        lr = args.lr * 1e-1
    elif epoch < 4*args.epochs//5:
        lr = args.lr * 1e-2
    else:
        lr = args.lr * 1e-3
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    global args
    args = parser.parse_args()
    
    model, dl_train, dl_val, device, aname, save_path = get_model(args)

    # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.weight_decay)

    # cosine learning rate
    ##scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ((len(train_loaders)*args.levels)//args.M)*args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (len(dl_train))*(args.epochs//args.epoch_scale))


    print(f"name: {aname}")

    writer_path = "%s/runs/%s" % (args.rundir,aname)
    if already_exists(writer_path,args.epochs):
        print("File already exists and is full: aborting")
        return 0

    writer = SummaryWriter(log_dir=writer_path, comment=str(args))


    loss_lst = []
    test_loss_lst = []

    for epoch in range(args.start_epoch,args.epochs//args.epoch_scale):

        if args.fwd_mode:
            train_loss,train_acc = train_fwd(dl_train, model, args , optimizer, scheduler, epoch*args.epoch_scale, device, writer, args.epsilon, args.mage, args.resample, args.sparsity )
        elif args.finite_diff:
            train_loss,train_acc = train_v(dl_train, model, args , optimizer, scheduler, epoch*args.epoch_scale, device, writer, args.epsilon, args.mage)
        else:
            train_loss,train_acc = train(dl_train, model, args , optimizer, scheduler, epoch*args.epoch_scale, device, writer)

        test_loss, test_acc = test(dl_val, model, args, device, epoch*args.epoch_scale , writer)
        loss_lst.append(train_loss)
        test_loss_lst.append(test_loss)


        print('Train Loss: %.3f | Test Loss: %.3f | Train Accuracy: %.3f | Test Accuracy: %.3f ' % (train_loss, test_loss, train_acc, test_acc))

    print('Finished training!')


    loss_lst = np.asarray(loss_lst)
    test_loss_lst = np.asarray(test_loss_lst)
    np.save(save_path + '/train_loss.npy', loss_lst)
    np.save(save_path + '/test_loss.npy', test_loss_lst)
    epochs = np.arange(args.epochs//args.epoch_scale)*args.epoch_scale

    plt.switch_backend('agg')

    plt.figure()
    plt.plot(epochs, test_loss_lst, label='Test Loss')
    plt.plot(epochs, loss_lst, label='Train Loss')
    plt.legend()
    plt.ylabel('Accuracy [%]')
    plt.xlabel('Number of epochs')
    plt.savefig(save_path + "/accuracy.png", bbox_inches='tight')
    plt.figure()
    plt.plot(epochs, loss_lst, label='loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Number of epochs')
    plt.savefig(save_path + "/loss.png", bbox_inches='tight')

    if args.save_model:
        torch.save(model.state_dict(), save_path + "/ED.pt")


def train(train_loader, model, args, optimizer, scheduler, epoch, device, writer=None):


    """Train for one epoch on the training set"""
    # switch to train mode
    
    model.train()
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    total = 0
    correct = 0



    for j, (input, target) in tqdm(enumerate(train_loader), total = len(train_loader)):

        optimizer.zero_grad()

        input = input.to(device)
        target = target.to(device)

        if args.double:
            input = input.double()
            
        
        output = model(input)
        loss =F.cross_entropy(output, target, reduction  = 'mean')

        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += loss.item()


        loss.backward()
        with torch.no_grad():
            if args.directional:
                
                V = {}

                if args.mage:
                    x = input.view(input.shape[0],-1).clone()
                    device =x.device
                    dtype = x.dtype
                epsilon = 1
                eps = 1E-6


                dFg = 0

                for i, linop in enumerate(model.linops):
                    ##import pdb; pdb.set_trace()
                    if args.mage:

                        W = linop.weight
                        B = linop.bias
                        vn = torch.randn([W.shape[0],1],device =device, dtype = dtype) *epsilon
                        vb = vn.clone().squeeze().expand(B.shape)

                        if args.normalize_v:
                            z = x/(x.norm(dim=1,keepdim = True)+ eps)
                            vw = torch.matmul(vn,z.unsqueeze(1)) ##     M X 1     mm  B X 1 X N   -> B X M X N   
                        else:
                            vw = torch.matmul(vn,x.unsqueeze(1))   



                        batch_size = vw.shape[0]
                        dFg = dFg +  (linop.weight.grad*vw).sum([1,2]).view([batch_size, 1])   ## B x 1
                        
                        
                        x = linop(x)
                        if i < len(model.linops) - 1:
                            x = model.act(x)

                    else:
                        vw = torch.randn_like(linop.weight)
                        vb = torch.randn_like(linop.bias)
                        dFg = dFg  + (linop.weight.grad*vw).sum()

                    dFg = dFg  + (linop.bias.grad*vb).sum()
                    V[i] = (vw, vb)

                for i, linop in enumerate(model.linops):
                    vw, vb = V[i]
                    if args.mage:
                        linop.weight.grad = torch.matmul(dFg.permute(1,0), vw.view(batch_size,-1)).view(vw.shape[1],vw.shape[2])   ## 1 x B   mm   Bx(N1xN2)  >   N1 x N2
                    else:
                        linop.weight.grad = (linop.weight.grad*vw).sum() * (vw)
                    linop.bias.grad = (linop.bias.grad*vb).sum() * (vb)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)


        total += input.size(0)
        optimizer.step()
        if args.cosine:
            scheduler.step()
        else:
            steps_lr(optimizer,args,epoch)


    train_acc = (100. * correct / len(train_loader.dataset))
    train_loss = train_loss/total
    
    if writer is not None:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        for i,linop in enumerate(model.linops) :
            if i == 0:
                stri=""
            else:
                stri = f"{i}"
            writer.add_scalar(f'L2norm/weight{stri}', linop.weight.norm().item(), epoch)
            if not linop.bias is None:
                writer.add_scalar(f'L2norm/bias{stri}', linop.bias.norm().item(), epoch)

    return train_loss, train_acc


def L2_np(x):
    return x.norm().cpu().numpy()**2






def train_v(train_loader, model, args, optimizer, scheduler, epoch, device, writer=None,epsilon = 1e-5, mage = False):
    """Train for one epoch on the training set"""
    # switch to train mode
    
    model.train()
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    total = 0
    correct = 0
    
    
    for i, (input, target) in tqdm(enumerate(train_loader), total = len(train_loader)):
        optimizer.zero_grad()

        batch_size = input.shape[0]

        
        input = input.to(device)
        target = target.to(device)

        if args.double:
            input = input.double()

        x1, x2, V = model.grad_v(input, mage = mage,  epsilon = epsilon, per_batch = args.per_batch, normalize_v = args.normalize_v, replicates = args.replicates)
        
        
        
        for linop in model.linops:
            linop.weight.grad = torch.zeros_like(linop.weight)
            linop.bias.grad = torch.zeros_like(linop.bias)
            
        if mage and not args.per_batch:
            loss1 =F.cross_entropy(x1, target, reduction  = 'none')
            loss2 =F.cross_entropy(x2, target, reduction  = 'none')

            for b in range(batch_size):
                v_b_norm = np.sqrt(np.sum([ L2_np(_vw[b]) + L2_np(_vb)  for _vw,_vb in V.values()]))
                dF = (loss2[b]-loss1[b])/(v_b_norm*epsilon)
                dF = dF/batch_size
                for i,(vw,vb) in V.items():
                    linop = model.linops[i]
                    linop.weight.grad += dF*vw[b]
                    linop.bias.grad += dF*vb
                    
            loss = loss1.mean()
        else:
            loss1 =F.cross_entropy(x1, target, reduction  = 'mean')
            loss2 =F.cross_entropy(x2, target, reduction  = 'mean')
            
            v_norm = np.sqrt(np.sum([ L2_np(_vw) + L2_np(_vb)  for _vw,_vb in V.values()]))
            dF = (loss2-loss1)/(v_norm*epsilon)
            for i,(vw,vb) in V.items():
                linop = model.linops[i]
                linop.weight.grad += dF*vw
                linop.bias.grad += dF*vb
                
            loss = loss1
        

        pred = x1.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss += loss.item()

        total += input.size(0)
        optimizer.step()

        if args.cosine:
            scheduler.step()
        else:
            steps_lr(optimizer,args,epoch)

    train_acc = (100. * correct / len(train_loader.dataset))
    train_loss = train_loss/total
    if writer is not None:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)

        for i,linop in enumerate(model.linops) :
            if i == 0:
                stri=""
            else:
                stri = f"{i}"
            writer.add_scalar(f'L2norm/weight{stri}', linop.weight.norm().item(), epoch)
            if not linop.bias is None:
                writer.add_scalar(f'L2norm/bias{stri}', linop.bias.norm().item(), epoch)

    return train_loss, train_acc

from torch.nn.modules.loss import _WeightedLoss

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0, constant_addition = 0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.C = constant_addition
    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        ##import pdb; pdb.set_trace()
        log_preds = F.log_softmax(inputs, -1)
        #exp_in = torch.exp(inputs-)
        #log_preds = inputs - c - torch.log(exp_in.sum(1, keepdim = True))

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss( (-(targets * log_preds).sum(dim=-1)))

class MSELoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0, constant_addition = 0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.C = constant_addition
    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)

        preds = (targets - inputs)**2

        if self.weight is not None:
            preds = preds * self.weight.unsqueeze(0)

        return self.reduce_loss(preds.sum(dim=-1))


def train_fwd(train_loader, model, args, optimizer, scheduler, epoch, device, writer=None,epsilon = 1e-5, mage = False, resample = False, sparsity = 0.0):
    """Train for one epoch on the training set"""
    # switch to train mode
    watcher = True
    model.train()
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    total = 0
    correct = 0
    
    #mloss = SmoothCrossEntropyLoss(reduction  = 'mean', smoothing  = 0.0, constant_addition = 1.0)
    #mloss = MSELoss(reduction  = 'mean', smoothing  = 0.0)
    # if mage and not args.per_batch:
    #     mloss = torch.nn.CrossEntropyLoss(reduction  = 'none')
    # else:    
    mloss = torch.nn.CrossEntropyLoss(reduction  = 'mean')

    for i, (input, target) in tqdm(enumerate(train_loader), total = len(train_loader)):
        optimizer.zero_grad()

        batch_size = input.shape[0]

        
        input = input.to(device)
        target = target.to(device)

        if args.double:
            input = input.double()


        # if i >= 20:
        #     import pdb; pdb.set_trace();

        for _ in range(args.ndirections):
            out = model.fwd_mode(input, target, lambda x,y: mloss(x,y)/args.ndirections, mage = mage,  epsilon = epsilon, per_batch = args.per_batch, normalize_v = args.normalize_v, resample = resample, sparsity = sparsity)        
            loss = F.cross_entropy(out, target, reduction  = 'mean')/args.ndirections
            train_loss += loss.item()
            total += input.size(0)

        pred = out.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)

        optimizer.step()

        if args.cosine:
            scheduler.step()
        else:
            steps_lr(optimizer,args,epoch)

        if watcher and np.isnan(train_loss):
            watcher = False
            print(f"Turned Nan on step {i}")

    train_acc = (100. * correct / len(train_loader.dataset))
    train_loss = train_loss/total
    if writer is not None:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)

        for i,linop in enumerate(model.linops) :
            if i == 0:
                stri=""
            else:
                stri = f"{i}"
            writer.add_scalar(f'L2norm/weight{stri}', linop.weight.norm().item(), epoch)
            if not linop.bias is None:
                writer.add_scalar(f'L2norm/bias{stri}', linop.bias.norm().item(), epoch)


    return train_loss, train_acc





def test(test_loader, model, args, device, epoch, writer=None):
    """Perform test on the test set"""
    # switch to evaluate mode
    model.eval()
    test_loss = 0
    total = 0
    correct = 0
    for i, (input, target) in tqdm(enumerate(test_loader), total = len(test_loader)):
        input = input.to(device)
        target = target.to(device)

        if args.double:
            input = input.double()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss =F.cross_entropy(output, target, reduction  = 'mean')
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss += loss.item()

        total += input.size(0)
        # if i % args.print_freq_test == 0:
        #     print('Test Loss: %.3f' % (test_loss / (i + 1)))

    test_acc = (100. * correct / len(test_loader.dataset))
    test_loss = test_loss/(i+1)

    if writer is not None:
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)


    return test_loss, test_acc






if __name__ == '__main__':
    main()
