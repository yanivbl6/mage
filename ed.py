'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import wandb

import math
from collections import OrderedDict
import torch.nn.functional as F

import numpy as np
def str2act(txt, param= None):
    return {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU(), "none": nn.Sequential() , "lrelu": nn.LeakyReLU(param), "selu": nn.SELU() }[txt.lower()]

def decompose(z, alpha):
    device = z.device

    znorm2 = (z.norm(dim=2, keepdim = True))**2
    ##znorm2 = torch.ones([1,1,1], device = device)


    factor = torch.sqrt(alpha**2+ (1-alpha**2)* (znorm2))  - alpha
    factor = factor/ znorm2
    L =  alpha*torch.eye(z.size(2), device = device).unsqueeze(0) + factor * (z.permute(0,2,1) @ z)
    return L


def batched_sherman_morrison(guess,t, alpha):
    z = guess[t].unsqueeze(1)
    z = z / z.norm(dim=2, keepdim = True)
    device = z.device
    I = torch.eye(z.shape[2], device = device).unsqueeze(0)

    if alpha == 0.0:
        return I

    factor = (1-alpha**2)/(alpha**2)
    den = 1 + factor* torch.matmul(z,z.permute(0,2,1))
    sol = I - factor* torch.matmul(z.permute(0,2,1),z)/den
    return sol/alpha**2


def batched_sherman_morrison2(z, alpha):
    device = z.device
    I = torch.eye(z.shape[2], device = device).unsqueeze(0)
    S1 = I
    S2 = (I - torch.matmul(z.permute(0,2,1),z)) *(1-alpha**2)/(alpha**2)

    return S1, S2

def projections(z):
    device = z.device
    I = torch.eye(z.shape[2], device = device).unsqueeze(0)
    gg = (torch.matmul(z.permute(0,2,1),z))
    I_gg = I - gg
    return gg, I_gg


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
   
    def forward(self, x):
        return x.view(x.shape[0],-1) 

def init_layer(op):
    stdv = 1. / math.sqrt(op.weight.size(1)) 
    op.weight.data.uniform_(-stdv, stdv) 
    if op.bias is not None: 
        op.bias.data.fill_(0.0)


class ED_Conv(nn.Module):
    def __init__(self, input_size, layers, tW, tD, skips, act, widening, lamb = 0.0, K = 3, input_channels = 1, strides = 1):
        super(ED_Conv,self).__init__()

        self.act = act


        self.input_size  = input_size
        self.layers  = layers
        self.tD  = tD
        self.tW  = tW
        self.skips = skips
        self.lamb = lamb
        self.use_batchnorm  = False
        layer_specs = [1] + [ 16*widening**i for i in range(layers)]


        layer_specs = layer_specs[0:self.layers+1]

        self.encoders = nn.ModuleList()
        num_strides = 0
        ins = input_size 
        while ins % strides == 0:
            num_strides = num_strides + 1
            ins = ins//strides

        strides_l = np.asarray([(l+1)*num_strides/self.layers for l in range(self.layers)  ]) 
        strides_l = strides_l.round()
        strides_l[1:] = strides_l[1:] - strides_l[:-1]


        conv, pad = self._gen_conv(layer_specs[0] ,layer_specs[1], rounding_needed  = True, strides= strides if strides_l[0] == 1.0 else 1)
        op = nn.Sequential(pad, conv)

        ##op = nn.Linear(layer_specs[0], layer_specs[1], bias = True)
        ##init_layer(op)




        self.encoders.append(op)
        
        last_ch = layer_specs[1]

        for i,ch_out in enumerate(layer_specs[2:]):
            d = OrderedDict()
            d['act'] = str2act(self.act,self.lamb)
            gain  = math.sqrt(2.0/(1.0+self.lamb**2))
            gain = gain / math.sqrt(2)  ## for naive signal propagation with residual w/o bn

            conv, pad  = self._gen_conv(last_ch ,ch_out, gain = gain, strides= strides if strides_l[i+1] == 1.0 else 1)
            if not pad is None:
                d['pad'] = pad

            conv.weight.data.fill_(0.0)

            for j in range(last_ch):
                if j < ch_out:
                    conv.weight.data[j,j,1,1] = 1.0


            d['conv'] = conv

            if self.use_batchnorm:
                d['bn']  = nn.BatchNorm2d(ch_out)

            print("last_ch: %d, ch_out: %d " %(last_ch, ch_out))

            ##linop = nn.Linear(last_ch, ch_out, bias = True)
            ##init_layer(linop)
            ##d['linear'] = linop

            encoder_block = nn.Sequential(d)

            self.encoders.append(encoder_block)
            last_ch = ch_out

        layer_specs.reverse()
        self.decoders = nn.ModuleList()

        for i,ch_out in enumerate(layer_specs[1:]):
            d = OrderedDict()
            d['act'] = str2act(self.act,self.lamb)
            gain  =  math.sqrt(2.0/(1.0+self.lamb**2))
            gain = gain / math.sqrt(2) 

            conv = self._gen_deconv(last_ch, ch_out , gain = gain, strides= strides if strides_l[self.layers-1-i] == 1.0 else 1, k = 3 if strides == 1 or (not strides_l[self.layers-1-i] == 1.0) else 4)
            conv.weight.data.fill_(0.0)

            if not self.skips:
                for j in range(last_ch):
                    if j < ch_out:
                        conv.weight.data[j,j,1,1] = 1.0



            d['conv'] = conv

            # # if i < self.num_dropout and self.droprate > 0.0:
            # #     d['dropout'] = nn.Dropout(self.droprate)

            if self.use_batchnorm and i < self.layers-1:
                d['bn']  = nn.BatchNorm2d(ch_out)

            print("last_ch: %d, ch_out: %d " %(last_ch, ch_out))
            ##linop = nn.Linear(last_ch, ch_out, bias = True)
            ##init_layer(linop)
            ##d['linear'] = linop


            decoder_block = nn.Sequential(d)

            self.decoders.append(decoder_block)
            last_ch = ch_out 

        self.flat = Flatten()


    def forward(self, x):
        src = x
        src_shape = x.shape

        ##x = x.view(src_shape[0],-1) 

        encoders_output = []

        for i,encoder in enumerate(self.encoders):
            x = encoder(x)
            encoders_output.append(x)

        for i,decoder in enumerate(self.decoders[:-1]):
            x = decoder(x)
            if self.skips:
                x = x + encoders_output[-(i+2)]

        x = self.decoders[-1](x) 
        x = x + src
        ##x = x.view(src_shape)
        return x

    def _gen_conv(self, in_ch,  out_ch, strides = 2, kernel_size = (3,3), gain = math.sqrt(2), rounding_needed= False):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        ky,kx = kernel_size
        p1x = (kx-1)//2
        p2x = kx-1 - p1x
        p1y = (ky-1)//2
        p2y = ky-1 - p1y

        if rounding_needed:
            pad_counts = (p1x,p1x,p1y , p1y)
            pad = torch.nn.ReplicationPad2d(pad_counts)
        else:
            pad = None

        if pad is None:
            conv =  nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride = strides, padding = (p1y, p1x) )
        else:
            conv =  nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride = strides , padding=0)

        w = conv.weight
        k = w.size(1) * w.size(2) * w.size(3)
        conv.weight.data.normal_(0.0, gain / math.sqrt(k) )
        nn.init.constant_(conv.bias,0.01)
        return conv, pad 

    def _gen_deconv(self, in_ch,  out_ch, strides = 2, k = 4, gain = math.sqrt(2), p = 1 ):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]

        conv =  nn.ConvTranspose2d(in_ch, out_ch, kernel_size= (k,k), stride = strides, padding_mode='zeros',padding = (p,p), dilation  = 1)

        w = conv.weight
        k = w.size(1) * w.size(2) * w.size(3)
        conv.weight.data.normal_(0.0, gain / math.sqrt(k) )
        nn.init.constant_(conv.bias,0.01)
        return conv


class ED_FC(nn.Module):
    def __init__(self, input_size, layers, tW, tD, skips, act, widening, lamb = 0.0):
        super(ED_FC,self).__init__()

        self.act = act


        self.input_size  = input_size
        self.layers  = layers
        self.tD  = tD
        self.tW  = tW
        self.skips = skips
        self.lamb = lamb

        layer_specs = [ (input_size*input_size)*widening**i for i in range(layers+1)]


        layer_specs = layer_specs[0:self.layers+1]


        self.encoders = nn.ModuleList()

        ##op, pad = self._gen_conv(layer_specs[0] ,layer_specs[1], convGlu = self.convGlu, rounding_needed  = True)
        ##op = nn.Sequential(pad, conv)

        op = nn.Linear(layer_specs[0], layer_specs[1], bias = True)
        init_layer(op)

        
        self.encoders.append(op)
        
        last_ch = layer_specs[1]

        for i,ch_out in enumerate(layer_specs[2:]):
            d = OrderedDict()
            d['act'] = str2act(self.act,self.lamb)
            # gain  = math.sqrt(2.0/(1.0+self.lamb**2))
            # gain = gain / math.sqrt(2)  ## for naive signal propagation with residual w/o bn
            # conv, pad  = self._gen_conv(last_ch ,ch_out, gain = gain, convGlu = self.convGlu, kernel_size = self.k_xy)
            # if not pad is None:
            #     d['pad'] = pad
            # d['conv'] = conv

            # if self.use_batchnorm:
            #     d['bn']  = nn.BatchNorm2d(ch_out)

            print("last_ch: %d, ch_out: %d " %(last_ch, ch_out))
            linop = nn.Linear(last_ch, ch_out, bias = True)
            init_layer(linop)

            
            d['linear'] = linop
            encoder_block = nn.Sequential(d)

            self.encoders.append(encoder_block)
            last_ch = ch_out

        layer_specs.reverse()
        self.decoders = nn.ModuleList()

        for i,ch_out in enumerate(layer_specs[1:]):

            d = OrderedDict()
            d['act'] = str2act(self.act,self.lamb)
            # gain  =  math.sqrt(2.0/(1.0+self.lamb**2))
            # gain = gain / math.sqrt(2) 

            # if i == len(layer_specs)-2:
            #      kernel_size = 5
            #      ch_out = 2
            # conv = self._gen_deconv(last_ch, ch_out , gain = gain, k= kernel_size)
            # d['conv'] = conv

            # # if i < self.num_dropout and self.droprate > 0.0:
            # #     d['dropout'] = nn.Dropout(self.droprate)

            # if self.use_batchnorm and i < self.n_layers-1:
            #     d['bn']  = nn.BatchNorm2d(ch_out)

            print("last_ch: %d, ch_out: %d " %(last_ch, ch_out))
            linop = nn.Linear(last_ch, ch_out, bias = True)
            init_layer(linop)
            ##d['linear'] = linop

            if self.skips or (not last_ch == ch_out):
                op.weight.data.fill_(0.0)
            else:
                op.weight.data = torch.eye(int(last_ch))

            if op.bias is not None: 
                op.bias.data.fill_(0.0)


            d['linear'] = linop
            decoder_block = nn.Sequential(d)

            self.decoders.append(decoder_block)
            last_ch = ch_out 

        self.flat = Flatten()


    def forward(self, x):
        

        src_shape = x.shape

        x = x.view(src_shape[0],-1) 

        encoders_output = []

        for i,encoder in enumerate(self.encoders):
            x = encoder(x)
            encoders_output.append(x)

        for i,decoder in enumerate(self.decoders[:-1]):
            x = decoder(x)
            if self.skips:
                x = x + encoders_output[-(i+2)]

        x = self.decoders[-1](x) 

        x = x.view(src_shape)
        return x



class dbl_FC(nn.Module):
    def __init__(self, input_size, layers, dbl = True):


        d_size = input_size*input_size
        super(dbl_FC, self).__init__()
        self.depth = layers


        self.weights = nn.ParameterList([nn.Parameter(torch.eye(int(d_size)) * (0.001 if i==0 else 1) ) for i in range(layers)])
        ##self.weights = nn.ParameterList([nn.Parameter(torch.randn(int(d_size),int(d_size)) *  1e-1  / math.sqrt(input_size)  ) for i in range(layers)])



        self.dbl = dbl

    def forward(self, input):
        output = input.view(input.size(0), -1)
        output_T = output.clone()
        masks = []
        for k in range(self.depth-1):

            
            output = F.linear(output, self.weights[k])
            output = F.relu(output)
            mask = (output != 0.0).to(dtype = torch.int)
            masks.append(mask.clone().detach())

        output = F.linear(output, self.weights[-1])
        if self.dbl:
            for k in range(self.depth-2,-1,-1):
                output_T = F.linear(output_T, torch.transpose(self.weights[k],0,1))
                output_T = output_T * masks.pop()

            output_T = F.linear(output_T, torch.transpose(self.weights[0],0,1))
            return (output + output_T).view(input.shape)
        else:
            return (output).view(input.shape)


# test()


class PlainFC(nn.Module):
    def __init__(self, input_size, layers, act, widening, lamb = 0.0, classes= 10, first_layer = None):
        super(PlainFC,self).__init__()

        self.actstr = act


        self.input_size  = input_size
        self.layers  = layers
        self.lamb = lamb


        if first_layer is None:
            layer_specs = [ (input_size*input_size)*widening**i for i in range(layers+1)]
        else:
            layer_specs = [ (input_size*input_size)] + [first_layer*widening**(i) for i in range(layers)]

        layer_specs = layer_specs[0:self.layers+1]

        self.linops = nn.ModuleList()

        ##op, pad = self._gen_conv(layer_specs[0] ,layer_specs[1], convGlu = self.convGlu, rounding_needed  = True)
        ##op = nn.Sequential(pad, conv)

        last_ch = layer_specs[0]
        self.act = str2act(self.actstr,self.lamb)

        self.num_neurons = np.sum(layer_specs[1:]) + classes

        if layers > 0:
            op = nn.Linear(last_ch, layer_specs[1], bias = True)
            init_layer(op)
            self.linops.append(op)
            last_ch = layer_specs[1]

        for i,ch_out in enumerate(layer_specs[2:]):

            linop = nn.Linear(last_ch, ch_out, bias = True)
            init_layer(linop)
            print("last_ch: %d, ch_out: %d " %(last_ch, ch_out))

            self.linops.append(linop)
            last_ch = ch_out

        print("last_ch: %d, ch_out: %d " %(last_ch, classes))
        linop = nn.Linear(last_ch, classes, bias = True)
        self.linops.append(linop)
        print(self)
        self.distinct_mapping = None


    def forward(self, x):
        
        self.guess_i = []
        def hook_fn_i(grad):
            self.guess_i.append(grad.clone().detach())
            return None

        self.anorm_tot = 0

        src_shape = x.shape
        x = x.view(src_shape[0],-1) 

        for i,linop in enumerate(self.linops):
            self.anorm_tot = self.anorm_tot + x.norm(dim=1,keepdim = True)**2 + 1
            x = linop(x)

            x = x.requires_grad_(True)
            ##x = torch.tensor(x, requires_grad = True)
            x.register_hook(hook_fn_i)

            if i < len(self.linops) - 1:
                x = self.act(x)

        self.anorm_tot = torch.sqrt(self.anorm_tot)

        return x


    def grad_v(self,x, mage = False, epsilon=1e-5, per_batch = False, normalize_v = False, replicates = 0):
        with torch.no_grad():

            src_shape = x.shape
            x = x.view(src_shape[0],-1) 
            x2 = x.clone()
            V = {}
            device =x.device
            dtype = x.dtype
            for i,linop in enumerate(self.linops):
                W = linop.weight
                B = linop.bias

                vb = torch.randn(B.shape,device =device, dtype = dtype) *epsilon

                if mage:
                    
                    # vn = torch.randn([W.shape[0],1],device =device) *epsilon

                    # if normalize_v:
                    #     z = x/((x.norm(dim=1,keepdim = True))**2)
                    #     vw = torch.matmul(vn,z.unsqueeze(1))
                    # else:
                    #     vw = torch.matmul(vn,x.unsqueeze(1))

                    # x2 = torch.matmul(W + vw,x2.unsqueeze(2)).squeeze() + (B + vb)

                    # if per_batch:
                    #     vw = torch.matmul(vn,x.unsqueeze(1)).mean(0)

                    if not per_batch:
                        vn = torch.randn([W.shape[0],1],device =device, dtype = dtype) *epsilon

                        if normalize_v:
                            z = x/(x.norm(dim=1,keepdim = True))
                            vw = torch.matmul(vn,z.unsqueeze(1))
                        else:
                            vw = torch.matmul(vn,x.unsqueeze(1))

                        x2 = torch.matmul(W + vw,x2.unsqueeze(2)).squeeze() + (B + vb)
                    else:
                        vn = torch.randn([W.shape[0],1],device =device, dtype = dtype) *epsilon

                        if replicates == 0:
                            if normalize_v:
                                z = x.mean(0)
                                z = z/z.norm()
                                vw = torch.matmul(vn,z.unsqueeze(0))
                            else:
                                vw = torch.matmul(vn,x.mean(0).unsqueeze(0))
                        else:

                            vn = vn.expand(W.shape[0],src_shape[0])

                            if replicates > 0:
                                mask =0
                                iterations = max(src_shape[0]//W.shape[0],replicates)
                                for _ in range(iterations):
                                    mask = mask + torch.eye(src_shape[0], device = device)[np.random.choice(src_shape[0], W.shape[0], replace = True)]
                                mask = mask/iterations
                                vn = vn*mask
                            

                            if normalize_v:
                                z = x/x.norm()
                                vw = torch.matmul(vn,z)
                            else:
                                vw = torch.matmul(vn,x)


                        x2 = F.linear(x2, W + vw) + (B + vb)


                else:
                    vw = torch.randn(W.shape,device =device) *epsilon
                    x2 = F.linear(x2, W + vw, B + vb)

                x = linop(x)

                V[i] = (vw, vb)

                if i < len(self.linops) - 1:
                    x = self.act(x)
                    x2 = self.act(x2)


        return x,x2,V 

    def fwd_mode(self,x, y, loss , mage = False, epsilon=1e-5, per_batch = False, normalize_v = False, replicates = 0, resample = True, sparsity = 0.0, binary = False):

        tot_norm = 0
        epsilon = 1
        eps = 1E-6
        batch_size = x.shape[0]
        old_grad = None
        delta = 1

        device =x.device
        dtype = x.dtype

        with torch.no_grad():
            assert(self.actstr == "relu")
            src_shape = x.shape
            x = x.view(src_shape[0],-1) 
            V = {}


            if resample:
                num_neurons = self.num_neurons
                if self.distinct_mapping is None:
                    if (batch_size >  num_neurons):
                        self.distinct_mapping  = torch.zeros([batch_size, num_neurons], device = device)
                        self.distinct_mapping[:num_neurons,:] = torch.eye(num_neurons ,device = device)
                    else:
                        times = ((num_neurons+batch_size-1)//batch_size)
                        self.distinct_mapping = torch.zeros([times*batch_size, num_neurons], device = device)
                        self.distinct_mapping[:num_neurons,:] = torch.eye(num_neurons ,device = device)
                        ## rest are zeros

                if (batch_size >  num_neurons):
                    sprase_mask = (torch.rand([batch_size-num_neurons,num_neurons],device = device) >= sparsity).to(torch.float)
                    self.distinct_mapping[num_neurons:,:] = torch.randn([batch_size-num_neurons,num_neurons],device = device) * sprase_mask
                    mmapping  = self.distinct_mapping[torch.randperm(batch_size),:]
                else:
                    self.distinct_mapping[:num_neurons,:] = self.distinct_mapping[torch.randperm(num_neurons),:]
                    times = ((num_neurons+batch_size-1)//batch_size)
                    mmapping = (self.distinct_mapping.view(times,batch_size,-1).sum(0))[torch.randperm(batch_size),:]                    
                w_start_idx  =0

            for i,linop in enumerate(self.linops):
                W = linop.weight
                B = linop.bias

                ##vb = torch.randn(B.shape,device =device, dtype = dtype) *epsilon

                if mage:
                    ##import pdb; pdb.set_trace()

                    if resample:


                        if binary:
                            vb = (torch.randint(0,2,[batch_size,W.shape[0]],device =device, dtype = dtype) *2 -1) *epsilon 
                        else:
                            vb = torch.randn([batch_size,W.shape[0]],device =device, dtype = dtype) *epsilon 

                        if False and sparsity > 0.0:
                            vnmask = (torch.rand(vb.shape,device =device, dtype = dtype) >= sparsity).to(torch.float)
                            vb = vb*vnmask
                        else:
                            ##import pdb; pdb.set_trace()
                            w_end_idx =  w_start_idx + W.shape[0]
                            vnmask = mmapping[:,w_start_idx:w_end_idx]
                            w_start_idx = w_end_idx
                            vb = vb*vnmask

                        if normalize_v:

                            mnorm = torch.sqrt(x.norm(dim=1,keepdim = True)**2+ 1 +  eps) 

                            z = x/mnorm  ## B X N
                            vw = torch.matmul(vb.unsqueeze(2),z.unsqueeze(1))   ##   B X M X 1    mm    B X 1 X N
                            vb = vb / mnorm
                            
                        else:
                            vw = torch.matmul(vb.unsqueeze(2),x.unsqueeze(1))

                    else:
                        if binary:
                            vn = (torch.randint(0,2,[W.shape[0],1],device =device, dtype = dtype) *2 -1) *epsilon 
                        else:
                            vn = torch.randn([W.shape[0],1],device =device, dtype = dtype) *epsilon 

                        if sparsity > 0.0:
                            vnmask = (torch.rand(vn.shape,device =device, dtype = dtype) >= sparsity).to(torch.float)
                            vn = vn*vnmask

                        vb = vn.clone().squeeze().expand(B.shape)

                        if normalize_v:
                            mnorm = torch.sqrt(x.norm(dim=1,keepdim = True)**2+ 1 +  eps) 
                            z = x/mnorm  ## B X N
                            vw = torch.matmul(vn,z.unsqueeze(1))
                            vb = vb / mnorm


                        else:
                            vw = torch.matmul(vn,x.unsqueeze(1))

                    new_grad = torch.matmul(vw, x.unsqueeze(2)).squeeze() + vb   ## B x N1

                    if per_batch:
                        vw = vw.mean(0)

                else:
                    vb = torch.randn(B.shape,device =device, dtype = dtype) *epsilon
                    vw = torch.randn(W.shape,device =device) *epsilon 
                    new_grad = F.linear(x, vw) + vb  ## B x 1 X N1

                if not old_grad is None:
                    old_grad = torch.matmul(old_grad, linop.weight.permute(1,0)) ## B X N0  mm  N0 X N1 -> B X N1
                else:
                    old_grad = 0
                old_grad = old_grad * delta + new_grad


                # if not per_batch and mage:
                #     tot_norm = tot_norm + vw.norm(1, keepdim = True)**2 + vb.norm()**2
                # else:
                #     tot_norm = tot_norm + vw.norm()**2 + vb.norm()**2

                x = linop(x)

                if i < len(self.linops) - 1:
                    mask  =  ((x >= 0).to(torch.float))

                    old_grad = old_grad * mask ## B X N1
                    if not mage:
                        maskd3 = mask.unsqueeze(2).expand([batch_size, vw.size(0), vw.size(1) ])
                        maskd3 = (mask.sum(0) > 0).to(torch.float).unsqueeze(1)
                    else:
                        maskd3 = mask.unsqueeze(2).expand( vw.shape)

                    vw = vw * maskd3   ##  B X N2 X N1  mm  

                    x = self.act(x)
                V[i] = (vw, vb)

                


        ##import pdb; pdb.set_trace()

        if mage:
            dLdout = torch.zeros_like(x)
        


        out =torch.autograd.Variable(x, requires_grad = True)
        out.grad = torch.zeros_like(out)
        L = loss(out, y)


        L.backward()
        ##import pdb; pdb.set_trace()
        dLdout = out.grad

        ##grad_transfer = dLdout.permute(1, 0) ## Batch x n_classes
        ##tot_norm = torch.sqrt(tot_norm)

        if not per_batch and mage:
            dFg = (dLdout*old_grad).sum(1, keepdim = True)
        else:
            dFg = (dLdout*old_grad).sum()

        ##dFg = dFg * ((dFg >= 0).to(torch.float)) ## DELETE ME, I AM ERROR

        ##import pdb; pdb.set_trace()
        ##import pdb; pdb.set_trace();

        for i in range(len(self.linops)):

            linop = self.linops[i]
            if linop.weight.grad is None:
                linop.weight.grad = torch.zeros_like(linop.weight)
                if not linop.bias is None:
                    linop.bias.grad = torch.zeros_like(linop.bias)



            vw, vb = V[i]
            K = len(self.linops) -1 - i
            if not per_batch and mage:
                linop.weight.grad +=  torch.matmul(dFg.permute(1,0), vw.view(vw.shape[0],-1)).view(vw.shape[1],vw.shape[2]) * (delta**K)   ## 1 x B   mm   Bx(N1xN2)  >   N1 x N2
                if not linop.bias is None:
                    if True or resample:
                        linop.bias.grad +=  (dFg* vb).sum(0) * (delta**K)
                    else:
                        linop.bias.grad +=dFg.sum() * vb * (delta**K) 

            else:
                linop.weight.grad +=   dFg* vw * (delta**K) 
                if not linop.bias is None:
                    linop.bias.grad +=  dFg * vb  * (delta**K) 



        return x

    def fwd_rb_mode(self,x, y, loss , transform = True, inv = "pinv"):


        tot_norm = 0
        eps = 1E-6
        batch_size = x.shape[0]
        old_grad = None
        delta = 1

        device =x.device
        dtype = x.dtype



        with torch.no_grad():
            assert(self.actstr == "relu")
            src_shape = x.shape
            x = x.view(src_shape[0],-1) 
            g = torch.randn_like(x).unsqueeze(2)

            V = {}


            J= torch.eye(x.shape[1],device = device).unsqueeze(0)


            for i,linop in enumerate(self.linops):

                W = linop.weight
                B = linop.bias

                x_pre = linop(x)
                if i < len(self.linops) - 1:
                    mask  =  ((x_pre >= 0).to(torch.float))
                    x_post = self.act(x_pre)
                    T = (W.t().unsqueeze(0)*mask.unsqueeze(1))
                else:
                    x_post = x_pre
                    T = (W.t().unsqueeze(0))



                ##import pdb; pdb.set_trace()

                if inv == "pinv":
                    J_cur = torch.linalg.pinv(T)
                    g = J_cur @ g
                    J = J_cur @ J 

                    
                elif inv == "lstsq":
                    T = T.cpu()
                    g = torch.linalg.lstsq(T,g.cpu(), driver = "gelsd").solution.to(device = device)
                    J = torch.linalg.lstsq(T,J.cpu(), driver = "gelsd").solution.to(device = device)
                else:
                    raise ValueError(inv)
                vb = g

                z = x/(x.norm(dim=1,keepdim = True)+ eps)  ## B X N
                vw = torch.matmul(vb,z.unsqueeze(1))   ##   B X M X 1    mm    B X 1 X N
                new_grad = (torch.matmul(vw, x.unsqueeze(2)) + vb).squeeze()   ## B x N1

                if not old_grad is None:
                    old_grad = torch.matmul(old_grad, linop.weight.permute(1,0)) ## B X N0  mm  N0 X N1 -> B X N1
                else:
                    old_grad = 0
                old_grad = old_grad + new_grad

                if i < len(self.linops) - 1:
                    old_grad = old_grad * mask ## B X N1
                    maskd3 = mask.unsqueeze(2).expand(vw.shape)
                    vw = vw * maskd3

                x = x_post


                if transform:
                    if inv == "pinv":
                        U = torch.linalg.pinv(J @ J.permute(0,2,1))
                        vw = U @ vw 
                        vb = U @ vb 
                    elif inv == "lstsq":
                        UU = (J @ J.permute(0,2,1)).cpu()
                        vw = torch.linalg.lstsq(UU, vw.cpu(), driver = "gelsd").solution.to(device = device)
                        vb = torch.linalg.lstsq(UU, vb.cpu(), driver = "gelsd").solution.to(device = device)

                V[i] = (vw.clone(), vb.clone())

                
        dLdout = torch.zeros_like(x)
        
        out =torch.autograd.Variable(x, requires_grad = True)
        out.grad = torch.zeros_like(out)
        L = loss(out, y)

        L.backward()
        dLdout = out.grad
        ##import pdb; pdb.set_trace()

        dFg = (dLdout*old_grad).sum(1, keepdim = True)

        for i in range(len(self.linops)):

            linop = self.linops[i]
            if linop.weight.grad is None:
                linop.weight.grad = torch.zeros_like(linop.weight)
                if not linop.bias is None:
                    linop.bias.grad = torch.zeros_like(linop.bias)

            vw, vb = V[i]
            linop.weight.grad +=  torch.matmul(dFg.permute(1,0), vw.reshape(vw.shape[0],-1)).view(vw.shape[1],vw.shape[2])    ## 1 x B   mm   Bx(N1xN2)  >   N1 x N2
            
            if not linop.bias is None:
                linop.bias.grad += torch.matmul(dFg.permute(1,0), vb.reshape(vw.shape[0],-1)).view(vb.shape[1])



        return x

    def fwd_rb_mode_log_eigens(self,x):


        tot_norm = 0
        eps = 1E-6
        batch_size = x.shape[0]
        old_grad = None
        delta = 1

        device =x.device
        dtype = x.dtype

        all_eigs = []

        with torch.no_grad():
            assert(self.actstr == "relu")
            src_shape = x.shape
            x = x.view(src_shape[0],-1) 
            g = torch.randn_like(x).unsqueeze(2)

            V = {}


            J= torch.eye(x.shape[1],device = device).unsqueeze(0)


            for i,linop in enumerate(self.linops):

                W = linop.weight
                B = linop.bias

                x_pre = linop(x)
                if i < len(self.linops) - 1:
                    mask  =  ((x_pre >= 0).to(torch.float))
                    x_post = self.act(x_pre)
                    T = (W.t().unsqueeze(0)*mask.unsqueeze(1))
                else:
                    x_post = x_pre
                    T = (W.t().unsqueeze(0))


                J_cur = torch.linalg.pinv(T)
                g = J_cur @ g
                J = J_cur @ J 

                _, eigs, _ = torch.linalg.svd(J)

                
                all_eigs.append(eigs.view(-1).tolist())

                vb = g

                z = x/(x.norm(dim=1,keepdim = True)+ eps)  ## B X N
                vw = torch.matmul(vb,z.unsqueeze(1))   ##   B X M X 1    mm    B X 1 X N
                new_grad = (torch.matmul(vw, x.unsqueeze(2)) + vb).squeeze()   ## B x N1

                if not old_grad is None:
                    old_grad = torch.matmul(old_grad, linop.weight.permute(1,0)) ## B X N0  mm  N0 X N1 -> B X N1
                else:
                    old_grad = 0
                old_grad = old_grad + new_grad

                if i < len(self.linops) - 1:
                    old_grad = old_grad * mask ## B X N1
                    maskd3 = mask.unsqueeze(2).expand(vw.shape)
                    vw = vw * maskd3

                x = x_post




        return all_eigs

    def fwd_mode_IG(self,x, y, loss, guess , ig = 1.0  ,   binary = False, anorm = None, gnorm = None, compensateV = None, parallel = True):

        mage = True
        tot_norm = 0
        epsilon = 1
        eps = 1E-6
        batch_size = x.shape[0]
        old_grad = None
        delta = 1

        device =x.device
        dtype = x.dtype

        with torch.no_grad():
            assert(self.actstr == "relu")
            src_shape = x.shape
            x = x.view(src_shape[0],-1) 
            V = {}
            Vsrc = {}
            for i,linop in enumerate(self.linops):
                W = linop.weight
                B = linop.bias

                g = guess[i].unsqueeze(1)
                if gnorm is None:
                    g = g / g.norm(dim=2, keepdim = True)
                else:
                    g = g / gnorm

                if compensateV is None:

                    if ig > 0.0:
                        L = decompose(g,ig)
                        v = torch.randn([L.shape[0],L.shape[1],1], device=device)
                        vn = L @ v
                    else:
                        vn = guess[i].unsqueeze(2)


                    vb = vn.clone().squeeze()

                    if anorm is None:
                        mnorm = torch.sqrt(x.norm(dim=1,keepdim = True)**2+ 1 +  eps) 
                    else:
                        mnorm = anorm


                    z = x/mnorm  ## B X N
                    vw = torch.matmul(vn,z.unsqueeze(1))
                    vb = vb / mnorm
                    Vsrc[i] = (vw.clone(), vb.clone())
                else:
                    vw_c, vb_c = compensateV[i]
                    gg, _ = projections(g)
                    vb = (gg @ vb_c.unsqueeze(2)).squeeze(2)
                    vw = gg @ vw_c


                new_grad = torch.matmul(vw, x.unsqueeze(2)).squeeze() + vb   ## B x N1


                if not old_grad is None:
                    old_grad = torch.matmul(old_grad, linop.weight.permute(1,0)) ## B X N0  mm  N0 X N1 -> B X N1
                else:
                    old_grad = 0

                old_grad = old_grad * delta + new_grad

                if compensateV is None:
                    S1, S2 = batched_sherman_morrison2(g, ig) ## < --- the issue is here
                    vb = vb + (S2 @ vb.unsqueeze(2)).squeeze(2)
                    vw = vw + S2 @ vw

                    # transform = batched_sherman_morrison(guess, i, ig) ## < --- the issue is here
                    # vb = (transform @ vb.unsqueeze(2)).squeeze(2)
                    # vw = transform @ vw

                    # breakpoint()
                    # Sigma = L @ L.permute(0,2,1)
                    # diff = (((S1 + S2) @ Sigma  - torch.eye(Sigma.shape[1], device = device))**2).mean()
                    # print("diff: %.2E, S1.mean(): %.2E, S2.mean(): %.2E" % (diff, S1.mean(), S2.mean()))

                    V[i] = (vw, vb)

                x = linop(x)
                if i < len(self.linops) - 1:
                    mask  =  ((x >= 0).to(torch.float))
                    old_grad = old_grad * mask ## B X N1
                    x = self.act(x)
                
        dLdout = torch.zeros_like(x)
        
        out =torch.autograd.Variable(x, requires_grad = True)
        out.grad = torch.zeros_like(out)
        L = loss(out, y)


        L.backward()
        dLdout = out.grad
        dFg = (dLdout*old_grad).sum(1, keepdim = True)

        for i in range(len(self.linops)):

            linop = self.linops[i]
            if linop.weight.grad is None:
                linop.weight.grad = torch.zeros_like(linop.weight)
                if not linop.bias is None:
                    linop.bias.grad = torch.zeros_like(linop.bias)


            if compensateV is None:
                vw, vb = V[i]
            else:
                vw, vb = compensateV[i]

                g = guess[i].unsqueeze(1)
                if gnorm is None:
                    g = g / g.norm(dim=2, keepdim = True)
                else:
                    g = g / gnorm

                _, I_gg = projections(g)
                factor = -(1-ig**2)/ig**2

                vb = (I_gg @ vb.unsqueeze(2)).squeeze(2) * factor
                vw = (I_gg @ vw ) * factor                



            linop.weight.grad +=  torch.matmul(dFg.permute(1,0), vw.view(vw.shape[0],-1)).view(vw.shape[1],vw.shape[2])   ## 1 x B   mm   Bx(N1xN2)  >   N1 x N2
            if not linop.bias is None:
                if True or resample:
                    linop.bias.grad += (dFg* vb).sum(0) 
                else:
                    linop.bias.grad += dFg.sum() * vb 


        return x, Vsrc, dFg


    def fwd_mode_IG2(self,x, y, loss, guess , binary = False, anorm = None, gnorm = None, parallel = True):

        mage = True
        tot_norm = 0
        epsilon = 1
        eps = 1E-6
        batch_size = x.shape[0]
        old_grad = None

        device =x.device
        dtype = x.dtype

        with torch.no_grad():
            assert(self.actstr == "relu")
            src_shape = x.shape
            x = x.view(src_shape[0],-1) 
            V = {}
            for i,linop in enumerate(self.linops):
                W = linop.weight
                B = linop.bias

                g = guess[i].unsqueeze(1)

                if gnorm is None:
                    g = g / g.norm(dim=2, keepdim = True)
                else:
                    g = g / gnorm


                # vn = torch.randn([W.shape[0],1],device =device, dtype = dtype).expand(W.shape[0], batch_size)
                # vb = vn.clone().squeeze().permute(1,0)

                vn = torch.randn([batch_size, W.shape[0], 1],device =device, dtype = dtype)
                vb = vn.clone().squeeze(2)

                if anorm is None:
                    mnorm = torch.sqrt(x.norm(dim=1,keepdim = True)**2 + 1 +  eps) 
                else:
                    mnorm = anorm

                z = x/mnorm  ## B X N

                vw = torch.matmul(vn,z.unsqueeze(1))
                vb = vb / mnorm

                if parallel:
                    gg, _ = projections(g)
                    vb = (gg @ vb.unsqueeze(2)).squeeze(2)
                    vw = gg @ vw
                else:
                    _, I_gg = projections(g)
                    vb = (I_gg @ vb.unsqueeze(2)).squeeze(2)
                    vw = I_gg @ vw



                V[i] = (vw.clone(), vb.clone())


                new_grad = torch.matmul(vw, x.unsqueeze(2)).squeeze() + vb   ## B x N1


                if not old_grad is None:
                    old_grad = torch.matmul(old_grad, linop.weight.permute(1,0)) ## B X N0  mm  N0 X N1 -> B X N1
                else:
                    old_grad = 0

                old_grad = old_grad + new_grad

                x = linop(x)
                if i < len(self.linops) - 1:
                    mask  =  ((x >= 0).to(torch.float))
                    old_grad = old_grad * mask ## B X N1
                    x = self.act(x)
                
        dLdout = torch.zeros_like(x)
        
        out =torch.autograd.Variable(x, requires_grad = True)
        out.grad = torch.zeros_like(out)
        L = loss(out, y)


        L.backward()
        dLdout = out.grad
        dFg = (dLdout*old_grad).sum(1, keepdim = True)

        for i in range(len(self.linops)):

            linop = self.linops[i]
            if linop.weight.grad is None:
                linop.weight.grad = torch.zeros_like(linop.weight)
                if not linop.bias is None:
                    linop.bias.grad = torch.zeros_like(linop.bias)


            vw, vb = V[i]

            linop.weight.grad +=  torch.matmul(dFg.permute(1,0), vw.view(vw.shape[0],-1)).view(vw.shape[1],vw.shape[2])   ## 1 x B   mm   Bx(N1xN2)  >   N1 x N2
            if not linop.bias is None:
                if True or resample:
                    linop.bias.grad += (dFg* vb).sum(0) 
                else:
                    linop.bias.grad += dFg.sum() * vb 

        return x, dFg

    def pop_guess(self):

        # gnorm = self.get_tot_norms()
        # anorm = self.anorm_tot

        out = self.guess_i
        self.guess_i = []
        out.reverse()
        return out, None, None



    def get_tot_norms(self):
        gnorm_tot = 0
        for g in self.guess_i:
            gnorm_tot = gnorm_tot + (g.unsqueeze(1).norm(dim=2, keepdim = True))**2

        return torch.sqrt(gnorm_tot)

