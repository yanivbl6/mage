'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn

import math
from collections import OrderedDict
import torch.nn.functional as F

import numpy as np
def str2act(txt, param= None):
    return {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU(), "none": nn.Sequential() , "lrelu": nn.LeakyReLU(param), "selu": nn.SELU() }[txt.lower()]



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
        

        src_shape = x.shape
        x = x.view(src_shape[0],-1) 

        for i,linop in enumerate(self.linops):
            x = linop(x)

            if i < len(self.linops) - 1:
                x = self.act(x)

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



    def fwd_mode(self,x, y, loss , mage = False, epsilon=1e-5, per_batch = False, normalize_v = False, replicates = 0, resample = True, sparsity = 0.0):

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
                            z = x/(x.norm(dim=1,keepdim = True)+ eps)  ## B X N
                            vw = torch.matmul(vb.unsqueeze(2),z.unsqueeze(1))   ##   B X M X 1    mm    B X 1 X N
                        else:
                            vw = torch.matmul(vb.unsqueeze(2),x.unsqueeze(1))


                    else:
                        vn = torch.randn([W.shape[0],1],device =device, dtype = dtype) *epsilon 
                        if sparsity > 0.0:
                            vnmask = (torch.rand(vn.shape,device =device, dtype = dtype) >= sparsity).to(torch.float)
                            vn = vn*vnmask

                        vb = vn.clone().squeeze().expand(B.shape)

                        if normalize_v:
                            z = x/(x.norm(dim=1,keepdim = True)+ eps)
                            vw = torch.matmul(vn,z.unsqueeze(1))
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
                    ##import pdb; pdb.set_trace()
                    mask  =  ((x >= 0).to(torch.float))
                    old_grad = old_grad * mask ## B X N1
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
        for i in range(len(self.linops)):

            linop = self.linops[i]
            vw, vb = V[i]
            K = len(self.linops) -1 - i
            if not per_batch and mage:
                linop.weight.grad = torch.matmul(dFg.permute(1,0), vw.view(vw.shape[0],-1)).view(vw.shape[1],vw.shape[2]) * (delta**K)   ## 1 x B   mm   Bx(N1xN2)  >   N1 x N2
                if not linop.bias is None:
                    if resample:
                        linop.bias.grad = (dFg* vb).sum(0) * (delta**K)
                    else:
                        linop.bias.grad = dFg.sum() * vb * (delta**K)

            else:
                linop.weight.grad = dFg* vw * (delta**K)
                if not linop.bias is None:
                    linop.bias.grad = dFg * vb  * (delta**K)


        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.3)

        return x
