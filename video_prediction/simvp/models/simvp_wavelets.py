import time
import pywt
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable, gradcheck
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        pw = w_ll.shape[3]/2 - 1
        ph = w_ll.shape[2]/2 - 1
        p = [int(ph), int(pw)]
        
        dim = x.shape[1]
        B, C, H, W = x.shape
        #print(x.shape)
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), padding = p, stride = 1, groups = dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), padding = p, stride = 1, groups = dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), padding = p, stride = 1, groups = dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), padding = p, stride = 1, groups = dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

        return x_ll, x_lh, x_hl, x_hh

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H//2, W//2)

            dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None

class DWT_Function_FFT(Function):
    @staticmethod
    def forward(ctx, x, w_l, w_h):
        x = x.contiguous()
        #ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        #ctx.shape = x.shape

        xx = x
        B, C, H, W = x.shape
        w_ls = w_l.shape[1];
        w_hs = w_h.shape[1];
        #print(f'1. x: {x.shape}')
        #print(f'2. w_l: {w_ls}')
        #print(w_l)
        #print(f'3. w_h: {w_hs}')
        #print(w_h)

        H2n = int(2 ** torch.ceil(torch.log2(torch.tensor(H))))
        W2n = int(2 ** torch.ceil(torch.log2(torch.tensor(W))))
        #print(f'4. H2n: {H2n}, W2n: {W2n}')

        if H2n > W2n:
            W2n = H2n
        else:
            H2n = W2n

        n_padding_w = W2n-W
        n_padding_h = H2n-H
        padding = (0, n_padding_w, 0, n_padding_h, 0, 0, 0, 0)
        x = torch.nn.functional.pad(x, padding)

        padding = (0, H2n-w_ls)
        w_l = torch.nn.functional.pad(w_l, padding)
        w_l = w_l.repeat(B,C,W2n,1)

        padding = (0, W2n-w_hs)
        w_h = torch.nn.functional.pad(w_h, padding)
        w_h = w_h.repeat(B,C,H2n,1)
        #print(f'5. x: {x.shape}')
        #print(x[0][0][220][:])
        #print(f'6. w_l: {w_l.shape}')
        #print(w_l[0][1][1][:])
        #print(f'7. w_h: {w_h.shape}')
        #print(w_h[0][1][1][:])

        x_fft = torch.fft.fft(x, dim = 3)
        w_l_fft = torch.fft.fft(w_l, dim = 3)
        w_h_fft = torch.fft.fft(w_h, dim = 3)
        x_l_fft = x_fft * w_l_fft
        x_h_fft = x_fft * w_h_fft
        x_l = torch.fft.ifft(x_l_fft, dim = 3)
        x_h = torch.fft.ifft(x_h_fft, dim = 3)

        x_l = x_l.real
        x_h = x_h.real
        
        shift = int(w_ls/2)
        x_l_s = torch.roll(x_l, shifts=-shift, dims=-1)
        shift = int(w_hs/2)
        x_h_s = torch.roll(x_h, shifts=-shift, dims=-1)
        x_l_s_t = x_l_s.transpose(-2, -1)
        x_h_s_t = x_h_s.transpose(-2, -1)

        x_l_fft = torch.fft.fft(x_l_s_t, dim = 3)
        x_h_fft = torch.fft.fft(x_h_s_t, dim = 3)

        x_ll_fft = x_l_fft * w_l_fft
        x_lh_fft = x_l_fft * w_h_fft
        x_hl_fft = x_h_fft * w_l_fft
        x_hh_fft = x_h_fft * w_h_fft

        x_ll = torch.fft.ifft(x_ll_fft, dim = 3)
        x_lh = torch.fft.ifft(x_lh_fft, dim = 3)
        x_hl = torch.fft.ifft(x_hl_fft, dim = 3)
        x_hh = torch.fft.ifft(x_hh_fft, dim = 3)

        x_ll = x_ll.real
        x_lh = x_lh.real
        x_hl = x_hl.real
        x_hh = x_hh.real

        shift = int(w_ls/2)
        x_ll_s = torch.roll(x_ll, shifts=-shift, dims=-1)
        x_lh_s = torch.roll(x_lh, shifts=-shift, dims=-1)
        shift = int(w_hs/2)
        x_hl_s = torch.roll(x_hl, shifts=-shift, dims=-1)
        x_hh_s = torch.roll(x_hh, shifts=-shift, dims=-1)

        x_ll_t = x_ll_s.transpose(-2, -1)
        x_lh_t = x_lh_s.transpose(-2, -1)
        x_hl_t = x_hl_s.transpose(-2, -1)
        x_hh_t = x_hh_s.transpose(-2, -1)

        x_ll = x_ll_t[..., ::2, ::2]
        x_lh = x_lh_t[..., ::2, ::2]
        x_hl = x_hl_t[..., ::2, ::2]
        x_hh = x_hh_t[..., ::2, ::2]

        n_padding_h = int(n_padding_h/2)
        n_padding_w = int(n_padding_w/2)

        if H2n-W == 0 and H2n-H == 0:
            x_ll = x_ll
            x_lh = x_lh
            x_hl = x_hl
            x_hh = x_hh
        else:
            x_ll = x_ll[..., :-n_padding_h, :-n_padding_w]
            x_lh = x_lh[..., :-n_padding_h, :-n_padding_w]
            x_hl = x_hl[..., :-n_padding_h, :-n_padding_w]
            x_hh = x_hh[..., :-n_padding_h, :-n_padding_w]

        # Wavelet transform of image, and plot approximation and details
        if x.shape[0] == 1000:
            titles = ['Approximation', ' Horizontal detail', 'Vertical detail', 'Diagonal detail']
            x_ll = x_ll.permute(0, 2, 3, 1)
            x_lh = x_lh.permute(0, 2, 3, 1)
            x_hl = x_hl.permute(0, 2, 3, 1)
            x_hh = x_hh.permute(0, 2, 3, 1)
            xx = xx.permute(0, 2, 3, 1)

            n_ll = torch.abs(x_ll[10])
            n_lh = torch.abs(x_lh[10])
            n_hl = torch.abs(x_hl[10])
            n_hh = torch.abs(x_hh[10])
            x_hh = xx[10]
            
            n_ll_max = torch.max(n_ll)
            n_lh_max = torch.max(n_lh)
            n_hl_max = torch.max(n_hl)
            n_hh_max = torch.max(n_hh)

            print(f'1. {n_ll_max}')

            n_ll = n_ll / n_ll_max
            n_lh = n_lh / n_lh_max
            n_hl = n_hl / n_hl_max
            n_hh = n_hh / n_hh_max

            print(f'2. {torch.max(n_ll)}')
            print(f'2. {torch.min(n_ll)}')

            LL = n_ll.float().cpu().numpy()
            LH = n_lh.float().cpu().numpy()
            HL = n_hl.float().cpu().numpy()
            HH = n_hh.float().cpu().numpy()
            NN = x_hh.float().cpu().numpy()
            
            plt.imshow(NN)
            plt.show()

            fig = plt.figure(figsize=(12, 3))
            for i, a in enumerate([LL, LH, HL, HH]):
                ax = fig.add_subplot(1, 4, i + 1)
                ax.imshow(a, interpolation="nearest")
                ax.set_title(titles[i], fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

            fig.tight_layout()
            plt.show()
            exit()
    
        return x_ll, x_lh, x_hl, x_hh

    #@staticmethod
    #def backward(ctx, dx):
        #if ctx.needs_input_grad[0]:
            #w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            #B, C, H, W = ctx.shape
            #dx = dx.view(B, 4, -1, H//2, W//2)

            #dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2)
            #filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            #dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        #return dx, None, None, None, None

class DWT_Function_FFT_L2(Function):
    @staticmethod
    def forward(ctx, x, w_l, w_h):
        #x = x.contiguous()
        #ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        #ctx.shape = x.shape

        xx = x
        B, C, H, W = x.shape
        w_ls = w_l.shape[1];
        w_hs = w_h.shape[1];
        #print(f'1. x: {x.shape}')
        #print(f'2. w_l: {w_ls}')
        #print(w_l)
        #print(f'3. w_h: {w_hs}')
        #print(w_h)

        H2n = int(2 ** torch.ceil(torch.log2(torch.tensor(H))))
        W2n = int(2 ** torch.ceil(torch.log2(torch.tensor(W))))
        #print(f'4. H2n: {H2n}, W2n: {W2n}')

        if H2n > W2n:
            W2n = H2n
        else:
            H2n = W2n

        n_padding_w = W2n-W
        n_padding_h = H2n-H
        padding = (0, n_padding_w, 0, n_padding_h, 0, 0, 0, 0)
        x = torch.nn.functional.pad(x, padding)
        
        padding = (0, H2n-w_ls)
        w_l = torch.nn.functional.pad(w_l, padding)
        w_l = w_l.repeat(B,C,W2n,1)

        padding = (0, W2n-w_hs)
        w_h = torch.nn.functional.pad(w_h, padding)
        w_h = w_h.repeat(B,C,H2n,1)
        #print(f'5. x: {x.shape}')
        #print(x[0][0][220][:])
        #print(f'6. w_l: {w_l.shape}')
        #print(w_l[0][1][1][:])
        #print(f'7. w_h: {w_h.shape}')
        #print(w_h[0][1][1][:])

        x_fft = torch.fft.fft(x, dim = 3)
        w_l_fft = torch.fft.fft(w_l, dim = 3)
        w_h_fft = torch.fft.fft(w_h, dim = 3)
        x_l_fft = x_fft * w_l_fft
        x_h_fft = x_fft * w_h_fft
        x_l = torch.fft.ifft(x_l_fft, dim = 3)
        x_h = torch.fft.ifft(x_h_fft, dim = 3)

        x_l = x_l.real
        x_h = x_h.real
        
        shift = int(w_ls/2)
        x_l_s = torch.roll(x_l, shifts=-shift, dims=-1)
        shift = int(w_hs/2)
        x_h_s = torch.roll(x_h, shifts=-shift, dims=-1)
        x_l_s_t = x_l_s.transpose(-2, -1)
        x_h_s_t = x_h_s.transpose(-2, -1)

        x_l_fft = torch.fft.fft(x_l_s_t, dim = 3)
        x_h_fft = torch.fft.fft(x_h_s_t, dim = 3)

        x_ll_fft = x_l_fft * w_l_fft
        x_lh_fft = x_l_fft * w_h_fft
        x_hl_fft = x_h_fft * w_l_fft
        x_hh_fft = x_h_fft * w_h_fft

        x_ll = torch.fft.ifft(x_ll_fft, dim = 3)
        x_lh = torch.fft.ifft(x_lh_fft, dim = 3)
        x_hl = torch.fft.ifft(x_hl_fft, dim = 3)
        x_hh = torch.fft.ifft(x_hh_fft, dim = 3)

        x_ll = x_ll.real
        x_lh = x_lh.real
        x_hl = x_hl.real
        x_hh = x_hh.real

        shift = int(w_ls/2)
        x_ll_s = torch.roll(x_ll, shifts=-shift, dims=-1)
        x_lh_s = torch.roll(x_lh, shifts=-shift, dims=-1)
        shift = int(w_hs/2)
        x_hl_s = torch.roll(x_hl, shifts=-shift, dims=-1)
        x_hh_s = torch.roll(x_hh, shifts=-shift, dims=-1)

        x_ll_t = x_ll_s.transpose(-2, -1)
        x_lh_t = x_lh_s.transpose(-2, -1)
        x_hl_t = x_hl_s.transpose(-2, -1)
        x_hh_t = x_hh_s.transpose(-2, -1)

        #print(f'x_ll_t.shape: {x_ll_t.shape}')
        x_ll = x_ll_t[..., ::1, ::1]
        x_lh = x_lh_t[..., ::1, ::1]
        x_hl = x_hl_t[..., ::1, ::1]
        x_hh = x_hh_t[..., ::1, ::1]
        #print(f'x_ll.shape: {x_ll.shape}')

        n_padding_h = int(n_padding_h)
        n_padding_w = int(n_padding_w)

        if H2n-W == 0 and H2n-H == 0:
            x_ll = x_ll
            x_lh = x_lh
            x_hl = x_hl
            x_hh = x_hh
        else:
            x_ll = x_ll[..., :-n_padding_h, :-n_padding_w]
            x_lh = x_lh[..., :-n_padding_h, :-n_padding_w]
            x_hl = x_hl[..., :-n_padding_h, :-n_padding_w]
            x_hh = x_hh[..., :-n_padding_h, :-n_padding_w]

        # Wavelet transform of image, and plot approximation and details
        if x.shape[0] == 1000:
            titles = ['Approximation', ' Horizontal detail', 'Vertical detail', 'Diagonal detail']
            x_ll = x_ll.permute(0, 2, 3, 1)
            x_lh = x_lh.permute(0, 2, 3, 1)
            x_hl = x_hl.permute(0, 2, 3, 1)
            x_hh = x_hh.permute(0, 2, 3, 1)
            xx = xx.permute(0, 2, 3, 1)

            n_ll = torch.abs(x_ll[10])
            n_lh = torch.abs(x_lh[10])
            n_hl = torch.abs(x_hl[10])
            n_hh = torch.abs(x_hh[10])
            x_hh = xx[10]
            
            n_ll_max = torch.max(n_ll)
            n_lh_max = torch.max(n_lh)
            n_hl_max = torch.max(n_hl)
            n_hh_max = torch.max(n_hh)

            print(f'1. {n_ll_max}')

            n_ll = n_ll / n_ll_max
            n_lh = n_lh / n_lh_max
            n_hl = n_hl / n_hl_max
            n_hh = n_hh / n_hh_max

            print(f'2. {torch.max(n_ll)}')
            print(f'2. {torch.min(n_ll)}')

            LL = n_ll.float().cpu().numpy()
            LH = n_lh.float().cpu().numpy()
            HL = n_hl.float().cpu().numpy()
            HH = n_hh.float().cpu().numpy()
            NN = x_hh.float().cpu().numpy()
            
            plt.imshow(NN)
            plt.show()

            fig = plt.figure(figsize=(12, 3))
            for i, a in enumerate([LL, LH, HL, HH]):
                ax = fig.add_subplot(1, 4, i + 1)
                ax.imshow(a, interpolation="nearest")
                ax.set_title(titles[i], fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

            fig.tight_layout()
            plt.show()
            exit()
    
        return x_ll, x_lh, x_hl, x_hh

    #@staticmethod
    #def backward(ctx, dx):
        #if ctx.needs_input_grad[0]:
            #w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            #B, C, H, W = ctx.shape
            #dx = dx.view(B, 4, -1, H//2, W//2)

            #dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2)
            #filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            #dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        #return dx, None, None, None, None
class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

class DWT_2D_FFT(nn.Module):
    def __init__(self, wave):
        super(DWT_2D_FFT, self).__init__()
        w = pywt.Wavelet(wave)
        #print(wave)
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        dec_hi = torch.Tensor(w.dec_hi[::-1]) 

        w_l = dec_lo.unsqueeze(0)
        w_h = dec_hi.unsqueeze(0)
        #print(w_l.shape)
        self.register_buffer('w_l', w_l)
        self.register_buffer('w_h', w_h)

        #print(w_l.shape)

        self.w_l = self.w_l.to(dtype=torch.float32)
        self.w_h = self.w_h.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function_FFT.apply(x, self.w_l, self.w_h)
        

class DWT_2D_FFT_L2(nn.Module):
    def __init__(self, wave):
        super(DWT_2D_FFT_L2, self).__init__()
        w = pywt.Wavelet(wave)
        #print(wave)
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        dec_hi = torch.Tensor(w.dec_hi[::-1]) 

        w_l = dec_lo.unsqueeze(0)
        w_h = dec_hi.unsqueeze(0)
        #print(w_l.shape)
        self.register_buffer('w_l', w_l)
        self.register_buffer('w_h', w_h)

        #print(w_l.shape)

        self.w_l = self.w_l.to(dtype=torch.float32)
        self.w_h = self.w_h.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function_FFT_L2.apply(x, self.w_l, self.w_h)
'''
def test_time(x, dwt1, dwt2):
    loop = 1000
    total_time1 = 0
    total_time2 = 0

    for i in range(loop):
        start = time.time()
        y1 = dwt1(x)
        torch.cuda.synchronize()
        end = time.time()
        total_time1 += end - start
    
    for i in range(loop):
        start = time.time()
        y2_ll, YH = dwt2(x)
        torch.cuda.synchronize()
        end = time.time()
        total_time2 += end - start

    print(total_time1)
    print(total_time2)

def test_diff(x, dwt1, dwt2):
    y1 = dwt1(x)
    B, C, H, W = y1.shape
    y1 = y1.view(B, 4, -1, H, W)
    y1_ll = y1[:, 0] 
    y1_lh = y1[:, 1]
    y1_hl = y1[:, 2]
    y1_hh = y1[:, 3]
    y2_ll, YH = dwt2(x)
    y2_lh = YH[0][:,:,0]
    y2_hl = YH[0][:,:,1]
    y2_hh = YH[0][:,:,2]
    diff1 = (y1_ll - y2_ll).max()
    diff2 = (y1_lh - y2_lh).max()
    diff3 = (y1_hl - y2_hl).max()
    diff4 = (y1_hh - y2_hh).max()
    print(diff1)
    print(diff2)
    print(diff3)
    print(diff4)

def test_idfiff(x, idwt1, idwt2):
    y1 = idwt1(x)

    x = x.view(x.size(0), 4, -1, x.size(-2), x.size(-1))
    y2 = idwt2((x[:, 0], [x[:,1:].transpose(1, 2)]))
    diff = (y1-y2).max()
    print(diff)

def test_itime(x, idwt1, idwt2):
    loop = 1000
    total_time1 = 0
    total_time2 = 0

    for i in range(loop):
        start = time.time()
        y1 = idwt1(x)
        torch.cuda.synchronize()
        end = time.time()
        total_time1 += end - start
    
    for i in range(loop):
        start = time.time()
        x = x.view(x.size(0), 4, -1, x.size(-2), x.size(-1))
        y2 = idwt2((x[:, 0], [x[:,1:].transpose(1, 2)]))
        torch.cuda.synchronize()
        end = time.time()
        total_time2 += end - start

    print(total_time1)
    print(total_time2)

if __name__ == '__main__':
    #size = (96, 32, 56, 56)
    #size = (96, 64, 28, 28)
    size = (96, 160, 14, 14)
    x = torch.randn(size).cuda().to(dtype=torch.float32)
    dwt1 = DWT_2D('haar').cuda()
    dwt2 = DWTForward(wave='haar').cuda()
    test_diff(x, dwt1, dwt2)
    test_time(x, dwt1, dwt2)

    #size = (96, 32*4, 28, 28)
    #size = (96, 64*4, 14, 14)
    #size = (96, 160*4, 7, 7)
    #x = torch.randn(size).cuda().to(dtype=torch.float16)
    #idwt1 = IDWT_2D('haar').cuda()
    #idwt2 = DWTInverse(wave='haar').cuda()
    #test_idfiff(x, idwt1, idwt2)
    #test_itime(x, idwt1, idwt2)

def test_dwt_grad():
    size = (4, 8, 14, 14)
    x = torch.randn(size).double()

    w = pywt.Wavelet('haar')
    dec_hi = torch.Tensor(w.dec_hi[::-1]) 
    dec_lo = torch.Tensor(w.dec_lo[::-1])

    w_ll = (dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()
    w_lh = (dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()
    w_hl = (dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()
    w_hh = (dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()

    input = (
        Variable(x, requires_grad=True),
        Variable(w_ll, requires_grad=False),
        Variable(w_lh, requires_grad=False),
        Variable(w_hl, requires_grad=False),
        Variable(w_hh, requires_grad=False),
    )
    test = gradcheck(DWT_Function.apply, input)
    print("test:", test)

def test_idwt_grad():
    size = (4, 2*8, 7, 7)
    x = torch.randn(size).double()

    w = pywt.Wavelet('haar')
    rec_hi = torch.Tensor(w.rec_hi)
    rec_lo = torch.Tensor(w.rec_lo)
        
    w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
    w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
    w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
    w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

    w_ll = w_ll.unsqueeze(0).unsqueeze(1)
    w_lh = w_lh.unsqueeze(0).unsqueeze(1)
    w_hl = w_hl.unsqueeze(0).unsqueeze(1)
    w_hh = w_hh.unsqueeze(0).unsqueeze(1)
    filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).double()

    input = (
        Variable(x, requires_grad=True),
        Variable(filters, requires_grad=False),
    )
    test = gradcheck(IDWT_Function.apply, input)
    print("test:", test)

if __name__ == "__main__":
    test_dwt_grad()
'''
