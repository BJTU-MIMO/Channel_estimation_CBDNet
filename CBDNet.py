# 普通D_cnn的训练，目前不需要
import scipy.io as sio
import numpy as np
import torch.nn as nn
import torch
import torchvision
import torch.utils.data as data
from torch.autograd import Variable
import os

import math
from scipy.fftpack import fft2, ifft2, fft, ifft
from utilss import svd_orthogonalization, weights_init_kaiming
import math
from models import CBDNet
from models import ENet
from tensorboardX import SummaryWriter
from function_forme import compute_SNR, fft_reshape, fft_shrink, add_noise_improve, real_imag_stack, Noise_map, \
    tensor_reshape, ifft_tensor, compute_NMSE, test, nomal_noiselevel


os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDE_VISIBLE_DIVICES"] = "0"

writer3 = SummaryWriter(comment='CBDNet')

EPOCH = 10
BATCH_SIZE = 80
TEST_SIZE = 20
LR = 0.001
focus = 0.1
img_height = 64
img_width = 32
img_channels = 1
Max_abs = 2500

mat = sio.loadmat('data/train.mat')
x_train = mat['H_ori']

x, y, H, H_get = fft_reshape(x_train, img_height, img_width)
del H
# print(H_get)
print(np.shape(H_get))
train_loader = data.DataLoader(dataset=H_get, batch_size=BATCH_SIZE, shuffle=True)
# del H_get
net = CBDNet(num_input_channels=1)
net.apply(weights_init_kaiming)

Loss = nn.MSELoss()
# 损失记录

device_ids = [0]
model = nn.DataParallel(net, device_ids=device_ids).cuda()
Loss.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
Step = 0

NMSE_list = []
# model_fn = 'model/E_cnn_10_30_linear_10out.pth'
# model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_fn)
# E_net = ENet(num_input_channels=1)
# state_dict1 = torch.load(model_fn)
# del model_fn
# model_Ecnn = nn.DataParallel(E_net, device_ids=device_ids).cuda()
# model_Ecnn.load_state_dict(state_dict1['state_dict'])


for epoch in range(EPOCH):
    tr_loss = 0.0
    running_loss = 0.0

    for i, x in enumerate(train_loader, 0):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        NN = len(x)
        sx = x.numpy()

        H_train_h = np.zeros([NN, img_height, img_width * 2], dtype=complex)
        H_train_h = H_train_h + sx
        del sx
        # 64x64->64x32  numpy
        H = fft_shrink(H_train_h, img_height, img_width)
        del H_train_h
        noise, E_output1 = add_noise_improve(H, 20, 20.1)
        H_n = noise + H
        SNR = compute_SNR(H, noise)
        del noise
        n_level = nomal_noiselevel(E_output1)

        # fft2
        H_n_reshape_fft = np.zeros([BATCH_SIZE, 64, 32], dtype=complex)
        H_reshape_fft = np.zeros([BATCH_SIZE, 64, 32], dtype=complex)
        for i_num in range(BATCH_SIZE):
            H_n_reshape_fft[i_num, :, :] = fft2(H_n[i_num, :, :])
            H_reshape_fft[i_num, :, :] = fft2(H[i_num, :, :])

        # 64x32->64x64 numpy real+imag
        H_n_fft_stack1 = real_imag_stack(H_n_reshape_fft)
        noise_stack1 = H_n_fft_stack1-real_imag_stack(H_reshape_fft)
        del H_n_reshape_fft
        del H_reshape_fft

        # input 64x64 torch
        H_n_fft_stack = tensor_reshape(H_n_fft_stack1)
        noise_stack = tensor_reshape(noise_stack1)
        del H_n_fft_stack1
        del noise_stack1

        n_level_01 = torch.from_numpy(n_level)
        H_n_fft_train = Variable(H_n_fft_stack.cuda())
        noise_train = Variable(noise_stack.cuda())

        del noise_stack

        for i_num in range(BATCH_SIZE):
            noise_train[i_num, :, :] = noise_train[i_num, :, :]/(100*n_level[i_num])

        # noise_level = model_Ecnn(H_n_fft_train)
        # H_n_fft_Noisemap = torch.cat((noise_map, 100 * H_n_fft_train / Max_abs + 120), 1)
        H_n_fft_Noisemap = 100 * H_n_fft_train / Max_abs + 120
        # del noise_map

        output = model(H_n_fft_Noisemap)
        del H_n_fft_Noisemap

        loss = Loss(output, noise_train/focus)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        h_fft_pre = torch.zeros([BATCH_SIZE, 64, 64])
        for i_num in range(BATCH_SIZE):
            h_fft_pre[i_num, :, :] = H_n_fft_train[i_num, :, :] - output[i_num, :, :] * focus * (100*n_level_01[i_num])

        ssx = h_fft_pre.detach().numpy()
        ssx_i = np.zeros([NN, 64, 64], dtype=complex)  # 64 X 64
        ssx_i = ssx + ssx_i
        H_fft_pre_last = fft_shrink(ssx_i, img_height, img_width)
        H_fft_pre_last1, H_fft_pre_last2 = ifft_tensor(H_fft_pre_last)

        NMSE = compute_NMSE(H, H_fft_pre_last1)
        writer3.add_scalar('NMSE', NMSE, global_step=i)
        if i % 20 == 19:
            tr_loss = running_loss / 20

        if Step % 2 == 0:
            model.apply(svd_orthogonalization)
            # print("[epoch %d][%d/%d] loss: %.4f SNR: %.4f NMSE: %.4f" %  (epoch + 1, i + 1, len(train_loader), loss, SNR, NMSE))
        Step += 1

    NNN = H_get[0:TEST_SIZE, :, :]
    NNNN = fft_shrink(NNN, img_height, img_width)
    H_n_test, n_level_torch = test(NNNN, 20)
    H_n_fft_test = Variable(H_n_test.cuda())
    H_n_fft_Noisemap_test = 100 * H_n_fft_test / Max_abs + 120
    output = model(H_n_fft_Noisemap_test)
    h_fft_pre_test = torch.zeros([TEST_SIZE, 64, 64])
    for i_num in range(TEST_SIZE):
        h_fft_pre_test[i_num, :, :] = H_n_fft_test[i_num, :, :] - output[i_num, :, :] * focus * (100 * n_level_torch[i_num])

    ssx1 = h_fft_pre_test.detach().numpy()
    ssx_i1 = np.zeros([TEST_SIZE, 64, 64], dtype=complex)  # 64 X 64
    ssx_i1 = ssx1 + ssx_i1

    H_fft_pre_lasttest = fft_shrink(ssx_i1, img_height, img_width)
    H_fft_pre_last1test, H_fft_pre_last2test = ifft_tensor(H_fft_pre_lasttest)

    NMSE_test = compute_NMSE(NNNN, H_fft_pre_last1test)
    # print("20dB NMSE: ", NMSE_test)

    print(NMSE_test, ',')

    save_dict = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(save_dict, os.path.join("lilun_cbd_layers", 'nmsepr.pth'))
    if epoch % EPOCH == 0:
        torch.save(save_dict, os.path.join("lilun_cbd_layers", 'kep1.pth'.format(epoch + 1)))

    del save_dict

writer3.close()