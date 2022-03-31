from scipy.fftpack import fft2, ifft2, fft, ifft
import torch
import numpy as np
import math


def Noise_map(input1, stdn):
    x = len(input1)
    noise_map = torch.zeros(x, 1, 64, 64)
    for i in range(x):
        sizen_map = noise_map[0, :, :].size()
        noise_map[i, 0, :, :] = torch.FloatTensor(sizen_map).fill_(stdn[i])

    return noise_map


def fft_reshape_tensor(H_shape):
    x = len(H_shape)    # 2048
    y = len(H_shape[0])
    H_reshape = np.zeros([y, 32, 64], dtype=complex)
    H_reshape_T = np.zeros([y, 64, 32], dtype=complex)
    H_reshape_fft = np.zeros([y, 64, 32], dtype=complex)
    H_real = np.zeros([y, 64, 32])
    H_imag = np.zeros([y, 64, 32])
    H_get = np.zeros([y, 64, 64])
    for i in range(y):
        H_reshape[i, :] = np.reshape(H_shape[:, i], (32, 64))
        H_reshape_T[i, :] = H_reshape[i, :].T
        H_reshape_fft[i, :] = fft2(H_reshape_T[i, :])
        H_real[i, :] = H_reshape_fft[i, :].real
        H_imag[i, :] = H_reshape_fft[i, :].imag
        H_get[i, :] = np.hstack((H_real[i, :], H_imag[i, :]))
    return x, y, H_reshape_T, H_reshape_fft, H_get


def zhenghe(H_shape, img_height, img_width):
    Nt = img_width
    Nr = img_height
    x = len(H_shape)   # n
    y = math.floor(x/4)
    H_real = np.zeros([y, Nr*2, Nt*2], dtype=complex)
    # m from 0
    for m in range(y):
        H_real[m, 0:32, 0:16] = H_real[m, 0:32, 0:16] + H_shape[4 * m+0, 0:Nr, 0:Nt]
        H_real[m, 0:32, 16:32] = H_real[m, 0:32, 16:32] + H_shape[4 * m + 1, 0:Nr, 0:Nt]
        H_real[m, 32:64, 0:16] = H_real[m, 32:64, 0:16] + H_shape[4 * m + 2, 0:Nr, 0:Nt]
        H_real[m, 32:64, 16:32] = H_real[m, 32:64, 16:32] + H_shape[4 * m + 3, 0:Nr, 0:Nt]
    return H_real, y


def fft_reshape(H_shape, img_height, img_width):
    x = len(H_shape)    # 2048
    y = len(H_shape[0])    # n
    H_reshape = np.zeros([y, img_width, img_height], dtype=complex)
    H_reshape_T = np.zeros([y, img_height, img_width], dtype=complex)
    H_real = np.zeros([y, img_height, img_width])
    H_imag = np.zeros([y, img_height, img_width])[[]]
    H_get = np.zeros([y, img_height, img_width*2])
    for i in range(y):
        H_reshape[i, :] = np.reshape(H_shape[:, i], (img_width, img_height))
        H_reshape_T[i, :] = H_reshape[i, :].T
        H_real[i, :] = H_reshape_T[i, :].real
        H_imag[i, :] = H_reshape_T[i, :].imag
        H_get[i, :] = np.hstack((H_real[i, :], H_imag[i, :]))
    return x, y, H_reshape_T, H_get


def fft_shrink(H_shape, img_height, img_width):
    x = len(H_shape)    # n
    H_real = np.zeros([x, img_height, img_width], dtype=complex)
    H_imag = np.zeros([x, img_height, img_width], dtype=complex)
    for m in range(x):
        H_real[m, :, :] = H_real[m, :, :] + H_shape[m, :, 0:img_width]
        H_imag[m, :, :] = H_real[m, :, :] + 1j*H_shape[m, :, img_width:img_width*2]
    return H_imag


def shrink_zero(H_shape, img_height, img_width):
    H_real = np.zeros([img_height, img_width], dtype=complex)
    H_imag = np.zeros([img_height, img_width], dtype=complex)
    H_real[:, :] = H_real[:, :] + H_shape[:,  0:img_width]
    H_imag[:, :] = H_real[:, :] + 1j*H_shape[:, img_width:img_width*2]
    return H_imag


def fft_shrink1(H_shape, img_height, img_width):
    x = len(H_shape)        # 32         16
    H = H_shape.cpu()
    ssx = H.detach().numpy()  # numpy
    ssx_i = np.zeros([x, 4, 32, 32], dtype=complex)  # 64 X 64
    H_fft_pre_h = ssx + ssx_i                # 4x32x32  numpy
    H1 = np.zeros([x, 64, 32], dtype=complex)
    for i in range(x):
        H1[i, 0:32, 0:16] = ifft2(shrink_zero(H_fft_pre_h[i, 0, :, :], img_height, img_width))
        H1[i, 32:64, 0:16] = ifft2(shrink_zero(H_fft_pre_h[i, 1, :, :], img_height, img_width))
        H1[i, 0:32, 16:32] = ifft2(shrink_zero(H_fft_pre_h[i, 2, :, :], img_height, img_width))
        H1[i, 32:64, 16:32] = ifft2(shrink_zero(H_fft_pre_h[i, 3, :, :], img_height, img_width))
    return H1


def add_noise(input, SNR):
    y = len(input)
    x_test_realc = np.reshape(input, (len(input), -1))
    power = np.sum(abs(x_test_realc) ** 2, axis=1)
    power_level = np.mean(power)
    power_level_1 = power_level/2048
    SNR_level = 10**(SNR/10)
    noise_level = power_level_1/SNR_level
    n_l = math.sqrt(noise_level)
    Noise_map = np.zeros([y, 64, 32], dtype=complex)
    Noise_map_real = np.zeros([y, 64, 32], dtype=complex)
    Noise_map_imag = np.zeros([y, 64, 32], dtype=complex)
    Noise_map_real = n_l * math.sqrt(1 / 2) * np.random.randn(len(input), 64, 32)
    Noise_map_imag = 1j * n_l * math.sqrt(1 / 2) * np.random.randn(len(input), 64, 32)
    Noise_map = Noise_map_real + Noise_map_imag
    return Noise_map, n_l


def add_noise1(input, SNR):
    x_test_realc = np.reshape(input, (1, -1))
    power = np.sum(abs(x_test_realc) ** 2, axis=1)
    power_level = power
    power_level_1 = power_level/2048
    SNR_level = 10**(SNR/10)
    noise_level = power_level_1/SNR_level
    n_l = math.sqrt(noise_level)
    Noise_map = np.zeros([64, 32], dtype=complex)
    Noise_map_real = np.zeros([64, 32], dtype=complex)
    Noise_map_imag = np.zeros([64, 32], dtype=complex)
    Noise_map_real = n_l * math.sqrt(1 / 2) * np.random.randn(64, 32)
    Noise_map_imag = 1j * n_l * math.sqrt(1 / 2) * np.random.randn(64, 32)
    Noise_map = Noise_map_real + Noise_map_imag
    return Noise_map, n_l


def add_noise_improve(input, SNRlow, SNRhign):
    NN = len(input)
    noise = np.zeros([NN, 64, 32], dtype=complex)
    SNR_divide = np.random.uniform(SNRlow, SNRhign, size=[NN])
    stdn = np.random.uniform(0.01, 0.02, size=[NN])
    for nx in range(NN):
        noise[nx, :, :], stdn[nx] = add_noise1(input[nx, :, :], SNR_divide[nx])
    return noise, stdn


def compute_SNR(H_noise_train, H_noise):
    test_noise_get = np.reshape(H_noise, (len(H_noise), -1))
    x_test_realc = np.reshape(H_noise_train, (len(H_noise_train), -1))
    noise_power = np.sum(abs(test_noise_get) ** 2, axis=1)
    power = np.sum(abs(x_test_realc) ** 2, axis=1)
    SNR = 10 * math.log10(np.mean(power / noise_power))
    return SNR


def compute_NMSE(H, H_pre):
    H1 = np.reshape(H, (len(H), -1))
    H_pre1 = np.reshape(H_pre, (len(H_pre), -1))
    power = np.sum(abs(H_pre1) ** 2, axis=1)
    mse = np.sum(abs(H1 - H_pre1) ** 2, axis=1)
    NMSE = 10 * math.log10(np.mean(mse / power))
    return NMSE


def fft_tensor(H_shape):
    x = len(H_shape)    # n
    H_reshape_fft = np.zeros([x, 64, 32], dtype=complex)
    H_real = np.zeros([x, 64, 32])
    H_imag = np.zeros([x, 64, 32])
    H_get = np.zeros([x, 64, 64])
    for i in range(x):
        H_reshape_fft[i, :, :] = fft2(H_shape[i, :, :])
        H_real[i, :] = H_reshape_fft[i, :].real
        H_imag[i, :] = H_reshape_fft[i, :].imag
        H_get[i, :] = np.hstack((H_real[i, :], H_imag[i, :]))
    return H_reshape_fft, H_get


def nomal_maxmin(X, N):
    XX = X.numpy()
    H = np.maximum(XX, -N)
    Y = np.minimum(H, N)
    out = torch.from_numpy(Y)
    return out


def real_imag_stack(H):
    x = len(H)    # n
    H_real = np.zeros([x, 64, 32])
    H_imag = np.zeros([x, 64, 32])
    H_out = np.zeros([x, 64, 64])
    for i in range(x):
        H_real[i, :] = H[i, :].real
        H_imag[i, :] = H[i, :].imag
        H_out[i, :] = np.hstack((H_real[i, :], H_imag[i, :]))
    return H_out


def nomal_noiselevel(H):
    x = len(H)    # n
    y = np.zeros([x])
    for i in range(x):
        if H[i] > 0:
            y[i] = H[i]
        elif H[i] <= 0:
            y[i] = H[i]

    return y

def tensor_reshape(input1):
    x = len(input1)
    input = torch.from_numpy(input1)
    H = torch.zeros([x, 1, 64, 64])
    for i in range(x):
        H[i, 0, :, :] = input[i, 0:64, 0:64]
    return H


def fft_tensor1632(H_shape):
    x = len(H_shape)    # n

    H_reshape_fft0 = np.zeros([x, 32, 16], dtype=complex)
    H_reshape_fft1 = np.zeros([x, 32, 16], dtype=complex)
    H_reshape_fft2 = np.zeros([x, 32, 16], dtype=complex)
    H_reshape_fft3 = np.zeros([x, 32, 16], dtype=complex)
    H_get = np.zeros([x, 64, 64])
    for i in range(x):
        H_reshape_fft0[i, :] = fft2(H_shape[i, 0:32, 0:16])
        H_reshape_fft1[i, :] = fft2(H_shape[i, 0:32, 16:32])
        H_reshape_fft2[i, :] = fft2(H_shape[i, 32:64, 0:16])
        H_reshape_fft3[i, :] = fft2(H_shape[i, 32:64, 16:32])
        H_get[i, 0:32, 0:32] = np.hstack((H_reshape_fft0[i, :].real, H_reshape_fft0[i, :].imag))
        H_get[i, 0:32, 32:64] = np.hstack((H_reshape_fft1[i, :].real, H_reshape_fft1[i, :].imag))
        H_get[i, 32:64, 0:32] = np.hstack((H_reshape_fft2[i, :].real, H_reshape_fft2[i, :].imag))
        H_get[i, 32:64, 32:64] = np.hstack((H_reshape_fft3[i, :].real, H_reshape_fft3[i, :].imag))
    return H_get


def expand_tensor(H_shape):
    x = len(H_shape)    # n
    H_reshape_fft = np.zeros([x, 64, 32], dtype=complex)
    H_real = np.zeros([x, 64, 32])
    H_imag = np.zeros([x, 64, 32])
    H_get = np.zeros([x, 64, 64])
    for i in range(x):
        H_reshape_fft[i, :, :] = H_shape[i, :, :]
        H_real[i, :] = H_reshape_fft[i, :].real
        H_imag[i, :] = H_reshape_fft[i, :].imag
        H_get[i, :] = np.hstack((H_real[i, :], H_imag[i, :]))
    return H_reshape_fft, H_get


def ifft_tensor(H_shape):
    x = len(H_shape)    # n
    H_reshape_fft = np.zeros([x, 64, 32], dtype=complex)
    H_real = np.zeros([x, 64, 32])
    H_imag = np.zeros([x, 64, 32])
    H_get = np.zeros([x, 64, 64])
    for i in range(x):
        H_reshape_fft[i, :, :] = ifft2(H_shape[i, :, :])
        H_real[i, :] = H_reshape_fft[i, :].real
        H_imag[i, :] = H_reshape_fft[i, :].imag
        H_get[i, :] = np.hstack((H_real[i, :], H_imag[i, :]))
    return H_reshape_fft, H_get


def pick_max_min(input1):
    max_ten1 = np.max(input1)
    min_ten1 = np.min(input1)
    return max_ten1, min_ten1


def nomalization_tensor(input, y):
    # max_ten = torch.max(input)
    # min_ten = torch.min(input)
    Max_abs = y  # max(max_ten-0, 0-min_ten)
    output = 100*input/Max_abs+120
    return output, Max_abs


def noise_nomalization(input, Max_abs):
    output = input
    return output


def subimag_n(input1):
    x = len(input1)
    input = torch.from_numpy(input1)
    H = torch.zeros([x, 4, 32, 32])
    for i in range(x):
        H[i, 0, :, :] = input[i, 0:32, 0:32]
        H[i, 1, :, :] = input[i, 32:64, 0:32]
        H[i, 2, :, :] = input[i, 0:32, 32:64]
        H[i, 3, :, :] = input[i, 32:64, 32:64]
    return H


def to_torch(input1):
    x = len(input1)
    input = torch.from_numpy(input1)
    H = torch.zeros([x, 64, 64])
    H = input
    return H


def polyimag_n(input):
    x = len(input)
    H = torch.zeros([x, 64, 64])
    for i in range(x):
        H[i, 0:32, 0:32] = input[i, 0, :, :]
        H[i, 32:64, 0:32] = input[i, 1, :, :]
        H[i, 0:32, 32:64] = input[i, 2, :, :]
        H[i, 32:64, 32:64] = input[i, 3, :, :]
    return H


def subimag_h(input1, stdn):
    x = len(input1)
    input = torch.from_numpy(input1)
    H = torch.zeros([x, 4, 32, 32])
    noise_map = torch.zeros(x, 1, 32, 32)
    for i in range(x):
        sizen_map = noise_map[0, :, :, :].size()
        H[i, 0, :, :] = input[i, 0:32, 0:32]
        H[i, 1, :, :] = input[i, 32:64, 0:32]
        H[i, 2, :, :] = input[i, 0:32, 32:64]
        H[i, 3, :, :] = input[i, 32:64, 32:64]
        noise_map[i, :, :, :] = torch.FloatTensor(sizen_map).fill_(stdn[i])

    # imag_train = torch.cat((noise_map, M), 1)     noise_map的值是否小或者大
    return noise_map, H


def val(H1, SNR):
    # H_noise_train, n_l = add_noise_improve(H1, 18, 22)
    n_val, n_2 = add_noise_improve(H1, SNR-0.5,  SNR+0.5)  # n  # 64x32
    h_n_val = n_val + H1

    h_n_fft_val, H_fft_val_expand = fft_tensor(h_n_val)  # 64x32
    h_n_fft_val1, H_fft_val_expand1 = fft_tensor(H1)  # 64x32
    n_fft = H_fft_val_expand - H_fft_val_expand1
    NNn = len(h_n_fft_val)
    n_fft_train_sub = subimag_n(n_fft)     # torch

    noise_map2, M2 = subimag_h(H_fft_val_expand, n_2)    # torch
    H_n_fft_val_sub2 = torch.cat((noise_map2, M2), 1)
    return H_n_fft_val_sub2, M2, n_fft_train_sub


def test(H1, SNR):
    BATCH_SIZE = len(H1)
    # H_noise_train, n_l = add_noise_improve(H1, 18, 22)
    n_val, n_2 = add_noise_improve(H1, SNR-0.05,  SNR+0.05)  # n  # 64x32
    h_n_val = n_val + H1
    n_level_02 = torch.from_numpy(n_2)

    H_reshape_fft = np.zeros([BATCH_SIZE, 64, 32], dtype=complex)
    for i in range(BATCH_SIZE):
        H_reshape_fft[i, :, :] = fft2(h_n_val[i, :, :])

    H_n_fft_stack1 = real_imag_stack(H_reshape_fft)
    H_n_fft_stack = tensor_reshape(H_n_fft_stack1)

    return H_n_fft_stack, n_level_02


def test_E(H_n):
    BATCH_SIZE = len(H_n)
    E_CNN_input1 = nomal_maxmin(H_n, 5)
    # H_noise_train, n_l = add_noise_improve(H1, 18, 22)
    n_val, n_2 = add_noise_improve(H1, SNR-0.05,  SNR+0.05)  # n  # 64x32
    h_n_val = n_val + H1
    n_level_02 = torch.from_numpy(n_2)

    H_reshape_fft = np.zeros([BATCH_SIZE, 64, 32], dtype=complex)
    for i in range(BATCH_SIZE):
        H_reshape_fft[i, :, :] = fft2(h_n_val[i, :, :])

    H_n_fft_stack1 = real_imag_stack(H_reshape_fft)
    H_n_fft_stack = tensor_reshape(H_n_fft_stack1)

    return H_n_fft_stack, n_level_02


def val_next(h_fft_pre_val, output2):
    NNn = len(h_fft_pre_val)
    h_po_fft1 = polyimag_n(h_fft_pre_val)
    n_pre_test1 = polyimag_n(output2)
    ssx1 = h_po_fft1.detach().numpy()  # numpy
    n_pre11 = n_pre_test1.detach().numpy()
    ssx_i1 = np.zeros([NNn, 64, 64], dtype=complex)  # 64 X 64
    H_fft_pre_h1 = ssx1 + ssx_i1

    # 生成复数
    H_fft_pre_last_val = fft_shrink(H_fft_pre_h1, 64, 32)
    H_fft_pre_last3, H_fft_pre_last4 = ifft_tensor(H_fft_pre_last_val)
    return H_fft_pre_last3
