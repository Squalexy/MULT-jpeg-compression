from ntpath import join
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import os
import menu
import scipy.fftpack as fft
import cv2
import math
from scipy import stats
from sklearn.metrics import mean_squared_error

# Ex 3.1


def read_image(img_path):
    return plt.imread(img_path)


# Ex 3.2
def colormap_function(colormap_name, color1, color2):
    return clr.LinearSegmentedColormap.from_list(colormap_name, [color1, color2], 256)


# Ex 3.3
def draw_plot(text, image, colormap=None):
    plt.figure()
    if colormap is not None:
        plt.title(text)
        plt.imshow(image, cmap=colormap)
    else:
        plt.title(text)
        plt.imshow(image)


# Ex 3.4
def rgb_components(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return R, G, B


# if T is a matrix -> Ti = np.linalg.inv(T) to get inversed matrix
def join_RGB(R, G, B):
    matrix_inverted = np.zeros((len(R), len(R[0]), 3), dtype=np.uint8)

    matrix_inverted[:, :, 0] = R
    matrix_inverted[:, :, 1] = G
    matrix_inverted[:, :, 2] = B
    return matrix_inverted


# Ex 3.5
def show_rgb(channel_R, channel_G, channel_B):
    K = (0, 0, 0)
    R = (1, 0, 0)
    G = (0, 1, 0)
    B = (0, 0, 1)

    cm_red = colormap_function("Reds", K, R)
    cm_green = colormap_function("Greens", K, G)
    cm_blue = colormap_function("Blues", K, B)

    fig = plt.figure()

    fig.add_subplot(1, 3, 1)
    plt.title("Channel R with padding")
    plt.imshow(channel_R, cmap=cm_red)

    fig.add_subplot(1, 3, 2)
    plt.title("Channel G with padding")
    plt.imshow(channel_G, cmap=cm_green)

    fig.add_subplot(1, 3, 3)
    plt.title("Channel B with padding")
    plt.imshow(channel_B, cmap=cm_blue)

    plt.subplots_adjust(wspace=0.5)


# Ex 4
def padding_function(img, lines, columns):
    original_img = img
    num_lines_to_add = 0
    num_columns_to_add = 0

    if columns % 16 != 0:
        num_columns_to_add = (16 - (columns % 16))
        array = img[:, -1:]

        aux_2 = np.repeat(array, num_columns_to_add, axis=1)
        img = np.append(img, aux_2, axis=1)

    if lines % 16 != 0:
        num_lines_to_add = (16 - (lines % 16))
        array = img[-1:]
        aux_2 = np.repeat(array, num_lines_to_add, axis=0)
        img = np.append(img, aux_2, axis=0)

    # Plotting
    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_img)

    fig.add_subplot(1, 2, 2)
    plt.title("Image with padding")
    plt.imshow(img)

    return img


# passar logo n linhas e colunas
def without_padding_function(img_with_padding, lines, columns):
    img_recovered = img_with_padding[:lines, :columns, :]
    return img_recovered


# Ex 5.1
def rgb_to_ycbcr(R, G, B):
    Y = (0.299 * R) + (0.587 * G) + (0.114 * B)
    Cb = (-0.168736 * R) + (-0.331264 * G) + (0.5 * B) + 128
    Cr = (0.5 * R) + (-0.418688 * G) + (-0.081312 * B) + 128

    return Y, Cb, Cr


def ycbcr_to_rgb(Y, Cb, Cr):
    matrix = np.array([[0.299,   0.587,    0.114],
                       [-0.168736, -0.331264,     0.5],
                       [0.5, -0.418688, -0.081312]])
    T_matrix = np.linalg.inv(matrix)

    R = np.round(T_matrix[0][0] * Y + T_matrix[0][1] *
                 (Cb - 128) + T_matrix[0][2] * (Cr - 128))
    R[R < 0] = 0
    R[R > 255] = 255

    G = np.round(T_matrix[1][0] * Y + T_matrix[1][1] *
                 (Cb - 128) + T_matrix[1][2] * (Cr - 128))
    G[G < 0] = 0
    G[G > 255] = 255

    B = np.round(T_matrix[2][0] * Y + T_matrix[2][1] *
                 (Cb - 128) + T_matrix[2][2] * (Cr - 128))
    B[B < 0] = 0
    B[B > 255] = 255

    return np.uint8(R), np.uint8(G), np.uint8(B)


# Ex 5.3
def show_ycbcr(Y, Cb, Cr):
    K = (0, 0, 0)
    W = (1, 1, 1)
    cm = colormap_function("Grays", K, W)

    # Plotting
    fig = plt.figure()

    # Y
    fig.add_subplot(1, 3, 1)
    plt.title("Y")
    plt.imshow(Y, cmap=cm)

    # Cb
    fig.add_subplot(1, 3, 2)
    plt.title("Cb")
    plt.imshow(Cb, cmap=cm)

    # Cr
    fig.add_subplot(1, 3, 3)
    plt.title("Cr")
    plt.imshow(Cr, cmap=cm)

    plt.subplots_adjust(wspace=0.5)


# Ex 6.1
def downsampling(Y, Cb, Cr, Yref, fatorCr, fatorCb):
    # Cr_d = Cr[:, ::fatorCr]

    # 4:2:0
    if fatorCb == 0:
        scaleX = 0.5
        scaleY = 0.5

        Cb_d = cv2.resize(Cb, None, fx=scaleX, fy=scaleY,
                          interpolation=cv2.INTER_NEAREST)
        Cr_d = cv2.resize(Cr, None, fx=scaleX, fy=scaleY,
                          interpolation=cv2.INTER_NEAREST)
        # Cb_d = Cb[::fatorCr, ::fatorCr]
        # Cr_d = Cr_d[::fatorCr]

    # 4:2:2
    else:
        scaleX = 0.5
        scaleY = 1
        Cb_d = cv2.resize(Cb, None, fx=scaleX, fy=scaleY,
                          interpolation=cv2.INTER_NEAREST)
        Cr_d = cv2.resize(Cr, None, fx=scaleX, fy=scaleY,
                          interpolation=cv2.INTER_NEAREST)
        # Cb_d = Cb[:, ::fatorCb]

    return Y, Cb_d, Cr_d


def upsampling(Y_d, Cb_d, Cr_d, type):

    if type == 0:
        scaleX = 0.5
        scaleY = 0.5
        stepX = int(1//scaleX)
        stepY = int(1//scaleY)

        Cb_u = cv2.resize(Cb_d, None, fx=stepX, fy=stepY,
                          interpolation=cv2.INTER_LINEAR)
        Cr_u = cv2.resize(Cr_d, None, fx=stepX, fy=stepY,
                          interpolation=cv2.INTER_LINEAR)

        # Cb_u = Cb_d.repeat(2, axis=0).repeat(2, axis=1)
        # Cr_u = Cr_d.repeat(2, axis=0).repeat(2, axis=1)

    else:
        scaleX = 0.5
        scaleY = 1
        stepX = int(1//scaleX)
        stepY = int(1//scaleY)
        Cb_u = cv2.resize(Cb_d, None, fx=stepX, fy=stepY,
                          interpolation=cv2.INTER_LINEAR)
        Cr_u = cv2.resize(Cr_d, None, fx=stepX, fy=stepY,
                          interpolation=cv2.INTER_LINEAR)
        # Cb_u = Cb_d.repeat(2, axis=1)
        # Cr_u = Cr_d.repeat(2, axis=1)

    return Y_d, Cb_u, Cr_u


# Ex 7
def dct(channel, blocks, channel_0):
    channel_dct_log = np.zeros(channel.shape)
    channel_dct = np.zeros(channel.shape)

    for i in range(0, len(channel), blocks):
        for j in range(0, len(channel[0]), channel_0):
            channel_sliced = channel[i:i+blocks, j:j+channel_0]
            channel_dct[i:i+blocks, j:j+channel_0] = fft.dct(
                fft.dct(channel_sliced, norm="ortho").T, norm="ortho").T
            channel_aux = fft.dct(
                fft.dct(channel_sliced, norm="ortho").T, norm="ortho").T
            channel_dct_log[i:i+blocks, j:j +
                            channel_0] = np.log(np.abs(channel_aux) + 0.0001)

    return channel_dct, channel_dct_log


def dct_inverse(channel_dct,  blocks, channel_0):
    channel_d = np.zeros(channel_dct.shape)

    for i in range(0, len(channel_dct), blocks):
        for j in range(0, len(channel_dct[0]), channel_0):
            channel_da = channel_dct[i:i+blocks, j:j+channel_0]
            channel_d[i:i+blocks, j:j+channel_0] = fft.idct(
                fft.idct(channel_da, norm="ortho").T, norm="ortho").T

    return channel_d

# Ex 8


def get_Q_matrixes(Y_dct, Cb_dct, blocks):
    Q_Y = np.array([[16,  11,  10,  16,  24,  40,  51,  61],
                    [12,  12,  14,  19,  26,  58,  60,  55],
                    [14,  13,  16,  24,  40,  57,  69,  56],
                    [14,  17,  22,  29,  51,  87,  80,  62],
                    [18,  22,  37,  56,  68, 109, 103,  77],
                    [24,  35,  55,  64,  81, 104, 113,  92],
                    [49,  64,  78,  87, 103, 121, 120, 101],
                    [72,  92,  95,  98, 112, 100, 103,  99]])
    Q_Y_with_tile = np.tile(
        Q_Y, (int(len(Y_dct)/blocks), int(len(Y_dct[0])/blocks)))

    Q_CbCr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                       [18, 21, 26, 66, 99, 99, 99, 99],
                       [24, 26, 56, 99, 99, 99, 99, 99],
                       [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])
    Q_CbCr_with_tile = np.tile(
        Q_CbCr, (int(len(Cb_dct)/blocks), int(len(Cb_dct[0])/blocks)))

    return Q_Y_with_tile, Q_CbCr_with_tile


def quantized_dct_coefficients_8x8(Y_dct, Cb_dct, Cr_dct, Q_Y_with_tile, Q_CbCr_with_tile):
    Y_q = np.round(Y_dct / Q_Y_with_tile)
    Cb_q = np.round(Cb_dct / Q_CbCr_with_tile)
    Cr_q = np.round(Cr_dct / Q_CbCr_with_tile)
    return Y_q, Cb_q, Cr_q


def quality_factor(Q, qf):
    if(qf >= 50):
        sf = (100 - qf) / 50
    else:
        sf = 50/qf

    if(sf != 0):
        Qs = np.round(Q * sf)
    else:
        Qs = np.ones(Q.shape, dtype=np.uint8)

    Qs[Qs > 255] = 255

    return Qs


def inverse_quantized_dct_coefficients_8x8(quantized_Y_dct, quantized_Cb_dct, quantized_Cr_dct, Qs_Y, Qs_CbCr):
    Y_dct = quantized_Y_dct * Qs_Y
    Cb_dct = quantized_Cb_dct * Qs_CbCr
    Cr_dct = quantized_Cr_dct * Qs_CbCr

    return Y_dct, Cb_dct, Cr_dct

# Ex 9


def coefficients_dc(dc, blocks):
    diff = dc

    for i in range(0, len(dc), blocks):
        for j in range(0, len(dc[0]), blocks):
            if j == 0:
                if i != 0:
                    diff[i][j] = dc[i][j] - dc[i-blocks][len(dc[0])-blocks-1]
            else:
                diff[i][j] = dc[i][j] - dc[i][j-blocks-1]

    return diff


def inverse_coefficients_dc(diff, blocks):
    dc = diff
    for i in range(0, len(diff), blocks):
        for j in range(0, len(diff[0]), blocks):
            if j == 0:
                if i != 0:
                    dc[i][j] = diff[i -
                                    blocks][len(diff[0])-blocks-1] + diff[i][j]
            else:
                dc[i][j] = diff[i][j-blocks-1] + diff[i][j]

    return dc
# -------------------------------------------------------------------------------------------- #

# Ex 10


def MSE(original_image, recovered_image):
    mse = np.sum((original_image - recovered_image) ** 2)
    mse /= float(original_image.shape[0] * original_image.shape[1])
    return mse


def RMSE(mse):
    rmse = math.sqrt(mse)
    return rmse


def SNR(original_image, mse):
    P = np.sum(original_image ** 2)
    P /= float(original_image.shape[0] * original_image.shape[1])
    snr = 10 * math.log10(P/mse)
    return snr


def PSNR(mse, original_image):
    original_image = original_image.flatten()
    max_ = max(original_image) ** 2
    psnr = 10 * math.log10(max_/mse)
    return psnr


def statistics(original_img, compressed_img, qf, img_name):
    print(
        f"\n\n=========== {qf} Quality factor for {img_name} ===========")
    mse = MSE(original_img, compressed_img)
    print(f"MSE: {mse}")
    rmse = RMSE(mse)
    print(f"RMSE: {rmse}")
    snr = SNR(original_img, mse)
    print(f"SNR: {snr}")
    psnr = PSNR(mse, original_img)
    print(f"PSNR: {psnr}\n")


def encoder(img, lines, columns):
    # -- 4 --
    img_padded = padding_function(img, lines, columns)
    R_p, G_p, B_p = rgb_components(img_padded)
    show_rgb(R_p, G_p, B_p)

    # -- 5 --
    Y, Cb, Cr = rgb_to_ycbcr(R_p, G_p, B_p)
    show_ycbcr(Y, Cb, Cr)

    # -- 6 --
    Y_d, Cb_d, Cr_d = downsampling(Y, Cb, Cr, 4, 2, 2)
    Y_d0, Cb_d0, Cr_d0 = downsampling(Y, Cb, Cr, 4, 2, 0)

    # Plotting
    fig = plt.figure()
    gray_colormap = colormap_function("gray", [0, 0, 0], [1, 1, 1])

    # Cb original
    fig.add_subplot(3, 2, 1)
    plt.title("Cb - Original")
    plt.imshow(Cb, cmap=gray_colormap)

    # Cr original
    fig.add_subplot(3, 2, 2)
    plt.title("Cr - Original")
    plt.imshow(Cr, cmap=gray_colormap)

    # Cb downsampled 4:2:2
    fig.add_subplot(3, 2, 3)
    plt.title("Cb - Downsampling 4:2:2")
    plt.imshow(Cb_d, cmap=gray_colormap)

    # Cr downsampled 4:2:2
    fig.add_subplot(3, 2, 4)
    plt.title("Cr - Downsampling 4:2:2")
    plt.imshow(Cr_d, cmap=gray_colormap)

    # Cb downsampled 4:2:0
    fig.add_subplot(3, 2, 5)
    plt.title("Cb - Downsampling 4:2:0")
    plt.imshow(Cb_d0, cmap=gray_colormap)

    # Cr downsampled 4:2:0
    fig.add_subplot(3, 2, 6)
    plt.title("Cr - Downsampling 4:2:0")
    plt.imshow(Cr_d0, cmap=gray_colormap)

    plt.subplots_adjust(hspace=0.5)

    # -- 7.1 --
    '''
    Y_dct, Y_dct_log = dct(Y_d0, len(Y_d0), len(Y_d0[0]))
    Cb_dct, Cb_dct_log = dct(Cb_d0, len(Cb_d0), len(Cb_d0[0]))
    Cr_dct, Cr_dct_log = dct(Cr_d0, len(Cr_d0), len(Cr_d0[0]))
    '''
    Y_dct, Y_dct_log = dct(Y_d0, 8, 8)
    Cb_dct, Cb_dct_log = dct(Cb_d0, 8, 8)
    Cr_dct, Cr_dct_log = dct(Cr_d0, 8, 8)

    # Plotting
    gray_colormap = colormap_function("gray", [0, 0, 0], [1, 1, 1])
    fig = plt.figure()

    # Y DCT
    fig.add_subplot(1, 3, 1)
    plt.title("Y DCT")
    plt.imshow(Y_dct_log, cmap=gray_colormap)
    plt.colorbar(shrink=0.5)

    # Cb DCT
    fig.add_subplot(1, 3, 2)
    plt.title("Cb DCT")
    plt.imshow(Cb_dct_log, cmap=gray_colormap)
    plt.colorbar(shrink=0.5)

    # Cr DCT
    fig.add_subplot(1, 3, 3)
    plt.title("Cr DCT")
    plt.imshow(Cr_dct_log, cmap=gray_colormap)
    plt.colorbar(shrink=0.5)

    plt.subplots_adjust(wspace=0.5)

    qf = np.array([10, 25, 50, 75, 100])

    # 8.1

    Q_Y_with_tile, Q_CbCr_with_tile = get_Q_matrixes(Y_dct, Cb_dct, 8)

    Qs_Y = quality_factor(Q_Y_with_tile, qf[2])
    Qs_CbCr = quality_factor(Q_CbCr_with_tile, qf[2])

    quantized_Y_dct, quantized_Cb_dct, quantized_Cr_dct = quantized_dct_coefficients_8x8(
        Y_dct, Cb_dct, Cr_dct, Qs_Y, Qs_CbCr)

    diff_Y = coefficients_dc(quantized_Y_dct, 8)
    diff_Cb = coefficients_dc(quantized_Cb_dct, 8)
    diff_Cr = coefficients_dc(quantized_Cr_dct, 8)

    # Plotting
    gray_colormap = colormap_function("gray", [0, 0, 0], [1, 1, 1])
    fig = plt.figure()

    # Y DCT
    fig.add_subplot(1, 3, 1)
    plt.title("Diff Y")
    plt.imshow(diff_Y, cmap=gray_colormap)
    plt.colorbar(shrink=0.5)

    # Cb DCT
    fig.add_subplot(1, 3, 2)
    plt.title("Diff Cb")
    plt.imshow(diff_Cb, cmap=gray_colormap)
    plt.colorbar(shrink=0.5)

    # Cr DCT
    fig.add_subplot(1, 3, 3)
    plt.title("Diff Cr")
    plt.imshow(diff_Cr, cmap=gray_colormap)
    plt.colorbar(shrink=0.5)

    plt.subplots_adjust(wspace=0.5)

    return diff_Y, diff_Cb, diff_Cr, qf[2]


def decoder(diff_Y, diff_Cb, diff_Cr, qf, n_lines, n_columns):

    # -- 9.1 and 9.2 --
    quantized_Y_dct = inverse_coefficients_dc(diff_Y, 8)
    quantized_Cb_dct = inverse_coefficients_dc(diff_Cb, 8)
    quantized_Cr_dct = inverse_coefficients_dc(diff_Cr, 8)

    # -- 8.1 and 8.2 --
    Q_Y_with_tile, Q_CbCr_with_tile = get_Q_matrixes(
        quantized_Y_dct, quantized_Cb_dct, 8)

    Qs_Y = quality_factor(Q_Y_with_tile, qf)
    Qs_CbCr = quality_factor(Q_CbCr_with_tile, qf)

    Y_dct, Cb_dct, Cr_dct = inverse_quantized_dct_coefficients_8x8(
        quantized_Y_dct, quantized_Cb_dct, quantized_Cr_dct, Qs_Y, Qs_CbCr)

    # -- 7.1 --
    Y_d0 = dct_inverse(Y_dct, 8, 8)
    Cb_d0 = dct_inverse(Cb_dct, 8, 8)
    Cr_d0 = dct_inverse(Cr_dct, 8, 8)

    '''
    Y_d0 = dct_inverse(Y_dct, len(Y_dct), len(Y_dct[0]))
    Cb_d0 = dct_inverse(Cb_dct, len(Cb_dct), len(Cb_dct[0]))
    Cr_d0 = dct_inverse(Cr_dct, len(Cr_dct), len(Cr_dct[0]))
    '''

    # -- 6 --
    # Downsampling 4:2:2
    # Y, Cb, Cr = upsampling(Y_d, Cr_d, Cb_d, 1)

    # Downsampling 4:2:0
    Y, Cb, Cr = upsampling(Y_d0, Cb_d0, Cr_d0, 0)

    # -- 5 --
    # YCbCr to RGB
    R, G, B = ycbcr_to_rgb(Y, Cb, Cr)

    # Joins all channels in one matrix
    matrix_joined_rgb = join_RGB(R, G, B)

    # -- 3 --
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.title("RGB channels joined with padding")
    plt.imshow(matrix_joined_rgb)

    # -- 4 --
    # Remove image padding
    img_without_padding = without_padding_function(
        matrix_joined_rgb, n_lines, n_columns)
    fig.add_subplot(1, 2, 2)
    plt.title("Final Image")
    plt.imshow(img_without_padding)

    #print(f"Final shape: {img_without_padding.shape}")

    return img_without_padding
# -------------------------------------------------------------------------------------------- #


def main():

    plt.close('all')

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_name = input("Image name: ")
    img_path = dir_path + "/imagens/" + img_name + ".bmp"
    img = read_image(img_path)

    # draw_plot("Original Image", img)

    #print(f"Initial shape: {img.shape}")
    (lines, columns, channels) = img.shape

    # retornar sempre o mais recente !!!
    Y_dct, Cb_dct, Cr_dct, qf = encoder(img, lines, columns)
    final_img = decoder(Y_dct, Cb_dct, Cr_dct, qf, lines, columns)

    # Print statistics
    statistics(img, final_img, qf, img_name)


if __name__ == "__main__":
    main()
    plt.show()
