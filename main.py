from ntpath import join
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import os
import menu
import scipy.fftpack as fft
import cv2

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
    #Cr_d = Cr[:, ::fatorCr]

    if fatorCb == 0:
        scaleX = 0.5
        scaleY = 0.5

        Cb_d = cv2.resize(Cb, None, fx=scaleX, fy=scaleY,
                          interpolation=cv2.INTER_NEAREST)
        Cr_d = cv2.resize(Cr, None, fx=scaleX, fy=scaleY,
                          interpolation=cv2.INTER_NEAREST)
        #Cb_d = Cb[::fatorCr, ::fatorCr]
        #Cr_d = Cr_d[::fatorCr]
    else:
        scaleX = 0.5
        scaleY = 1
        Cb_d = cv2.resize(Cb, None, fx=scaleX, fy=scaleY,
                          interpolation=cv2.INTER_NEAREST)
        Cr_d = cv2.resize(Cr, None, fx=scaleX, fy=scaleY,
                          interpolation=cv2.INTER_NEAREST)
        #Cb_d = Cb[:, ::fatorCb]

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

        #Cb_u = Cb_d.repeat(2, axis=0).repeat(2, axis=1)
        #Cr_u = Cr_d.repeat(2, axis=0).repeat(2, axis=1)

    else:
        scaleX = 0.5
        scaleY = 1
        stepX = int(1//scaleX)
        stepY = int(1//scaleY)
        Cb_u = cv2.resize(Cb_d, None, fx=stepX, fy=stepY,
                          interpolation=cv2.INTER_LINEAR)
        Cr_u = cv2.resize(Cr_d, None, fx=stepX, fy=stepY,
                          interpolation=cv2.INTER_LINEAR)
        #Cb_u = Cb_d.repeat(2, axis=1)
        #Cr_u = Cr_d.repeat(2, axis=1)

    return Y_d, Cb_u, Cr_u


# Ex 7
def dct(canal, blocks):
    canal_dct_log = np.zeros(canal.shape)
    canal_dct = np.zeros(canal.shape)

    if blocks == "all":
        canal_dct = fft.dct(fft.dct(canal, norm="ortho").T, norm="ortho").T
        canal_dct_log = np.log(np.abs(canal_dct) + 0.0001)

    # 7.2 - fazer mais tarde
    elif blocks == "8":
        for i in range(0, len(canal), 8):
            for j in range(0, len(canal[0]), 8):
                canal_sliced = canal[i:i+8, j:j+8]
                canal_dct[i:i+8, j:j+8] = fft.dct(
                    fft.dct(canal_sliced, norm="ortho").T, norm="ortho").T

                canal_aux = fft.dct(
                    fft.dct(canal_sliced, norm="ortho").T, norm="ortho").T
                canal_dct_log[i:i+8, j:j +
                              8] = np.log(np.abs(canal_aux) + 0.0001)

    # 7.3 - fazer mais tarde
    elif blocks == "64":
        for i in range(0, len(canal), 64):
            for j in range(0, len(canal[0]), 64):
                canal_sliced = canal[i:i+64, j:j+64]
                canal_dct[i:i+64, j:j+64] = fft.dct(
                    fft.dct(canal_sliced, norm="ortho").T, norm="ortho").T

                canal_aux = fft.dct(
                    fft.dct(canal_sliced, norm="ortho").T, norm="ortho").T
                canal_dct_log[i:i+64, j:j +
                              64] = np.log(np.abs(canal_aux) + 0.0001)

    return canal_dct, canal_dct_log


def dct_inverse(canal_dct,  blocks):
    canal_d = np.zeros(canal_dct.shape)
    canal_ = np.zeros(canal_dct.shape)

    if blocks == "all":
        canal_d = fft.idct(fft.idct(canal_dct, norm="ortho").T, norm="ortho").T

    # 7.2
    elif blocks == "8":
        for i in range(0, len(canal_dct), 8):
            for j in range(0, len(canal_dct[0]), 8):
                canal_da = canal_dct[i:i+8, j:j+8]
                canal_d[i:i+8, j:j+8] = fft.idct(
                    fft.idct(canal_da, norm="ortho").T, norm="ortho").T

    # 7.3
    elif blocks == "64":
        for i in range(0, len(canal_dct), 64):
            for j in range(0, len(canal_dct[0]), 64):
                canal_dct_sliced = canal_dct[i:i+64, j:j+64]
                canal_d[i:i+64, j:j+64] = fft.idct(
                    fft.idct(canal_dct_sliced, norm="ortho").T, norm="ortho").T

    return canal_d


# -------------------------------------------------------------------------------------------- #
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
    Y_dct, Y_dct_log = dct(Y_d0, "8")
    Cb_dct, Cb_dct_log = dct(Cb_d0, "8")
    Cr_dct, Cr_dct_log = dct(Cr_d0, "8")

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

    return Y_dct, Cb_dct, Cr_dct


def decoder(Y_dct, Cb_dct, Cr_dct, n_lines, n_columns):
    # -- 7.1 --
    Y_d0 = dct_inverse(Y_dct, "8")
    Cb_d0 = dct_inverse(Cb_dct, "8")
    Cr_d0 = dct_inverse(Cr_dct, "8")

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

    print(f"Final shape: {img_without_padding.shape}")
# -------------------------------------------------------------------------------------------- #


def main():

    plt.close('all')

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_name = input("Image name: ")
    img_path = dir_path + "/imagens/" + img_name + ".bmp"
    img = read_image(img_path)

    #draw_plot("Original Image", img)

    print(f"Initial shape: {img.shape}")
    (lines, columns, channels) = img.shape

    # retornar sempre o mais recente !!!
    Y_dct, Cb_dct, Cr_dct = encoder(img, lines, columns)
    decoder(Y_dct, Cb_dct, Cr_dct, lines, columns)


if __name__ == "__main__":
    main()
    plt.show()
