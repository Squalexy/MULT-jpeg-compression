import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import os
import menu 


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
        plt.imshow(image, cmap = colormap)
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
def inversa(img):
    
    R, G, B = rgb_components(img)
    # return tuple (R_inv, G_inv, B_inv)
    return np.linalg.inv(R), np.linalg.inv(G), np.linalg.inv(B)


# Ex 3.5
def show_img_and_rgb(img):
    K = (0, 0, 0)
    R = (1, 0, 0)
    G = (0, 1, 0)
    B = (0, 0, 1)

    channel_R, channel_G, channel_B = rgb_components(img)
    
    # isto é para usar algures em algum momento... idk when
    # inv_R, inv_G, inv_B = inversa(img)

    cm = colormap_function("Reds", K, R)                    # red
    draw_plot("channel_R", channel_R, cm)
    cm = colormap_function("Greens", K, G)                  # green
    draw_plot("channel_G", channel_G, cm)
    cm = colormap_function("Blues", K, B)                   # blue
    draw_plot("channel_B", channel_B, cm)


# Ex 4
def padding_function(img):
    (lines_n, columns_n, channels_n) = img.shape
    num_lines_to_add = 0
    num_columns_to_add = 0

    if columns_n % 16 != 0:
        num_columns_to_add = (16 - (columns_n % 16))
        for i in range(num_columns_to_add):
            img = np.hstack((img, img[:, -1:]))

    if lines_n % 16 != 0:
        num_lines_to_add = (16 - (lines_n % 16))
        for i in range(num_lines_to_add):
            img = np.vstack((img, img[-1:]))
    return img


def without_padding_function(img, img_with_padding):

    draw_plot("Original image", img)
    draw_plot("Image with Padding", img_with_padding)

    (lines_n, columns_n, channels_n) = img.shape
    (lines_n_padding, columns_n_padding, channels_n_padding) = img_with_padding.shape
    
    # acrescentei esta linha porque é necessário inicializar a variável antes dela entrar no if
    img_recovered = img_with_padding

    if(lines_n_padding-lines_n != 0):
        img_recovered = img_with_padding[:-(lines_n_padding-lines_n), :, :]
        img_with_padding = img_recovered
    if(columns_n_padding-columns_n != 0):
        img_recovered = img_with_padding[:, :-(columns_n_padding-columns_n), :]

    draw_plot("Recovered image <=> Original", img_recovered)


# Ex 5
def convert_rgb_to_YCbCr(img):
    # print(img)
    (lines_n, columns_n, channels_n) = img.shape
    YCbCr_matrix = np.zeros(img.shape, dtype=np.float64)
    matrix = np.array([[0.299,   0.587,    0.114],
                       [-0.168736, -0.331264,     0.5],
                       [0.5, -0.418688, -0.081312]])
    Cb_Cr_matrix = np.array([0,    128,     128])

    for i in range(lines_n):
        for j in range(columns_n):
            RGB_matrix = np.array([img[i, j, 0], img[i, j, 1], img[i, j, 2]])
            YCbCr_aux = matrix.dot(RGB_matrix) + Cb_Cr_matrix
            YCbCr_matrix[i][j] = YCbCr_aux

    draw_plot("Image with YCbCr model", (YCbCr_matrix*255).astype(np.uint8))


# -------------------------------------------------------------------------------------------- #
def encoder(img):
    show_img_and_rgb(img)                                   # mostrar imagem + canais RGB
    padding_function(img)                                   # padding
    without_padding_function(img, padding_function(img))    # padding inverso
    convert_rgb_to_YCbCr(img)                               # converter RGB para YCbCr
    

def decoder(img):
    pass
# -------------------------------------------------------------------------------------------- #


def main():

    plt.close('all')

    # menu.show_menu()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_name = input("Image name: ")
    img_path = dir_path + "/imagens/" + img_name + ".bmp"
    img = read_image(img_path)

    encoder(img)
    # decoder(img) 


if __name__ == "__main__":
    main()
    plt.show()
