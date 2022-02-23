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
def show_rgb(channel_R, channel_G, channel_B):
    K = (0, 0, 0)
    R = (1, 0, 0)
    G = (0, 1, 0)
    B = (0, 0, 1)    

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
    print(img.shape)        

    if columns_n % 16 != 0:
        num_columns_to_add = (16 - (columns_n % 16))
        array = img[:, -1:]
        
        #aux = np.hstack((img, img[:, -1:]))
        aux_2 = np.repeat(array, num_columns_to_add, axis=1)
        img = np.append(img, aux_2, axis= 1)

    if lines_n % 16 != 0:
        num_lines_to_add = (16 - (lines_n % 16))
        #img = np.vstack((img, img[-1:]))
        array= img[-1:]
        aux_2= np.repeat(array, num_lines_to_add, axis=0)
        img = np.append(img, aux_2, axis= 0)
        
    print(img.shape)        
    draw_plot("imagem com padding", img)
    return img


def without_padding_function(img, img_with_padding):

    draw_plot("Original image", img)
    draw_plot("Image with Padding", img_with_padding)

    (lines_n, columns_n, channels_n) = img.shape
    (lines_n_padding, columns_n_padding, channels_n_padding) = img_with_padding.shape
    
    img_recovered = img_with_padding

    if(lines_n_padding-lines_n != 0):
        img_recovered = img_with_padding[:-(lines_n_padding-lines_n), :, :]
        img_with_padding = img_recovered
    if(columns_n_padding-columns_n != 0):
        img_recovered = img_with_padding[:, :-(columns_n_padding-columns_n), :]

    draw_plot("Recovered image <=> Original", img_recovered)


# Ex 5.1
def rgb_to_ycbcr(img):
    R, G, B = rgb_components(img)
        
    Y = (0.299 * R) + (0.587 * G) + (0.114 * B)
    Cb = (-0.168736 * R) + (-0.331264 * G) + (0.5 * B) + 128
    Cr = (0.5 * R) + (-0.418688 * G) + (-0.081312 * B) + 128
    
    return Y, Cb, Cr

def ycbcr_to_rgb(Y, Cb, Cr):
    matrix = np.array([[0.299,   0.587,    0.114],
                       [-0.168736, -0.331264,     0.5],
                       [0.5, -0.418688, -0.081312]])
    T_matrix = np.linalg.inv(matrix)

    R = np.round(T_matrix[0][0] * Y + T_matrix[0][1] * (Cb - 128) + T_matrix[0][2] * (Cr - 128))
    R[R < 0] = 0
    R[R > 255] = 255
    
    G = np.round(T_matrix[1][0] * Y + T_matrix[1][1] * (Cb - 128) + T_matrix[1][2] * (Cr - 128))
    G[G < 0] = 0
    G[G > 255] = 255
    
    B = np.round(T_matrix[2][0] * Y + T_matrix[2][1] * (Cb - 128) + T_matrix[2][2] * (Cr - 128))
    B[B < 0] = 0
    B[B > 255] = 255

    return np.uint8(R), np.uint8(G), np.uint8(B)
    
    
    
# Ex 5.3
def show_ycbcr(Y, Cb, Cr):
    K = (0, 0, 0)
    W = (1, 1, 1)
    cm = colormap_function("Whites", K, W)
    draw_plot("Y", Y, cm)
    draw_plot("Cb", Cb, cm)
    draw_plot("Cr", Cr, cm)


# -------------------------------------------------------------------------------------------- #
def encoder(img):
    R, G, B = rgb_components(img)
    show_rgb(R, G, B)                                   # mostrar imagem + canais RGB
    padding_function(img)                               # padding
    Y, Cb, Cr = rgb_to_ycbcr(img)
    R, G, B = ycbcr_to_rgb(Y, Cb, Cr)
    show_ycbcr(Y, Cb, Cr)
    show_rgb(R, G, B)

def decoder(img):
    pass
    # without_padding_function(img, padding_function(img))    # padding inverso
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
