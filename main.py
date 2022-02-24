from ntpath import join
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
def join_RGB(R, G, B):
    matrix_inverted = np.zeros((len(R), len(R[0]), 3), dtype = np.uint8)
    # return tuple (R_inv, G_inv, B_inv)
    #aux = np.append(R, G, axis = 1)
    #matrix_inverted = np.append(aux, B, axis = 1)
    matrix_inverted[:,:,0] = R
    matrix_inverted[:,:,1] = G
    matrix_inverted[:,:,2] = B
    return matrix_inverted


# Ex 3.5
def show_rgb(channel_R, channel_G, channel_B):
    K = (0, 0, 0)
    R = (1, 0, 0)
    G = (0, 1, 0)
    B = (0, 0, 1)    

    cm = colormap_function("Reds", K, R)                    # red
    draw_plot("Encoder - channel_R with padding", channel_R, cm)
    cm = colormap_function("Greens", K, G)                  # green
    draw_plot("Encoder - channel_G with padding", channel_G, cm)
    cm = colormap_function("Blues", K, B)                   # blue
    draw_plot("Encoder - channel_B with padding", channel_B, cm)


# Ex 4
def padding_function(img, lines, columns):
    num_lines_to_add = 0
    num_columns_to_add = 0
    print(img.shape)        

    if columns % 16 != 0:
        num_columns_to_add = (16 - (columns % 16))
        array = img[:, -1:]

        aux_2 = np.repeat(array, num_columns_to_add, axis=1)
        img = np.append(img, aux_2, axis= 1)

    if lines % 16 != 0:
        num_lines_to_add = (16 - (lines % 16))
        array= img[-1:]
        aux_2= np.repeat(array, num_lines_to_add, axis=0)
        img = np.append(img, aux_2, axis= 0)
        
    print(img.shape)        
    draw_plot("Encoder - Image with padding", img)
    global img_padded 
    img_padded = img
    return img_padded


# passar logo n linhas e colunas 
def without_padding_function(img_with_padding, lines, columns):
    img_recovered = img_with_padding[:lines, :, :]
    img_recovered = img_recovered[:, :columns, :]
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
    draw_plot("Encoder - Y with padding", Y, cm)
    draw_plot("Encoder - Cb with padding", Cb, cm)
    draw_plot("Encoder - Cr with padding", Cr, cm)


# -------------------------------------------------------------------------------------------- #
def encoder(img, lines, columns):
    
    img_padded = padding_function(img, lines, columns) 
    R_p, G_p, B_p = rgb_components(img_padded)         
    show_rgb(R_p, G_p, B_p)          
    Y, Cb, Cr = rgb_to_ycbcr(R_p, G_p, B_p)
    
    show_ycbcr(Y, Cb, Cr)
    
    # retornar sempre o mais recente
    return Y, Cb, Cr

def decoder(Y, Cb, Cr, n_lines, n_columns):
    R, G, B = ycbcr_to_rgb(Y, Cb, Cr)
    matrix_joined_rgb = join_RGB(R, G, B)
    img_without_padding = without_padding_function(matrix_joined_rgb, n_lines, n_columns)
    draw_plot("decoder - Img without padding", img_without_padding)
    draw_plot("decoder - RGB channels joined with padding", matrix_joined_rgb)
    print(f"Decoder - Final shape: {img_without_padding.shape}" )
# -------------------------------------------------------------------------------------------- #


def main():

    plt.close('all')

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_name = input("Image name: ")
    img_path = dir_path + "/imagens/" + img_name + ".bmp"
    img = read_image(img_path)

    (lines, columns, channels) = img.shape
    
    # retornar sempre o mais recente !!!
    Y, Cb, Cr = encoder(img, lines, columns)
    decoder(Y, Cb, Cr, lines, columns) 


if __name__ == "__main__":
    main()
    plt.show()
