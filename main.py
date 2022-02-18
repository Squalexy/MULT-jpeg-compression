import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import os

'''
def encoder(image):
    pass


def decoder(image):
    pass
'''

# Ex 3.1


def read_image(img_path):
    return plt.imread(img_path)


# Ex 3.2
def colormap_function(colormap_name, color1, color2):
    return clr.LinearSegmentedColormap.from_list(colormap_name, [color1, color2], 256)


# Ex 3.3
def visualize_image_with_colormap(channel, colormap):
    plt.figure()
    plt.imshow(channel, colormap)


# Ex 3.4
def rgb_components(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return R, G, B


def inversa(img):
    # if T is a matrix -> Ti = np.linalg.inv(T) to get inversed matrix
    R, G, B = rgb_components(img)
    # return tuple (R_inv, G_inv, B_inv)
    return np.linalg.inv(R), np.linalg.inv(G), np.linalg.inv(B)


# Ex 3.5
def show_plot(img):
    K = (0, 0, 0)
    R = (1, 0, 0)
    G = (0, 1, 0)
    B = (0, 0, 1)

    # Ex 3.4
    channel_R, channel_G, channel_B = rgb_components(img)
    inv_R, inv_G, inv_B = inversa(img)

    # red
    cm = colormap_function("myRed", K, R)
    visualize_image_with_colormap(channel_R, cm)
    # green
    cm = colormap_function("myGreen", K, G)
    visualize_image_with_colormap(channel_G, cm)
    # blue
    cm = colormap_function("myBlue", K, B)
    visualize_image_with_colormap(channel_B, cm)


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
    plt.figure()
    plt.title("Original image")
    plt.imshow(img)

    plt.figure()
    plt.title("Image with Padding")
    plt.imshow(img_with_padding)

    (lines_n, columns_n, channels_n) = img.shape
    (lines_n_padding, columns_n_padding, channels_n_padding) = img_with_padding.shape

    if(lines_n_padding-lines_n != 0):
        img_recovered = img_with_padding[:-(lines_n_padding-lines_n), :, :]
        img_with_padding = img_recovered
    if(columns_n_padding-columns_n != 0):
        img_recovered = img_with_padding[:, :-(columns_n_padding-columns_n), :]

    plt.figure()
    plt.title("Recovered image <=> Original")
    plt.imshow(img_recovered)

# Ex 5


def convert_rgb_to_YCbCr(img):
    print(img)
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

    plt.figure()
    plt.title("Image with YCbCr model")
    plt.imshow((YCbCr_matrix*255).astype(np.uint8))


def main():
    plt.close('all')
    option = 0
    while option not in [1, 2, 3, 4]:
        option = int(input("\nChoose one of the following options:\n"
                           "[1] 3.5 - View image and each one of its channels\n"
                           "[2] 4.0 - Padding\n"
                           "[3] 5.3 - View image in YCbCr color model\n"
                           "[4] Exit\n"
                           "Choice: "))
    # Ex 3
    if option == 1:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        name = input("Image name: ")
        img_path = dir_path + "/imagens/"+name+".bmp"
        # Ex3.1
        img = read_image(img_path)

        # Ex3.5
        show_plot(img)

    # Ex 4
    elif option == 2:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        name = input("Image name: ")
        img_path = dir_path + "/imagens/"+name+".bmp"
        img = read_image(img_path)

        img_with_padding = padding_function(img)
        without_padding_function(img, img_with_padding)

    elif option == 3:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        name = input("Image name: ")
        img_path = dir_path + "/imagens/"+name+".bmp"
        img = read_image(img_path)

        convert_rgb_to_YCbCr(img)

    elif option == 4:
        exit(0)

    '''
    print(img.shape)
    print(R.shape)
    print(img.dtype)
    '''

    '''
    COLORMAP
        Each channel has 256 colors -> because of the 8 bits
        Red colormap -> first color black, and the last saturated as red 
        It will create 256 dots between 0 and 1 (int the red channel)
    
    cmRed = clr.LinearSegmentedColormap.from_list('myRed', [(0, 0, 0), (1, 0, 0)], 256)
    cmGray = clr.LinearSegmentedColormap.from_list(
        'myGray', [(0, 0, 0), (1, 1, 1)], 256)
    plt.figure()
    plt.imshow(R, cmRed)
    plt.imshow(R, cmGray)
    '''

    # encoder(img)
    # imgRec = decoder()


# np.hstack e no.vstack
# if T is a matrix -> Ti = np.linalg.inv(T) to get inversed matrix
if __name__ == "__main__":
    main()
    plt.show()
