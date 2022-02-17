import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np


def encoder(image):
    pass


def decoder(image):
    pass

# ex 3.1
def read_image(img_location):
    img = plt.imread(img_location)

# ex 3.3
def colormap_function(colormap_name, color1, color2):
    cm = clr.LinearSegmentedColormap.from_list(colormap_name, [color1, color2], 256)
    """
    plt.figure()
    plt.imshow(R, cm)
    """
    

# ex 3.4
def rgb_components(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return R, G, B

def inversa(img): 
    R_inv, G_inv, B_inv = rgb_components(img) # if T is a matrix -> Ti = np.linalg.inv(T) to get inversed matrix
    return (np.linalg.inv(R_inv), np.linalg.inv(G_inv), np.linalg.inv(B_inv)) # return tuple (R_inv, G_inv, B_inv)
    
# ex 3.5
def show_plot(img):
    figure = plt.figure()
    figure.figimage(img)

def show_rgb_plots(img):
    r, g, b = rgb_components(img)
    figure = plt.figure()
    figure.figimage(r)
    figure.figimage(g)
    figure.figimage(b)


def main():
    O = [0, 0, 0]
    R = [1, 0, 0]
    G = [0, 1, 0]
    B = [0, 0, 1]
    
    
    plt.close('all')
    option = 0
    while option not in [1,2,3,4,5]:
        print("Choose one of the following options:"
        "\3.1 - Visualize an image and each one of its channels \n")
    

    if option == 1:
        name = input(print("Image name: "))
        location = "/imagens/"+name+".bmp"
        img = read_image(img_location)

    
    print(img.shape)
    print(R.shape)

    print(img.dtype)
    
    '''
    COLORMAP
        Each channel has 256 colors -> because of the 8 bits
        Red colormap -> first color black, and the last saturated as red 
        It will create 256 dots between 0 and 1 (int the red channel)
    '''
    cmRed = clr.LinearSegmentedColormap.from_list('myRed', [(0, 0, 0), (1, 0, 0)], 256)
    cmGray = clr.LinearSegmentedColormap.from_list('myGray', [(0, 0, 0), (1, 1, 1)], 256)
    plt.figure()
    plt.imshow(R, cmRed)
    plt.imshow(R, cmGray)


    # encoder(img)
    #imgRec = decoder()
    
    

## np.hstack e no.vstack


if __name__ == "__main__":
    main()
    plt.show()
