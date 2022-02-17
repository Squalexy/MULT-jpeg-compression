@ -0,0 +1,51 @@
import matplotlib 
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as clr


def encoder(image):
    pass


def decoder(image):
    pass


def main():
    plt.close('all')

    img = plt.imread('imagens/peppers.bmp')
    plt.figure()
    plt.imshow(img)
    
    print(img.shape)

    R = img[:, :, 0]
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
## if T is a matrix -> Ti = np.linalg.inv(T) to get inversed matrix

if __name__ == "__main__":
    main()