# antigo menu feito
# está apenas aqui caso seja realmente necessário

import main
import os

def show_menu():
    option = 0
    while option not in [1, 2, 3, 4]:
        option = int(input("\nChoose one of the following options:\n"
                           "[1] 3.5 - View image and its RGB channels\n"
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
        img = main.read_image(img_path)

        # Ex3.5
        main.show_img_and_rgb(img)

    # Ex 4
    elif option == 2:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        name = input("Image name: ")
        img_path = dir_path + "/imagens/"+name+".bmp"
        img = main.read_image(img_path)

        img_with_padding = main.padding_function(img)
        main.without_padding_function(img, img_with_padding)

    elif option == 3:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        name = input("Image name: ")
        img_path = dir_path + "/imagens/"+name+".bmp"
        img = read_image(img_path)

        main.convert_rgb_to_YCbCr(img)

    elif option == 4:
        exit(0)


# comentários random que haviam

'''
print(img.shape)
print(R.shape)
print(img.dtype)

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
