



import sys





def invert_image(path_image):

    '''
    :param image: path of the image to invert
    :return: Nothing.
    '''


from PIL import Image
import PIL.ImageOps

image = Image.open(path_image) 
inverted_image = PIL.ImageOps.invert(image)
inverted_image.save(path_image)

if __name__ == "__main__":
    path_image = string(sys.argv[1])
    invert_image(path_image)



