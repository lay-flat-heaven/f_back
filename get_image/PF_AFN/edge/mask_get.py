### Author: Matiur Rahman Minar ###
### EMCOM Lab, SeoulTech, 2021 ###
### Task: Generating binary mask/silhouette/segmentation ###
### especially for clothing image ###
### Focused method: Binary thresholding ###


import os
import cv2
import numpy as np
from PIL import Image
# from matplotlib import pyplot as plt


def cloth_edging(im_path, save_path, viz=False):

    img = cv2.imread(im_path, 0)

    # img1 = Image.open(im_path).convert('RGB')

    lo = 250
    hi = 255

    # mode_select

    # ret,thresh1 = cv2.threshold(img, lo, hi,cv2.THRESH_BINARY)
    ret,th_bin2 = cv2.threshold(img, lo, hi, cv2.THRESH_BINARY_INV)
    # ret,thresh3 = cv2.threshold(img, lo, hi,cv2.THRESH_TRUNC)
    # ret,thresh4 = cv2.threshold(img, lo, hi,cv2.THRESH_TOZERO)
    # ret,thresh5 = cv2.threshold(img, lo, hi,cv2.THRESH_TOZERO_INV)
    # ret, thresh3 = cv2.threshold(img, lo, hi, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    # th_otsu = cv2.bitwise_not(thresh3)
    # thresh4 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # th_mean = cv2.bitwise_not(thresh4)
    # thresh5 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # th_gauss = cv2.bitwise_not(thresh5)

    th_bin = th_bin2

    # Filling operation:
    # Copy the thresholded image.
    im_floodfill = th_bin.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = th_bin.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    th_filled = th_bin | im_floodfill_inv

    # Morphology operation:
    kernel = np.ones((2,2),np.uint8)

    # opening for salt noise removal
    th_opened = cv2.morphologyEx(th_filled, cv2.MORPH_OPEN, kernel)

    # closing for pepper noise removal (not needed it seems)
    # th_closed = cv2.morphologyEx(th_opened, cv2.MORPH_CLOSE, kernel)

    # erosion for thinning out boundary
    kernel = np.ones((3,3),np.uint8)
    th_eroded = cv2.erode(th_opened,kernel,iterations=1)

    # if viz:
    #     # plot figures:
    #     titles = ['Original Image','Binary thresholding', 'Filling',  'Image', 'Opening', 'Erosion']
    #     images = [img1, th_bin, th_filled, img1, th_opened, th_eroded]
    #
    #     for i in range(6):
    #         plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    #         plt.title(titles[i])
    #         plt.xticks([]),plt.yticks([])
    #
    #     plt.show()

    # save result

    cv2.imwrite(save_path, th_eroded)

def batch_edging(im_path,save_path):

    ori = os.listdir(im_path)
    for name in ori:
        impath = os.path.join(im_path, name)
        sapath = os.path.join(save_path, name)
        cloth_edging(impath, sapath)

def single_edging(im_path,save_path):
    cloth_edging(im_path, save_path)




if __name__ == "__main__":

    image_path = "../dataset/real_clothes"
    save_path = "../dataset/real_edge"
    batch_edging(image_path, save_path)
