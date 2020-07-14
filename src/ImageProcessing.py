import cv2
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import csv


class ImageProcessing:

    # resize images
    # def resize_image(image):
    #     # img = cv2.resize(img, (256, 256))
    #     resized_img = cv2.resize(image, (300, 300))
    #     return resized_img

    # img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

    # gray-scale conversion
    def gray_scale_cvt(image):
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_img

    # median filter
    def denoise_image(img):
        denoised_img = cv2.medianBlur(img, 5)
        return denoised_img

    def denoise_gray_image(img):
        denoised_gray_img = cv2.medianBlur(img, 5)
        return denoised_gray_img

    # CLAHE histogram equalization
    def clahe_img(img):
        b, green_fundus, r = cv2.split(img)
        # print(green_fundus)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # creation of a CLAHE object
        clahe_img = clahe.apply(green_fundus)
        return clahe_img

    # image thresholding
    # ret,thresh1 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    # ret,thresh2 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
    # ret,thresh3 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_TRUNC)
    # ret,thresh4 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_TOZERO)
    # ret,thresh5 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_TOZERO_INV)

    # ret,thresh1 = cv2.threshold(clahe_img, 127, 255, cv2.THRESH_BINARY)
    # ret,thresh2 = cv2.threshold(clahe_img, 127, 255, cv2.THRESH_BINARY_INV)
    # ret,thresh3 = cv2.threshold(clahe_img, 127, 255, cv2.THRESH_TRUNC)
    # ret,thresh4 = cv2.threshold(clahe_img, 127, 255, cv2.THRESH_TOZERO)
    # ret,thresh5 = cv2.threshold(clahe_img, 127, 255, cv2.THRESH_TOZERO_INV)

    # titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    # images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    #
    # for i in range(6):
    #     plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()

    # adaptive thresholding: adaptive mean and adaptive gaussian
    def img_thresholding(img):
        ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        images = [img, th1, th2, th3]

        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()
        return

        # path = "D:/DR_Datasets/healthy/06_h.JPG"

    if __name__ == "__main__":
        path = "D:/DR_Datasets/B. Disease Grading/1. Original Images/a. Training Set\IDRiD_111.jpg"
        # pathFolder = "D:/DR_Datasets/B. Disease Grading/1. Original Images/a. Training Set/"
        # filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder, x))]
        # print(filesArray)

        # for preparation of training data
        # df = pd.read_csv("IDRiD_Disease_Grading_Training_Labels.csv")
        # df["density_of_blood_vessels"] = ""
        # df["no_of_microaneurysms"] = ""
        # df["no_of_haemorrhages"] = ""
        # df["no_of_exudates"] = ""
        # df.to_csv("data/a. IDRiD_Disease Grading_Training Labels.csv", index=False)
        # print(df.head())
        # end preparation of training data

        # detect the current working directory and print it
        current_directory = os.getcwd()
        # print(current_directory)
        destinationFolder = os.path.join(current_directory, r'images')
        folder = destinationFolder + '/'

        if not os.path.exists(destinationFolder):
            os.mkdir(destinationFolder)
        # des_folder = current_directory + "/images"
        # for file_name in filesArray:
        #     file_name_no_extension = os.path.splitext(file_name)[0]
        #     img = cv2.imread(pathFolder + '/' + file_name)
        img = cv2.imread(path)
        gray_img = gray_scale_cvt(img)
        denoised_img = denoise_image(img)
        denoised_gray_img = cv2.medianBlur(gray_img, 5)
        clahe_image = clahe_img(denoised_img)
        cv2.imshow('', clahe_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        filename_w_extension = os.path.basename(path)
        file_name_no_extension, file_extension = os.path.splitext(filename_w_extension)

        cv2.imwrite(folder + file_name_no_extension + "_clahe.png", clahe_image)
            # canny edge detection start
            # edges = cv2.Canny(clahe_img, 100, 200)
            # plt.subplot(121),plt.imshow(img,cmap = 'gray')
            # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
            # plt.title('Edge Detected Image'), plt.xticks([]), plt.yticks([])
            # plt.show()
            # edge detection end
            # file_name_no_extension = os.path.splitext(file_name)[0]
            # cv2.imwrite(destinationFolder + file_name_no_extension + "_clahe.png", clahe_image)
