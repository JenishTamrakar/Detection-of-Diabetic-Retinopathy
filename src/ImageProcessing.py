import cv2
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import csv


class ImageProcessing:

    # gray-scale conversion
    def gray_scale_cvt(self, image):
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_img

    # median filter
    def denoise_image(self, img):
        denoised_img = cv2.medianBlur(img, 5)
        return denoised_img

    def denoise_gray_image(self, img):
        denoised_gray_img = cv2.medianBlur(img, 5)
        return denoised_gray_img

    # CLAHE histogram equalization
    def clahe_img(self, img):
        b, green_fundus, r = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # creation of a CLAHE object
        clahe_img = clahe.apply(green_fundus)
        return clahe_img


    # adaptive thresholding: adaptive mean and adaptive gaussian
    def img_thresholding(self, img):
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

    def main(self):
        df = pd.read_csv("records.csv")
        # print(df.head())
        path = df.at[0, 'filepath']

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
        img = cv2.imread(path)
        ip = ImageProcessing()
        gray_img = ip.gray_scale_cvt(img)
        denoised_img = ip.denoise_image(img)
        denoised_gray_img = cv2.medianBlur(gray_img, 5)
        clahe_image = ip.clahe_img(denoised_img)

        filename_w_extension = os.path.basename(path)
        file_name_no_extension, file_extension = os.path.splitext(filename_w_extension)
        # adding records into dataframe and storing in csv file
        df2 = pd.read_csv(current_directory + '/records.csv')
        df2["image_name"] = filename_w_extension

        # Delete the unwanted columns from the dataframe
        df2 = df2.drop("filepath", axis=1)
        df2 = df2.drop("Unnamed: 0", axis=1)
        df2.to_csv("records.csv", index=False)

        resized_img = cv2.resize(clahe_image, (360, 360))
        cv2.imshow('Preprocessed Image', resized_img)
        cv2.waitKey(0)
        cv2.imwrite(folder + file_name_no_extension + "_clahe.png", clahe_image)

if __name__ == "__main__":
    ip = ImageProcessing()
    ip.main()