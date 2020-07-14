# optimisation required
import cv2
import numpy as np
import os
import pandas as pd


def adjust_gamma(image, gamma=1.0):
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def extract_ma(image):
    r, green, b = cv2.split(image)
    comp = 255 - green
    # clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    # histe = clahe.apply(comp)
    adjustImage = adjust_gamma(comp, gamma=3)
    comp = 255 - adjustImage
    J = adjust_gamma(comp, gamma=4)
    J = 255 - J
    J = adjust_gamma(J, gamma=4)

    K = np.ones((11, 11), np.float32)
    L = cv2.filter2D(J, -1, K)

    ret3, thresh2 = cv2.threshold(L, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel2 = np.ones((9, 9), np.uint8)
    tophat = cv2.morphologyEx(thresh2, cv2.MORPH_TOPHAT, kernel2)
    kernel3 = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernel3)
    return opening


if __name__ == "__main__":
    # detect the current working directory and print it
    current_directory = os.getcwd()
    pathFolder = current_directory + "\images/"
    filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder, x))]
    # print(filesArray)
    for file_name in filesArray:
        file_name_no_extension = os.path.splitext(file_name)[0]
        fundus = cv2.imread(pathFolder + '/' + file_name)
    # fundus = cv2.imread("D:/DR_Datasets/CLAHE_images/IDRiD_210_clahe.png")
    ma = extract_ma(fundus)

    # threshold
    th, thresh3 = cv2.threshold(ma, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # findcontours
    cnts = cv2.findContours(thresh3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # filter by area
    s1 = 2
    s2 = 500
    xcnts = []
    for cnt in cnts:
        if s1 < cv2.contourArea(cnt) < s2:
            xcnts.append(cnt)
    # printing number of MAs
    no_of_mas = len(xcnts)
    # print(no_of_mas)
    print("Number of MAs: {}".format(len(xcnts)))

    df = pd.read_csv(current_directory +'/records.csv')
    df["no_of_microaneurysms"] = no_of_mas
    df.to_csv("records.csv", index=False)
    img = cv2.resize(ma, (600, 600))
    cv2.imshow('MA', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

