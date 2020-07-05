import csv
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def findMA(image):
    b, green_fundus, r = cv2.split(image)

    # Canny edge detection algorithm. t1 specifies the threshold value to begin an edge. t2 specifies the strength required
    eye_edges = cv2.Canny(green_fundus, 70, 35)

    edge_test = green_fundus+eye_edges
    eye_final = eye_edges

    # Perform closing to find individual objects
    kernel = np.ones((5, 5), np.uint8)
    eye_final = cv2.dilate(eye_final, kernel, iterations=2)
    eye_final = cv2.erode(eye_final, kernel, iterations=1)

    eye_final = cv2.dilate(eye_final, kernel, iterations=4)
    eye_final = cv2.erode(eye_final, kernel, iterations=3)

    # Setup SimpleBlobDetector parameters for detecting big MAs.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Set up the detector with parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(green_fundus)
    df = pd.read_csv("IDRiD_Disease_Grading_Training_Labels.csv")
    # create a blank image to mask
    masked_image = image.size

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (255, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image_with_mas = cv2.resize(im_with_keypoints, (1024, 840))
    cv2.imshow('Image With Big MAs', image_with_mas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # titles = ['Original Image', 'Image with MA']
    # images = [image, image_with_mas]
    #
    # for i in range(2):
    #     plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()
    b, green_im_with_keypoints, r = cv2.split(im_with_keypoints)
    eye_final_big_blobs = eye_final - green_im_with_keypoints
    # eye_final = eye_final - im_with_keypoints
    eye_final_big_blobs = cv2.erode(eye_final_big_blobs, kernel, iterations=1)
    resized_eye_final_big_blobs = cv2.resize(eye_final_big_blobs, (512, 512))
    # eye_final = cv2.cvtColor(eye_final, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Image with Big MAs', resized_eye_final_big_blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # params for for detecting smaller MAs.
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Set up the detector with parameters.
    detector1 = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints1 = detector1.detect(green_fundus)

    # create a blank image to mask
    masked_image1 = image.size

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints1 = cv2.drawKeypoints(image, keypoints1, np.array([]), (255, 255, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image_with_mas1 = cv2.resize(im_with_keypoints1, (1024, 840))
    cv2.imshow('Image With Big MAs', image_with_mas1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    b, green_im_with_keypoints1, r = cv2.split(im_with_keypoints1)
    eye_final_small_blobs = eye_final - green_im_with_keypoints1
    # eye_final = eye_final - im_with_keypoints
    eye_final_small_blobs = cv2.erode(eye_final_small_blobs, kernel, iterations=1)
    resized_eye_final_small_blobs = cv2.resize(eye_final_small_blobs, (512, 512))
    # eye_final = cv2.cvtColor(eye_final, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Image with Smaller MAs', resized_eye_final_small_blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return eye_final

if __name__ == "__main__":
    pathFolder = "D:/DR_Datasets/CLAHE_imgs/"
    filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder, x))]
    destinationFolder = "D:/DR_Datasets/MAs/"
    if not os.path.exists(destinationFolder):
        os.mkdir(destinationFolder)
    for file_name in filesArray:
        file_name_no_extension = os.path.splitext(file_name)[0]
        fundus = cv2.imread(pathFolder + '/' + file_name)
        # fundus = cv2.imread("D:/DR_Datasets/diabetic_retinopathy/01_dr.JPG")
        print(fundus)
        mas = findMA(fundus)
        cv2.imwrite(destinationFolder + file_name_no_extension + "_mas.png", mas)
        # break
# pathFolder = "D:/DR_Datasets/CLAHE_imgs/"
# filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder, x))]
# MA_Folder = pathFolder+"/MA/"

# if not os.path.exists(MA_Folder):
#     os.mkdir(MA_Folder)


# with open('ma.csv', 'w') as csvfile:
#     filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     filewriter.writerow(['microaneurysmcount', 'countvalue'])
#     for file_name in filesArray:
#         print(pathFolder + '/' + file_name)
#         print(os.path.splitext(file_name)[0])
#         final_madetected = findMA(pathFolder + '/' + file_name, os.path.splitext(file_name)[0])
#
#         i = 0
#         j = 0
#         white = 0
#         black = 0
#
#         # while i < final_madetected.size()[0]:
#         #     j = 0
#         #     while j < final_madetected.size()[1]:
#         #         if final_madetected[i,j] == 255:
#         #             white = white + 1
#         #         j = j+1
#         #     i = i+1
#         # print('lag gye dobara')
#         print(final_madetected.size())
#
#         while i < final_madetected.size()[0]:
#             j = 0
#             while j < final_madetected.size()[1]:
#
#                 if final_madetected[i, j][0] != 0:
#                     white = white + 1
#                     # print(final_madetected[i,j][k],i,j,k)
#                 else:
#                     black = black + 1
#
#                 j = j + 1
#             i = i + 1
#
#         print(white)
#         # print(black)
#         final_white_pixels = white / 3
#
#         filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         filewriter.writerow([os.path.splitext(file_name)[0] + "_microaneurysm.jpg", white])
    # cv2.imwrite(os.path.splitext(file_name)[0]+"_microaneurysm.jpg",bloodvessel)

    # file_name_no_extension = os.path.splitext(file_name)[0]
    # counter = maskWhiteCounter(exudate_image)
    # array_exudate_pixels.append(counter)
    # cv2.imwrite(exudateFolder+file_name_no_extension+"_exudates.jpg",exudate_image)




# new
# import cv2
# import numpy as np
# import os
#
# def adjust_gamma(image, gamma=1.0):
#     table = np.array([((i / 255.0) ** gamma) * 255
#                       for i in np.arange(0, 256)]).astype("uint8")
#
#     return cv2.LUT(image, table)
#
#
# def extract_ma(image):
#     r, g, b = cv2.split(image)
#     comp = 255 - g
#     clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
#     histe = clahe.apply(comp)
#     adjustImage = adjust_gamma(histe, gamma=3)
#     comp = 255 - adjustImage
#     J = adjust_gamma(comp, gamma=4)
#     J = 255 - J
#     J = adjust_gamma(J, gamma=4)
#
#     K = np.ones((11, 11), np.float32)
#     L = cv2.filter2D(J, -1, K)
#
#     ret3, thresh2 = cv2.threshold(L, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     kernel2 = np.ones((9, 9), np.uint8)
#     tophat = cv2.morphologyEx(thresh2, cv2.MORPH_TOPHAT, kernel2)
#     kernel3 = np.ones((7, 7), np.uint8)
#     opening = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernel3)
#     return opening
#
#
# if __name__ == "__main__":
#     pathFolder = "D:/DR_Datasets/healthy/"
#     filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder, x))]
#     destinationFolder = "D:/DR_Datasets/MAs/"
#     if not os.path.exists(destinationFolder):
#         os.mkdir(destinationFolder)
#     for file_name in filesArray:
#         file_name_no_extension = os.path.splitext(file_name)[0]
#         fundus = cv2.imread(pathFolder + '/' + file_name)
#         # fundus = cv2.imread("D:/DR_Datasets/diabetic_retinopathy/01_dr.JPG")
#         # print(fundus)
#         mas = extract_ma(fundus)
#         cv2.imwrite(destinationFolder + file_name_no_extension + "_ma.png", mas)
#
#     # fundus = cv2.imread("D:/DR_Datasets/CLAHE_imgs/")
#     #     # bloodvessel = extract_ma(fundus)
#     #     #
#     #     # cv2.imwrite("22_MA.png", bloodvessel)