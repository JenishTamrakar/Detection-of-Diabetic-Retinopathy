import numpy as np
import cv2
import os
import pandas as pd
import glob

class HemorrhageDetection:
    def detect_he(self, image):
        blur = cv2.GaussianBlur(image, (7, 7), 2)
        h, w = image.shape[:2]

        # Morphological gradient

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
        # resized_gradient = cv2.resize(gradient, (600, 600))
        # cv2.imshow('Morphological gradient', resized_gradient)
        # cv2.waitKey(0)

        # Binarize gradient
        lowerb = np.array([0, 0, 0])
        upperb = np.array([15, 15, 15])
        binary = cv2.inRange(gradient, lowerb, upperb)
        # resized_binary = cv2.resize(binary, (600, 600))
        # cv2.imshow('Binarized gradient', resized_binary)
        # cv2.waitKey(0)

        # Flood fill from the edges to remove edge crystals
        for row in range(h):
            if binary[row, 0] == 255:
                cv2.floodFill(binary, None, (0, row), 0)
            if binary[row, w-1] == 255:
                cv2.floodFill(binary, None, (w-1, row), 0)

        for col in range(w):
            if binary[0, col] == 255:
                cv2.floodFill(binary, None, (col, 0), 0)
            if binary[h-1, col] == 255:
                cv2.floodFill(binary, None, (col, h-1), 0)
        # resized_binary1 = cv2.resize(binary, (600, 600))
        # cv2.imshow('Filled binary gradient', resized_binary1)
        # cv2.waitKey(0)

        # Cleaning up mask
        foreground = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
        resized_foreground = cv2.resize(foreground, (360, 360))
        # cv2.imshow('Cleanup up crystal foreground mask', resized_foreground)
        cv2.imshow('Hemorrhage', resized_foreground)
        cv2.waitKey(0)

        # Creating background and unknown mask for labeling
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        background = cv2.dilate(foreground, kernel, iterations=3)
        unknown = cv2.subtract(background, foreground)
        # resized_background = cv2.resize(background, (600, 600))
        # cv2.imshow('Background', resized_background)
        # cv2.waitKey(0)

        # Watershed

        markers = cv2.connectedComponents(foreground)[1]
        markers += 1  # Add one to all labels so that background is 1, not 0
        markers[unknown==255] = 0  # mark the region of unknown with zero
        markers = cv2.watershed(image, markers)

        # Assign the markers a hue between 0 and 179
        hue_markers = np.uint8(179*np.float32(markers)/np.max(markers))
        blank_channel = 255*np.ones((h, w), dtype=np.uint8)
        marker_img = cv2.merge([hue_markers, blank_channel, blank_channel])
        marker_img = cv2.cvtColor(marker_img, cv2.COLOR_HSV2BGR)
        # resized_marker_img = cv2.resize(marker_img, (600, 600))
        # cv2.imshow('Colored markers', resized_marker_img)
        # cv2.waitKey(0)

        # Label the original image with the watershed markers
        labeled_img = image.copy()
        labeled_img[markers>1] = marker_img[markers>1]  # 1 is background color
        labeled_img = cv2.addWeighted(image, 0.5, labeled_img, 0.5, 0)
        # resized_labeled_img = cv2.resize(labeled_img, (360, 360))
        # cv2.imshow('watershed_result.png', resized_labeled_img)
        # cv2.waitKey(0)

        # threshold
        th, thresh3 = cv2.threshold(background, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # findcontours
        cnts = cv2.findContours(thresh3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # detect the current working directory and print it
        current_directory = os.getcwd()

        # filter by area
        s1 = 10000
        # s2 = 1000
        xcnts = []
        for cnt in cnts:
            if s1 < cv2.contourArea(cnt):
                xcnts.append(cnt)
        # printing number of Hemorrhages
        # print(xcnts)
        if format(len(xcnts))==1 and xcnts[0] == 0 :
            print("Number of Hemorrhages = 0")
        else:
            print("Number of Hemorrhages: {}".format(len(xcnts)))
        no_of_he = len(xcnts)
        df = pd.read_csv(current_directory + '/records.csv')
        df["no_of_haemorrhages"] = no_of_he
        df.to_csv("records.csv", index=False)

        # removing images directory
        files = glob.glob(current_directory+'/images/*')
        for f in files:
            os.remove(f)
        dir_path = current_directory + "/images"
        os.rmdir(dir_path)


    def main(self):
        # detect the current working directory and print it
        current_directory = os.getcwd()
        path = current_directory + "\images/"
        # print(path)
        filesArray = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]

        for file_name in filesArray:
            file_name_no_extension = os.path.splitext(file_name)[0]
            img = cv2.imread(path + '/' + file_name)
        hd = HemorrhageDetection()
        blur_img = hd.detect_he(img)

if __name__ == "__main__":
    hd = HemorrhageDetection()
    hd.main()