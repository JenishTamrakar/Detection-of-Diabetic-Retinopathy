import cv2
import numpy as np
import pandas as pd
import os
import csv
from csv import writer
from csv import reader


def extract_bv(image):
    b, green_fundus, r = cv2.split(image)
    # cv2.imshow('gr', green_fundus)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)
    # contrast_enhanced_green_fundus = image

    # applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    f4 = cv2.subtract(R3, contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)

    # removing very small contours through area parameter noise removal
    ret, f6 = cv2.threshold(f5, 15, 255, cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(f6.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # change removed im2
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    # removal of blobs of unwanted bigger chunks taking in consideration they are not straight lines
    fundus_eroded = cv2.bitwise_not(newfin)
    xmask = np.ones(fundus.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(), cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_SIMPLE)  # changed removed x1
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"
        else:
            shape = "veins"
        if (shape == "circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)

    final_image = cv2.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)

    # cv2.imshow('fi', final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    blood_vessels = final_image
    return blood_vessels


# Append a column in existing csv using csv.reader and csv.writer classes
def add_column_in_csv(input_file, output_file, transform_row):

    # Open the input_file in read mode and output_file in write mode
    with open(input_file, 'r') as read_obj, \
            open(output_file, 'w', newline='') as write_obj:
        # Create a csv.reader object from the input file object
        csv_reader = reader(read_obj)
        # Create a csv.writer object from the output file object
        csv_writer = writer(write_obj)
        # Read each row of the input csv file as list
        for row in csv_reader:
            # Pass the list / row in the transform function to add column text for this row
            transform_row(row, csv_reader.line_num)
            # Write the updated row / list to the output file
            csv_writer.writerow(row)


if __name__ == "__main__":
    # pathFolder = "D:/DR_Datasets/CLAHE_images/"
    current_directory = os.getcwd()
    pathFolder = current_directory + "\images/"
    filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder, x))]

    # destinationFolder = "D:/DR_Datasets/Blood_vessels/"
    # if not os.path.exists(destinationFolder):
    #     os.mkdir(destinationFolder)
    lst = []
    for file_name in filesArray:
        file_name_no_extension = os.path.splitext(file_name)[0]
        fundus = cv2.imread(pathFolder + '/' + file_name)
        bloodvessel = extract_bv(fundus)
        # cv2.imwrite(destinationFolder + file_name_no_extension + "_bloodvessel.png", bloodvessel)

        # calculation of density of white pixels representing blood vessels
        n_white_pix = np.sum(bloodvessel == 255)
        height = bloodvessel.shape[0]
        width = bloodvessel.shape[1]
        total_pix = height * width
        density_white_pix = n_white_pix / total_pix
        print("No. of white pixels = ", n_white_pix)
        print("Density of white pixels = ", density_white_pix)

        # for preparation of training data
        # list of tuples of required data
        lst.append((density_white_pix))

        df2 = pd.DataFrame({'image_name': [file_name_no_extension], 'density_of_blood_vessels': [density_white_pix]})
        df2.to_csv('records.csv')

        # Create a DataFrame object for density of blood vessels
        df1 = pd.DataFrame(lst, columns=['density_of_blood_vessels'])
        # print(df1)

    # for preparation of training data
    # header_of_new_col = 'density_of_blood_vessels'
    # Add a list as column
    # add_column_in_csv('C:/Users/Jenish Tamrakar/Desktop/DR/training_sample.csv',
    #                   'C:/Users/Jenish Tamrakar/Desktop/DR/training_sample1.csv',
    #                   lambda row, line_num: row.append(lst[line_num - 1]))