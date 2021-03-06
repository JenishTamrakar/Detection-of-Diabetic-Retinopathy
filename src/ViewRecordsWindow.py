# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ViewRecordsWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import csv

from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
# from src.DiagnosisWindow import DiagnosisWindow


class ViewRecordsWindow(object):
    def setupUi(self, Form):

        Form.setObjectName("Form")
        Form.resize(976, 587)
        Form.setStyleSheet("background-color: rgb(255, 170, 127);")
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setEnabled(True)
        self.frame.setStyleSheet("background-color: rgb(255, 170, 127);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.line_2 = QtWidgets.QFrame(self.frame)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_5.addWidget(self.line_2, 1, 1, 1, 1)
        self.frame_4 = QtWidgets.QFrame(self.frame)
        self.frame_4.setMaximumSize(QtCore.QSize(16777215, 100))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.line = QtWidgets.QFrame(self.frame_4)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_4.addWidget(self.line, 2, 0, 1, 2)
        self.label = QtWidgets.QLabel(self.frame_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 0, 0, 1, 2)
        self.gridLayout_5.addWidget(self.frame_4, 0, 0, 1, 3)
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setMaximumSize(QtCore.QSize(250, 16777215))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.btnExit = QtWidgets.QPushButton(self.frame_2)
        # button clicked
        self.btnExit.clicked.connect(self.btnExit_clicked)
        self.btnExit.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.btnExit.setFont(font)
        self.btnExit.setObjectName("btnExit")
        self.gridLayout_2.addWidget(self.btnExit, 3, 0, 1, 1)
        self.btnDiagnose = QtWidgets.QPushButton(self.frame_2)
        # btn action
        self.btnDiagnose.clicked.connect(self.openDiagnosisWindow)
        self.btnDiagnose.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.btnDiagnose.setFont(font)
        self.btnDiagnose.setObjectName("btnDiagnose")
        self.gridLayout_2.addWidget(self.btnDiagnose, 1, 0, 1, 1)
        self.btnDashboard = QtWidgets.QPushButton(self.frame_2)
        # btn action
        self.btnDashboard.clicked.connect(self.openDashboard)
        self.btnDashboard.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.btnDashboard.setFont(font)
        self.btnDashboard.setObjectName("btnDashboard")
        self.gridLayout_2.addWidget(self.btnDashboard, 0, 0, 1, 1)
        self.btnViewRecs = QtWidgets.QPushButton(self.frame_2)
        self.btnViewRecs.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.btnViewRecs.setFont(font)
        self.btnViewRecs.setObjectName("btnViewRecs")
        self.gridLayout_2.addWidget(self.btnViewRecs, 2, 0, 1, 1)
        self.gridLayout_5.addWidget(self.frame_2, 1, 0, 1, 1)
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setMaximumSize(QtCore.QSize(750, 400))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_3 = QtWidgets.QLabel(self.frame_3)
        self.label_3.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 0, 0, 1, 1)
        self.frame_5 = QtWidgets.QFrame(self.frame_3)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.frame_5)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.tableWidget = QtWidgets.QTableWidget(self.frame_5)
        self.tableWidget.setRowCount(10)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(8)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(7, item)
        self.gridLayout_6.addWidget(self.tableWidget, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.frame_5, 1, 0, 1, 1)
        self.gridLayout_5.addWidget(self.frame_3, 1, 2, 1, 1)
        self.line_2.raise_()
        self.frame_2.raise_()
        self.frame_3.raise_()
        self.frame_4.raise_()
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)


    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Detection of Diabetic Retinopathy and Classification of its Severity"))
        self.label.setText(_translate("Form", "Detection of Diabetic Retinopathy and Classification of its Severity"))
        self.btnExit.setText(_translate("Form", "Exit"))
        self.btnDiagnose.setText(_translate("Form", "Diagnosis"))
        self.btnDashboard.setText(_translate("Form", "Dashboard"))
        self.btnViewRecs.setText(_translate("Form", "View Previous Records"))
        self.label_3.setText(_translate("Form", "View Previous Records"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Form", "Patient ID"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("Form", "Name"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("Form", "Gender"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("Form", "Age"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("Form", "Image Name"))
        item = self.tableWidget.horizontalHeaderItem(5)
        item.setText(_translate("Form", "DR Status"))
        item = self.tableWidget.horizontalHeaderItem(6)
        item.setText(_translate("Form", "DR Grade"))
        item = self.tableWidget.horizontalHeaderItem(7)
        item.setText(_translate("Form", "Severity"))

    def openDashboard(self):
        self.window = QtWidgets.QMainWindow()
        from src.Ui_MainWindow import Ui_MainWindow
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.window)
        self.window.show()

    def openDiagnosisWindow(self):
        self.window = QtWidgets.QWidget()
        from src.DiagnosisWindow import DiagnosisWindow
        self.ui = DiagnosisWindow()
        self.ui.setupUi(self.window)
        self.window.show()

    def btnExit_clicked(self):
        QtCore.QCoreApplication.instance().quit()

    def loadData(self):
        df = pd.read_csv('data.csv')
        # removing unwanted columns
        df = df.drop("Unnamed: 0", axis=1)
        df = df.drop("density_of_blood_vessels", axis=1)
        df = df.drop("no_of_microaneurysms", axis=1)
        df = df.drop("no_of_haemorrhages", axis=1)

        # headers = list(df)
        self.tableWidget.setRowCount(df.shape[0])
        self.tableWidget.setColumnCount(df.shape[1])
        # self.tableWidget.setHorizontalHeaderLabels(headers)

        # populate data from csv file to table widget
        # getting data from df is computationally costly so convert it to array first
        df_array = df.values
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                self.tableWidget.setItem(row, col, QtWidgets.QTableWidgetItem(str(df_array[row, col])))

    def main(self):
        import sys

        app = QtWidgets.QApplication(sys.argv)
        Form = QtWidgets.QWidget()
        ui = ViewRecordsWindow()
        ui.setupUi(Form)
        ui.loadData()
        Form.show()

        sys.exit(app.exec_())

if __name__ == "__main__":
    ui = ViewRecordsWindow()
    ui.main()
    # import sys
    #
    # app = QtWidgets.QApplication(sys.argv)
    # Form = QtWidgets.QWidget()
    # ui = ViewRecordsWindow()
    # ui.setupUi(Form)
    # ui.loadData()
    # Form.show()

    # sys.exit(app.exec_())

