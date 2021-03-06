# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

from src.DiagnosisWindow import DiagnosisWindow
from src.ViewRecordsWindow import ViewRecordsWindow


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(976, 587)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("background-color: rgb(255, 170, 127);")
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setEnabled(True)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setMaximumSize(QtCore.QSize(250, 16777215))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.btnExit = QtWidgets.QPushButton(self.frame_2)

        # exit button clicked event
        self.btnExit.clicked.connect(self.btnExit_clicked)

        self.btnExit.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.btnExit.setFont(font)
        self.btnExit.setObjectName("pushButton")
        self.gridLayout_2.addWidget(self.btnExit, 3, 0, 1, 1)
        self.btnDiagnose = QtWidgets.QPushButton(self.frame_2)
        # btn action
        self.btnDiagnose.clicked.connect(self.openDiagnoseWindow)
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
        # btn action
        self.btnViewRecs.clicked.connect(self.openViewRecordsWindow)
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
        self.label_2 = QtWidgets.QLabel(self.frame_3)
        self.label_2.setStyleSheet("background-image: url(:/images/Diabetic_Retinopathy.jpg);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 1, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.frame_3)
        self.label_3.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 0, 1, 1, 1)
        self.gridLayout_5.addWidget(self.frame_3, 1, 2, 1, 1)
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
        self.line_2 = QtWidgets.QFrame(self.frame)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_5.addWidget(self.line_2, 1, 1, 1, 1)
        self.line_2.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.label_3.raise_()
        self.line.raise_()
        self.frame_2.raise_()
        self.frame_3.raise_()
        self.frame_4.raise_()
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Detection of Diabetic Retinopathy and Classification of its Severity"))
        self.btnExit.setText(_translate("MainWindow", "Exit"))
        self.btnDiagnose.setText(_translate("MainWindow", "Diagnosis"))
        self.btnDashboard.setText(_translate("MainWindow", "Dashboard"))
        self.btnViewRecs.setText(_translate("MainWindow", "View Previous Records"))
        self.label_3.setText(_translate("MainWindow", "Dashboard"))
        self.label.setText(_translate("MainWindow", "Detection of Diabetic Retinopathy and Classification of its Severity"))

    def openDiagnoseWindow(self):
        self.window = QtWidgets.QWidget()
        self.ui = DiagnosisWindow()
        self.ui.setupUi(self.window)
        # MainWindow.close()
        self.window.show()

    def openViewRecordsWindow(self):
        self.window = QtWidgets.QWidget()
        self.ui = ViewRecordsWindow()
        self.ui.setupUi(self.window)
        self.window.show()
        # MainWindow.close()
        self.ui.loadData()

    def btnExit_clicked(self):
        QtCore.QCoreApplication.instance().quit()


import images_rc

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

