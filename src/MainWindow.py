from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QMainWindow, QWidget
import sys
from PyQt5.QtGui import QPixmap


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.title = "Detection of Diabetic Retinopathy"
#         self.top = 500
#         self.left = 200
#         self.width = 1000
#         self.height = 400
#         self.iconName = "icon.png"
#         self.InitWindow()
#
#     def InitWindow(self):
#         self.setWindowIcon(QtGui.QIcon(self.iconName))
#         self.setWindowTitle(self.title)
#         self.setGeometry(self.left, self.top, self.width, self.height)
#         vbox = QVBoxLayout()
#         labelImage = QLabel(self)
#         pixmap = QPixmap("Diabetic_retinopathy.jpg")
#         labelImage.setPixmap(pixmap)
#         vbox.addWidget(labelImage)
#         self.setLayout(vbox)
#         self.show()
#
# if __name__ == "__main__":
#     App = QApplication(sys.argv)
#     window = MainWindow()
#     sys.exit(App.exec())


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Detection of Diabetic Retinopathy'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create widget
        label = QLabel(self)
        pixmap = QPixmap('Diabetic_retinopathy.jpg')
        label.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())

        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())