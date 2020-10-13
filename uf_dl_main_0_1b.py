# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dl_frame.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import dl_functions as dlf


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.radioButton = QtWidgets.QRadioButton(Dialog)
        self.radioButton_2 = QtWidgets.QRadioButton(Dialog)
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.spinBox = QtWidgets.QSpinBox(self.groupBox)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(Dialog)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox_2)
        self.toolButton = QtWidgets.QToolButton(self.groupBox_2)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        Dialog.setObjectName("Dialog")
        Dialog.resize(226, 202)
        Dialog.setMaximumSize(QtCore.QSize(1000, 300))
        Dialog.setStyleSheet("border-color: rgb(255, 19, 19);")
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.radioButton.setMinimumSize(QtCore.QSize(100, 15))
        self.radioButton.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.radioButton.setChecked(True)
        self.radioButton.setAutoExclusive(True)
        self.radioButton.setObjectName("radioButton")
        self.horizontalLayout.addWidget(self.radioButton)
        self.radioButton_2.setMinimumSize(QtCore.QSize(100, 15))
        self.radioButton_2.setObjectName("radioButton_2")
        self.horizontalLayout.addWidget(self.radioButton_2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.groupBox.setMinimumSize(QtCore.QSize(0, 70))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout.setObjectName("gridLayout")
        self.checkBox.setObjectName("checkBox")
        self.gridLayout.addWidget(self.checkBox, 0, 0, 1, 1)
        self.label_4.setAlignment(QtCore.Qt.AlignRight |
                                  QtCore.Qt.AlignTrailing |
                                  QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 1, 1, 1)
        self.spinBox.setMaximumSize(QtCore.QSize(50, 16777215))
        self.spinBox.setObjectName("spinBox")
        self.gridLayout.addWidget(self.spinBox, 0, 2, 1, 1)
        self.label_3.setAlignment(QtCore.Qt.AlignRight |
                                  QtCore.Qt.AlignTrailing |
                                  QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 2)
        self.doubleSpinBox.setMaximumSize(QtCore.QSize(50, 16777215))
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.gridLayout.addWidget(self.doubleSpinBox, 1, 2, 1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_2.setMinimumSize(QtCore.QSize(0, 50))
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_2.addWidget(self.lineEdit_2)
        self.toolButton.setObjectName("toolButton")
        self.horizontalLayout_2.addWidget(self.toolButton)
        self.verticalLayout.addWidget(self.groupBox_2)
        spacerItem = QtWidgets.QSpacerItem(20, 40,
                                           QtWidgets.QSizePolicy.Minimum,
                                           QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "DL V: 0.1.b"))
        self.radioButton.setText(_translate("Dialog", "Symmetric"))
        self.radioButton_2.setText(_translate("Dialog", "Asymmetric"))
        self.groupBox.setTitle(_translate("Dialog", "Butterworth filter"))
        self.checkBox.setText(_translate("Dialog", "Enabled"))
        self.label_4.setText(_translate("Dialog", "Order:"))
        self.label_3.setText(_translate("Dialog", "Cutt of frequency:"))
        self.groupBox_2.setTitle(_translate("Dialog", "Path to *.dl:"))
        self.toolButton.setText(_translate("Dialog", "..."))
        self.spinBox.setValue(3)
        self.doubleSpinBox.setValue(30.0)

        # self.toolButton.clicked.connect(self.temp)

    def temp(self):
        texta = dlf.dialog_open_file()
        self.lineEdit_2.setText(texta)


# def initialize():
if __name__ == "__main__":

    # Create app
    app = QtWidgets.QApplication(sys.argv)


    # init
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    Dialog.setFocus(True)


    # Logic
    def dialog_to_text(text_object):
        def text_func():
            tmp_text = dlf.dialog_open_file()
            text_object.setText(tmp_text)
        return text_func


    def par_run():
        import DL_main

        params = {}
        params['run'] = 1
        params['butfilt_order'] = ui.spinBox.value()
        params['butfilt_cuttoff'] = ui.doubleSpinBox.value()
        params['simetric'] = ui.radioButton.isChecked()
        params['dl_path'] = ui.lineEdit_2.text()

        DL_main.main(params)

    ui.toolButton.clicked.connect(dialog_to_text(ui.lineEdit_2))
    ui.buttonBox.accepted.connect(par_run)

    # Main Loop
    sys.exit(app.exec_())
