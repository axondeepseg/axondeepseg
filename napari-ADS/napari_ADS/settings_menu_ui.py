# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'settings_menu_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from qtpy import QtCore, QtGui, QtWidgets


class Ui_Settings_menu_ui(object):
    def setupUi(self, Settings_menu_ui):
        Settings_menu_ui.setObjectName("Settings_menu_ui")
        Settings_menu_ui.setWindowModality(QtCore.Qt.WindowModal)
        Settings_menu_ui.resize(399, 209)
        self.verticalLayout = QtWidgets.QVBoxLayout(Settings_menu_ui)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_5 = QtWidgets.QLabel(Settings_menu_ui)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_6.addWidget(self.label_5)
        self.gpu_id_spinBox = QtWidgets.QSpinBox(Settings_menu_ui)
        self.gpu_id_spinBox.setObjectName("gpu_id_spinBox")
        self.horizontalLayout_6.addWidget(self.gpu_id_spinBox)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.no_patch_checkBox = QtWidgets.QCheckBox(Settings_menu_ui)
        self.no_patch_checkBox.setObjectName("no_patch_checkBox")
        self.horizontalLayout_5.addWidget(self.no_patch_checkBox)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(Settings_menu_ui)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.axon_shape_comboBox = QtWidgets.QComboBox(Settings_menu_ui)
        self.axon_shape_comboBox.setObjectName("axon_shape_comboBox")
        self.axon_shape_comboBox.addItem("")
        self.axon_shape_comboBox.addItem("")
        self.horizontalLayout_4.addWidget(self.axon_shape_comboBox)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(Settings_menu_ui)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.overlap_value_spinBox = QtWidgets.QSpinBox(Settings_menu_ui)
        self.overlap_value_spinBox.setMaximum(1000)
        self.overlap_value_spinBox.setProperty("value", 50)
        self.overlap_value_spinBox.setObjectName("overlap_value_spinBox")
        self.horizontalLayout_2.addWidget(self.overlap_value_spinBox)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QtWidgets.QLabel(Settings_menu_ui)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.zoom_factor_spinBox = QtWidgets.QDoubleSpinBox(Settings_menu_ui)
        self.zoom_factor_spinBox.setDecimals(4)
        self.zoom_factor_spinBox.setSingleStep(0.05)
        self.zoom_factor_spinBox.setStepType(QtWidgets.QAbstractSpinBox.DefaultStepType)
        self.zoom_factor_spinBox.setProperty("value", 1.0)
        self.zoom_factor_spinBox.setObjectName("zoom_factor_spinBox")
        self.horizontalLayout.addWidget(self.zoom_factor_spinBox)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.done_button = QtWidgets.QPushButton(Settings_menu_ui)
        self.done_button.setObjectName("done_button")
        self.verticalLayout.addWidget(self.done_button)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)

        self.retranslateUi(Settings_menu_ui)
        QtCore.QMetaObject.connectSlotsByName(Settings_menu_ui)

    def retranslateUi(self, Settings_menu_ui):
        _translate = QtCore.QCoreApplication.translate
        Settings_menu_ui.setWindowTitle(_translate("Settings_menu_ui", "Settings menu"))
        self.label_5.setText(_translate("Settings_menu_ui", "GPU ID"))
        self.no_patch_checkBox.setText(_translate("Settings_menu_ui", "No patch"))
        self.label_4.setText(_translate("Settings_menu_ui", "Axon Shape"))
        self.axon_shape_comboBox.setItemText(0, _translate("Settings_menu_ui", "circle"))
        self.axon_shape_comboBox.setItemText(1, _translate("Settings_menu_ui", "ellipse"))
        self.label.setText(_translate("Settings_menu_ui", "Overlap Value"))
        self.label_3.setText(_translate("Settings_menu_ui", "Zoom factor"))
        self.done_button.setText(_translate("Settings_menu_ui", "Done"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Settings_menu_ui = QtWidgets.QWidget()
    ui = Ui_Settings_menu_ui()
    ui.setupUi(Settings_menu_ui)
    Settings_menu_ui.show()
    sys.exit(app.exec_())
