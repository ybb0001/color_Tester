#pragma once

#include <QtWidgets/QWidget>
#include "ui_color_Tester.h"
#include <QString>
#include <QFileDialog>
#include <QMessageBox>
#include <string>

#include "core.hpp"
#include "highgui.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class color_Tester : public QWidget
{
	Q_OBJECT

public:
	color_Tester(QWidget *parent = Q_NULLPTR);

	int Deco_Color_Test();
	void displayResult();
	void display_Image();



public slots:
	void on_pushButton_cnt_reduce_clicked();
	void on_pushButton_clear_clicked();
	void on_pushButton_Color_Diff_clicked();

private:
	Ui::color_TesterClass ui;
	QImage img;
	QImage imgScaled;

	Mat image, img2, img3, dst, bin;
	Mat imageCopy, normImage, g_srcImage1;
	Mat temp_image;
	Mat gray_image;

};
