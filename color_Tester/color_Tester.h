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

#include "ChessBoard_CamCalibration.h"
#include "ChessBoard_CornerDetection.h"
#include "SFR_Calculator.h"

using namespace cv;
using namespace std;

#define PYTHON_DLL

class color_Tester : public QWidget
{
	Q_OBJECT

public:
	color_Tester(QWidget *parent = Q_NULLPTR);

	int Deco_Color_Test();
	void displayResult();
	void display_Image();

	void ROISelection(cv::Mat& img, cv::Mat& roi, cv::Rect& ROIBox);


public slots:
	void on_pushButton_cnt_reduce_clicked();
	void on_pushButton_clear_clicked();
	void on_pushButton_Color_Diff_clicked();
	void on_pushButton_Iris_clicked();

	
	void on_pushButton_raw2bmp_clicked();
	void on_pushButton_PIMA_clicked();
	void on_pushButton_PIMA_2_clicked();
	void on_pushButton_open_raw_clicked();
	void on_pushButton_open_bmp_clicked();


	void on_pushButton_Xiaomi_SFR_clicked();
	void on_pushButton_SFR_ROI4_clicked();



private:
	Ui::color_TesterClass ui;
	QImage img;
	QImage imgScaled;

	Mat image, img2, img3, dst, bin;
	Mat imageCopy, normImage, g_srcImage1;
	Mat temp_image;
	Mat gray_image;

	string name, sub_name;
};


struct POLAR_COORDINATE {
	double field;
	double angle;
};

struct SFRVALUE {
	double freqDivide;
	double HValue;
	double VValue;
};

struct SFR_RESULT_LIST {
	int errH;
	int errV;
	POLAR_COORDINATE spot_Polar;
	std::vector<SFRVALUE> SFRValue;
	CvPoint spot_Rect;
	std::vector<CvRect> ROIs;
};

struct INPUTIMG_INFO {
	unsigned char* bmpData;
	unsigned char* rawDate;
	char* pattern;
	int width;
	int height;
};

