#include "color_Tester.h"
#include <afx.h>
#include <Windows.h>
#include <direct.h>
#include <String>
#include <direct.h>
#include <stdlib.h>
#include <stdio.h>
#include <io.h>
#include <iostream>     
#include <fstream>   
#include <QTextCodec>
#include <math.h>
#include <tchar.h>
#include <shellapi.h>
#include <cmath>
#include <algorithm> 


#ifdef  PYTHON_DLL
#include <Python.h>
PyObject * pModule = NULL;//声明变量
PyObject * pFunc1 = NULL;// 声明变量
PyObject * pFunc2 = NULL;// 声明变量
PyObject * pFunc3 = NULL;// 声明变量
PyObject * mBestModel = NULL;


#endif

int NG = 0, Color_Tcnt = 0, PIMA[9][2] = { 0 };
int  bayer_pattern = 0, raw_width = 4000, raw_height = 3000;
ofstream fout;
unsigned int Width = 0, Height = 0;


color_Tester::color_Tester(QWidget *parent)
	: QWidget(parent)
{
	ui.setupUi(this);
	//connect(ui.pushButton_clear, SIGNAL(clicked()), this, SLOT(on_pushButton_clear_clicked()));
	//connect(ui.pushButton_Color_Diff, SIGNAL(clicked()), this, SLOT(on_pushButton_Color_Diff_clicked()));
	//connect(ui.pushButton_cnt_reduce, SIGNAL(clicked()), this, SLOT(on_pushButton_cnt_reduce_clicked()));


#ifdef  PYTHON_DLL
	Py_Initialize();//使用python之前，要调用Py_Initialize();这个函数进行初始化

	if (!Py_IsInitialized())	{
		ui.log->setText("Python Initialize Fail!\n");
	}
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./')");//这一步很重要，修改Python路径

#if 0
	pModule = PyImport_ImportModule("color_test");//这里是要调用的文件名hello.py
	if (pModule == NULL)	{
		ui.log->setText("Python pModule Initialize Fail!\n");
	}

	pFunc3 = PyObject_GetAttrString(pModule, "color_check");//这里是要调用的函数名

	if (!pFunc3 || !PyCallable_Check(pFunc3)) {
		ui.log->setText("Python PyObject Initialize Fail!\n");
	}
#else
	
	pModule = PyImport_ImportModule("Iris_Predict");//这里是要调用的文件名hello.py
	if (pModule == NULL) {
		ui.log->setText("Python pModule Initialize Fail!\n");
	}

	pFunc1 = PyObject_GetAttrString(pModule, "mLoadModel");//这里是要调用的函数名
	if (!pFunc1 || !PyCallable_Check(pFunc1)) {
		ui.log->setText("Python PyObject Initialize Fail!\n");
	}

	pFunc2 = PyObject_GetAttrString(pModule, "mModel_predict");//这里是要调用的函数名
	if (!pFunc2 || !PyCallable_Check(pFunc2)) {
		ui.log->setText("Python PyObject Initialize Fail!\n");
	}

#if 1
	PyObject* args1 = Py_BuildValue("ii", 10, 1);//给python函数参数赋值
	mBestModel = PyObject_CallObject(pFunc1, args1);//调用函数PyObject_CallFunctionObjArgs 
	if (mBestModel == NULL) {
		PyErr_Print();  // 打印错误信息
		ui.log->setText("Call Python Function mAdd2 failed!\n");
		return;
	}
	if (PyErr_Occurred()) {
		PyErr_Print();
		ui.log->setText("Python error occurred!\n");
		return;
	}

#endif

#endif

#endif

	raw_width = GetPrivateProfileInt(TEXT("TEST_OPTION"), TEXT("raw_width"), 4096, TEXT(".\\Setting\\specValue.ini"));
	raw_height = GetPrivateProfileInt(TEXT("TEST_OPTION"), TEXT("raw_height"), 3072, TEXT(".\\Setting\\specValue.ini"));
	bayer_pattern = GetPrivateProfileInt(TEXT("TEST_OPTION"), TEXT("bayer_pattern"), 0, TEXT(".\\Setting\\specValue.ini"));

	ui.width->setText(QString::number(raw_width, 10));
	ui.height->setText(QString::number(raw_height, 10));

	if (bayer_pattern == 0)
		ui.RG->setChecked(true);
	if (bayer_pattern == 1)
		ui.BG->setChecked(true);
	if (bayer_pattern == 2)
		ui.GR->setChecked(true);
	if (bayer_pattern == 3)
		ui.GB->setChecked(true);

	PIMA[3][0] = GetPrivateProfileInt(TEXT("PIMA"), TEXT("3F_Min"), 1800, TEXT(".\\Setting\\specValue.ini"));
	PIMA[3][1] = GetPrivateProfileInt(TEXT("PIMA"), TEXT("3F_Max"), 2800, TEXT(".\\Setting\\specValue.ini"));

	PIMA[5][0] = GetPrivateProfileInt(TEXT("PIMA"), TEXT("5F_Min"), 1800, TEXT(".\\Setting\\specValue.ini"));
	PIMA[5][1] = GetPrivateProfileInt(TEXT("PIMA"), TEXT("5F_Max"), 2800, TEXT(".\\Setting\\specValue.ini"));

	PIMA[7][0] = GetPrivateProfileInt(TEXT("PIMA"), TEXT("7F_Min"), 1600, TEXT(".\\Setting\\specValue.ini"));
	PIMA[7][1] = GetPrivateProfileInt(TEXT("PIMA"), TEXT("7F_Max"), 3400, TEXT(".\\Setting\\specValue.ini"));

	PIMA[8][0] = GetPrivateProfileInt(TEXT("PIMA"), TEXT("8F_Min"), 1600, TEXT(".\\Setting\\specValue.ini"));
	PIMA[8][1] = GetPrivateProfileInt(TEXT("PIMA"), TEXT("8F_Max"), 3400, TEXT(".\\Setting\\specValue.ini"));

}

void color_Tester::on_pushButton_clear_clicked() {
	ui.log->clear();
}

void color_Tester::displayResult() {

	ui.log->setFontPointSize(12);
	if (NG == 1) {
		ui.log->setTextColor(QColor(255, 0, 0, 255));
		ui.log->insertPlainText("NG");
	}
	else if (NG == 2) {
		ui.log->setTextColor(QColor(0, 255, 0, 255));
		ui.log->insertPlainText("OK");
	}
	else if (NG == 0) {
		ui.log->setTextColor(QColor(0, 0, 0, 255));
		ui.log->setText("NA");
	}
	ui.log->setAlignment(Qt::AlignCenter);
}

void color_Tester::display_Image() {

	//displayResult();
	cvtColor(image, dst, CV_BGR2RGB);
	QImage showImage((const uchar*)dst.data, dst.cols, dst.rows, dst.cols*dst.channels(), QImage::Format_RGB888);
	imgScaled = showImage.scaled(ui.label_show_image->size(), Qt::KeepAspectRatio);
	ui.label_show_image->setPixmap(QPixmap::fromImage(imgScaled));

}

vector<string> getFiles(string cate_dir)
{
	vector<string> files;//存放文件名
	_finddata_t file;
	long lf = 0;
	//输入文件夹路径
	if ((lf = _findfirst(cate_dir.c_str(), &file)) == -1) {
		cout << cate_dir << " not found!!!" << endl;
	}
	else {
		while (_findnext(lf, &file) == 0) {
			//输出文件名
			//cout<<file.name<<endl;
			if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0)
				continue;
			files.push_back(file.name);
		}
	}
	_findclose(lf);

	//排序，按从小到大排序
	sort(files.begin(), files.end());
	return files;
}

string TCHAR2STRING(TCHAR* str){
	std::string strstr;
	try	{
		int iLen = WideCharToMultiByte(CP_ACP, 0, str, -1, NULL, 0, NULL, NULL);

		char* chRtn = new char[iLen * sizeof(char)];

		WideCharToMultiByte(CP_ACP, 0, str, -1, chRtn, iLen, NULL, NULL);

		strstr = chRtn;
	}
	catch (std::exception e)
	{
	}

	return strstr;
}

void color_Tester::on_pushButton_Color_Diff_clicked() {
	ui.log->clear();

	TCHAR lpTexts[256] = { 0 };
	string dino_path = "";

	GetPrivateProfileString(TEXT("TEST_OPTION"), TEXT("path"), TEXT(""), lpTexts, 255, TEXT(".\\Setting\\specValue.ini"));
	dino_path = TCHAR2STRING(lpTexts);
	if (dino_path.length() < 5)return;

	vector<string> files = getFiles(dino_path + "/*");
	vector<string> ::iterator iVector = files.begin();
	while (iVector != files.end())
	{
		if ((*iVector).length()>4) {
			string sub_name = dino_path + "/" + *iVector;
			int len = sub_name.length();
			bool isImg = false;

			if (sub_name[len - 3] == 'j'&&sub_name[len - 2] == 'p')isImg = true;
			if (sub_name[len - 4] == 'j'&&sub_name[len - 3] == 'p')isImg = true;
			if (sub_name[len - 3] == 'b'&&sub_name[len - 2] == 'm')isImg = true;

			if (isImg) {
				image = imread(sub_name);
				Deco_Color_Test();
				break;
			}
		}
		++iVector;
	}
	//RemoveAllFileInFolder(dino_path);

	Color_Tcnt++;
	ui.textBrowser_cnt->setFontPointSize(48);
	ui.textBrowser_cnt->setText(QString::number(Color_Tcnt));
	ui.textBrowser_cnt->setAlignment(Qt::AlignCenter);
}

int color_Tester::Deco_Color_Test() {

	imageCopy = image.clone();
	img2 = imageCopy(Rect(image.rows / 8, image.rows / 3, image.rows / 16, image.rows / 3));
	//imwrite("roi2.bmp", img2);
	rectangle(image, Rect(image.rows / 8, image.rows / 3, image.rows / 16, image.rows / 3), Scalar(0, 0, 255), 2, 8, 0);

	float BGR_avg[3] = { 0 };
	cvtColor(img2, gray_image, CV_BGR2GRAY);
	Scalar m = mean(gray_image);
	int BV_low = m[0] * 0.8, BV_up = m[0] * 1.2, p_cnt = 0;

	for (int i = 0; i < img2.rows; i++) {
		const uchar* inData = img2.ptr<uchar>(i);
		const uchar* inData2 = gray_image.ptr<uchar>(i);
		for (int j = 0; j < img2.cols; j++) {
			if (inData2[j] > BV_low&&inData2[j] < BV_up) {
				p_cnt++;
				for (int k = 0; k < 3; k++) {
					BGR_avg[k] += inData[3 * j + k];
				}
			}
		}
	}
	float lrud[4][3] = { 0 };

	for (int i = 0; i < img2.rows; i++) {
		const uchar* inData = img2.ptr<uchar>(i);
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 3; k++) {
				lrud[0][k] += inData[3 * j + k];
			}
		}
		for (int j = img2.cols - 2; j < img2.cols; j++) {
			for (int k = 0; k < 3; k++) {
				lrud[1][k] += inData[3 * j + k];
			}
		}
	}

	for (int i = 0; i < 2; i++) {
		const uchar* inData = img2.ptr<uchar>(i);
		for (int j = 0; j < img2.cols; j++) {
			for (int k = 0; k < 3; k++) {
				lrud[2][k] += inData[3 * j + k];
			}
		}
	}
	for (int i = img2.rows - 2; i < img2.rows; i++) {
		const uchar* inData = img2.ptr<uchar>(i);
		for (int j = 0; j < img2.cols; j++) {
			for (int k = 0; k < 3; k++) {
				lrud[3][k] += inData[3 * j + k];
			}
		}
	}

	for (int k = 0; k < 3; k++) {
		lrud[0][k] /= img2.rows * 2;
		lrud[1][k] /= img2.rows * 2;
		lrud[2][k] /= img2.cols * 2;
		lrud[3][k] /= img2.cols * 2;
	}


	for (int k = 0; k < 3; k++) {
		BGR_avg[k] /= p_cnt;
	}
	if (BGR_avg[1] > 20) {
		for (int m = 0; m < 4; m++) {
			for (int k = 0; k < 3; k++) {
				if (lrud[m][k] < BGR_avg[k] * 0.4 || lrud[m][k] > BGR_avg[k] * 1.7) {
					ui.log->insertPlainText("ROI error\n");
					display_Image();
					return -1;
				}
			}
		}
	}
	float RG = BGR_avg[2] / BGR_avg[1], BG = BGR_avg[0] / BGR_avg[1];

	if (BGR_avg[0] < 32 && BGR_avg[0] < 32 && BGR_avg[0] < 32) {
		NG = 0;
	}
	else if (RG>1.3) {
		NG = 1;
	}
	else if (BG>1.2) {
		if (BG>2.5)	NG = 1;
		if (BG>1.5&&RG>1.5)	NG = 1;
	}
	else if (BG<0.8&&RG<0.8) {
		NG = 1;
	}

	double res = 2;

#ifdef  PYTHON_DLL

	PyObject* args2 = Py_BuildValue("ddd", BGR_avg[2], BGR_avg[1], BGR_avg[0]);//给python函数参数赋值
	PyObject* pRet = PyObject_CallObject(pFunc3, args2);//调用函数
	PyArg_Parse(pRet, "d", &res);//转换返回类型
	if (res>0.5)NG = 1;
	else NG = 2;
	if (res == 0) {
		NG = 0;
		ui.log->setTextColor(QColor(255, 0, 0, 255));
		ui.log->insertPlainText("Model result is invalid!\n");
	}
	else {
		ui.log->setTextColor(QColor(255, 255, 255, 255));
		string str = "Model result is :	" + to_string(res) + '\n';
		ui.log->insertPlainText(str.c_str());
	}
#endif
	if (fout.is_open()) {
		fout << BGR_avg[2] << "	" << BGR_avg[1] << "	" << BGR_avg[0] << "	";
	}
	else {
		ui.log->setFontPointSize(8);
		ui.log->setTextColor(QColor(0, 0, 0, 255));
		string str = "R:	" + to_string(BGR_avg[2]) + '\n';
		ui.log->insertPlainText(str.c_str());
		str = "G:	" + to_string(BGR_avg[1]) + '\n';
		ui.log->insertPlainText(str.c_str());
		str = "B:	" + to_string(BGR_avg[0]) + '\n';
		ui.log->insertPlainText(str.c_str());
		str = "RG:	" + to_string(RG) + '\n';
		ui.log->insertPlainText(str.c_str());
		str = "BG:	" + to_string(BG) + '\n';
		ui.log->insertPlainText(str.c_str());
		display_Image();

	}
	displayResult();
	return res;
}

void color_Tester::on_pushButton_cnt_reduce_clicked() {

	Color_Tcnt--;
	if (Color_Tcnt < 0)Color_Tcnt = 0;
	ui.textBrowser_cnt->setFontPointSize(48);
	ui.textBrowser_cnt->setText(QString::number(Color_Tcnt));
	ui.textBrowser_cnt->setAlignment(Qt::AlignCenter);

}

void color_Tester::on_pushButton_raw2bmp_clicked() {

	
	QString filename = QFileDialog::getOpenFileName(this, tr("Open Image"), "", tr("Image File(*.raw)"));
	QTextCodec *code = QTextCodec::codecForName("gb18030");
	name = code->fromUnicode(filename).data();


	FILE *fp = NULL;

	if (raw_width*raw_height == 0) {
		QMessageBox msgBox;
		msgBox.setText(tr("Plz input Img_width and Img_height value"));
		msgBox.exec();
		return;
	}
	unsigned short *pRawData = (unsigned short *)calloc(raw_width*raw_height, sizeof(unsigned short));

	if (NULL == pRawData)
	{
		QMessageBox msgBox;
		msgBox.setText(tr("Fail to calloc buf"));
		msgBox.exec();
		return;
	}

	ifstream in(name.c_str());
	in.seekg(0, ios::end); //设置文件指针到文件流的尾部
	streampos ps = in.tellg(); //读取文件指针的位置
	in.close(); //关闭文件流

	if (NULL == (fp = fopen(name.c_str(), "rb")))
	{
		QMessageBox msgBox;
		msgBox.setText(tr("Fail to read"));
		msgBox.exec();
		return;
	}

	if (raw_width*raw_height * 2 != ps)
	{
		QMessageBox msgBox;
		msgBox.setText(tr("Width * Height Size does not match Raw Size!"));
		msgBox.exec();
		return;
	}

	int ret = fread(pRawData, sizeof(unsigned short)*raw_width*raw_height, 1, fp);
	IplImage *pBayerData = cvCreateImage(cvSize(raw_width, raw_height), 16, 1);
	IplImage *pRgbDataInt8 = cvCreateImage(cvSize(raw_width, raw_height), 8, 1);

	memcpy(pBayerData->imageData, (char *)pRawData, raw_width*raw_height*sizeof(unsigned short));
	free(pRawData);

	cvConvertScale(pBayerData, pRgbDataInt8, 0.25, 0);
	temp_image = cvarrToMat(pRgbDataInt8);

	cvtColor(temp_image, temp_image, CV_BayerGB2BGR);

	imwrite(name+"raw2bmp.bmp", temp_image);
	string str = " raw to BMP done\n";
	ui.log->insertPlainText(str.c_str());
}

void ROImouseEvent(int event, int x, int y, int flag, void* params)
{
	cv::Point* ptr = (cv::Point*)params;

	if (event == CV_EVENT_LBUTTONDOWN && ptr[0].x == -1 && ptr[0].y == -1)
	{
		ptr[0].x = x;
		ptr[0].y = y;
	}

	if (flag == CV_EVENT_FLAG_LBUTTON)
	{
		ptr[1].x = x;
		ptr[1].y = y;
	}

	if (event == CV_EVENT_LBUTTONUP && ptr[2].x == -1 && ptr[2].y == -1)
	{
		ptr[2].x = x;
		ptr[2].y = y;
	}

}

void color_Tester::ROISelection(cv::Mat& img, cv::Mat& roi, cv::Rect& ROIBox)
{
	cv::Point* Corners = new cv::Point[3];
	Corners[0].x = Corners[0].y = -1;
	Corners[1].x = Corners[1].y = -1;
	Corners[2].x = Corners[2].y = -1;

	cv::namedWindow("ROI select(Press Esc to close window)", CV_WINDOW_NORMAL);
	cv::imshow("ROI select(Press Esc to close window)", img);

	bool downFlag = false, upFlag = false;
	while (cv::waitKey(1) != 27||roi.empty())
	{
		cv::setMouseCallback("ROI select(Press Esc to close window)", ROImouseEvent, Corners);

		if (Corners[0].x != -1 && Corners[0].y != -1) { downFlag = true; }
		if (Corners[2].x != -1 && Corners[2].y != -1) { upFlag = true; }

		if (downFlag && !upFlag && Corners[1].x != -1)
		{
			cv::Mat LocalImage = img.clone();
			cv::rectangle(LocalImage, Corners[0], Corners[1], cv::Scalar(255, 255, 255), 8);
			cv::imshow("ROI select(Press Esc to close window)", LocalImage);
		}

		if (downFlag && upFlag)
		{
			//cv::Rect ROIBox;
			ROIBox.width = abs(Corners[0].x - Corners[2].x);
			ROIBox.height = abs(Corners[0].y - Corners[2].y);

			if (ROIBox.width < 32 && ROIBox.height < 32)
			{
				Corners[0].x = Corners[0].y = -1;
				Corners[1].x = Corners[1].y = -1;
				Corners[2].x = Corners[2].y = -1;
				downFlag = upFlag = false;
				continue;
			}
			if (ROIBox.width > img.cols*0.05 && ROIBox.height > img.cols*0.05)
			{
				Corners[0].x = Corners[0].y = -1;
				Corners[1].x = Corners[1].y = -1;
				Corners[2].x = Corners[2].y = -1;
				downFlag = upFlag = false;
				continue;
			}

			ROIBox.x = Corners[0].x < Corners[2].x ? Corners[0].x : Corners[2].x;
			ROIBox.y = Corners[0].y < Corners[2].y ? Corners[0].y : Corners[2].y;

			roi = img(ROIBox);
			downFlag = upFlag = false;

			Corners[0].x = Corners[0].y = -1;
			Corners[1].x = Corners[1].y = -1;
			Corners[2].x = Corners[2].y = -1;
		}
	}
	cv::destroyWindow("ROI select(Press Esc to close window)");

	delete[] Corners;
}

void color_Tester::on_pushButton_PIMA_clicked() {

	QString filename = QFileDialog::getOpenFileName(this, tr("Open Image"), "", tr("Image File(*.bmp *.jpg *.jpeg *.png *.pbm *.pgm *.ppm)"));
	QTextCodec *code = QTextCodec::codecForName("gb18030");
	name = code->fromUnicode(filename).data();

	temp_image = cv::imread(name);
	if (!temp_image.data)
	{
		QMessageBox msgBox;
		msgBox.setText(tr("image data is null"));
		msgBox.exec();
	}
	else
	{
		cvtColor(temp_image, gray_image, CV_BGR2GRAY);
		temp_image.release();
	}
	Mat roi; Rect ROIBox;
	ROISelection(gray_image, roi, ROIBox);

	if (roi.empty())
	{
		QMessageBox msgBox;
		msgBox.setText(tr("No roi has been cropped"));
		msgBox.exec();
		return;
	}
	int BV_cnt[256] = { 0 };
	int t_sum = 0, back_BV = 255, black_BV = 0;
	for (int i = 0; i < roi.rows; i++) {
		uchar* indata = roi.ptr<uchar>(i);
		for (int j = 0; j < roi.cols; j++) {
			BV_cnt[indata[j]]++;
			t_sum += indata[j];
		}
	}
	int avg_BV = t_sum / (roi.rows*roi.cols);
	t_sum = 0;
	int t_cnt = roi.rows * roi.cols;
	int t_cnt10 = roi.rows * roi.cols*0.1;
	for (int i = 255; i > 50; i--) {
		t_sum += BV_cnt[i];
		if (t_sum > t_cnt10) {
			back_BV = i;
			break;
		}
	}
	t_sum = 0;
	for (int i = 0; i < 200; i++) {
		t_sum += BV_cnt[i];
		if (t_sum > 32) {
			black_BV = i;
			break;
		}
	}
	int roi_x = ROIBox.x + ROIBox.width / 2;
	int roi_y = ROIBox.y + ROIBox.height / 2;


	cv::imshow("roi", roi);
	cv::waitKey(0);
	cv::destroyAllWindows();

	int ret = 0;
	int Pos_flag = 8;
	long steps = 50;
	long my_steps = 50;

	long stepGap = 4;
	long tvLine = 11;
	float stepVa = 20.0f;
	long startV = PIMA[8][1];//3400;
	long endV = PIMA[8][0];// 1600;
	float MTF_spec = 0.025;
	float roi_d = sqrt(pow(roi_x - gray_image.cols / 2, 2) + pow(roi_y - gray_image.rows / 2, 2));
	float d10 = sqrt(pow(gray_image.cols / 2, 2) + pow(gray_image.rows / 2, 2));

	if (roi_d<d10*0.75) {
		startV = PIMA[7][1];
		endV = PIMA[7][0];;
		MTF_spec = 0.025;
		Pos_flag = 7;
	}

	if (roi_d<d10*0.6) {
		startV = PIMA[5][1];
		endV = PIMA[5][0];;
		MTF_spec = 0.03;
		Pos_flag = 5;
	}

	if (roi_d<d10*0.4) {
		startV = PIMA[3][1];
		endV = PIMA[3][0];;
		MTF_spec = 0.03;
		Pos_flag = 3;
	}

	long thresh = 2;
	float specMtf = 5.0f;
	long slope = 4;
	float mtfRes = 0;
	float stepPixels = 0;
	int pimaTV = 0;
	int pos_x = 0;
	int pos_y = 0;
	int PIMA_Result = startV;

	imwrite("PIMA.bmp", roi);
	int th_BV = black_BV + (back_BV - black_BV)*0.5;

	if (roi.cols > roi.rows) {
		Pos_flag += 10;
		//left
		int left_limit = 0, edge1 = -1, rowshalf = roi.rows / 2;
		bool scan = false;
		int top_limit = roi.rows / 2, bottom_limit = roi.rows / 2;

		for (int x = 0; x < roi.cols; x++) {
			int x_sum = 0, y_start = roi.rows / 4, y_end = roi.rows * 3 / 4;;
			for (int y = y_start; y < y_end; y++) {
				x_sum += roi.at<uchar>(y, x);
			}
			x_sum /= y_end - y_start;
			if (scan&&x_sum > avg_BV*0.5) {
				left_limit = x;
				break;
			}
			if (x_sum < avg_BV*0.5) {
				scan = true;
				if (edge1 == -1)edge1 = x;
			}
		}
		int left_edge_center = (edge1 + left_limit) / 2 - 1;

		for (int y = rowshalf; y < roi.rows; y++) {
			uchar indata = roi.at<uchar>(y, left_edge_center);
			if (indata > avg_BV) {
				if (y > bottom_limit)bottom_limit = y;
				break;
			}
			if (y == roi.rows - 1)bottom_limit = y;
		}

		for (int y = rowshalf; y > 0; y--) {
			uchar indata = roi.at<uchar>(y, left_edge_center);
			if (indata > avg_BV) {
				if (y < top_limit)top_limit = y;
				break;
			}
			if (y == 0)top_limit = y;
		}

		if (left_limit == 0 || left_limit>roi.cols / 2) {
			QMessageBox msgBox;
			msgBox.setText(tr("PIMA ROI is not full"));
			msgBox.exec();
			return;
		}
		//right
		int right_limit = 0, edge2 = -1; scan = false;
		for (int x = roi.cols - 1; x > 0; x--) {
			int x_sum = 0, y_start = roi.rows / 4, y_end = roi.rows * 3 / 4;
			for (int y = y_start; y < y_end; y++) {
				x_sum += roi.at<uchar>(y, x);
			}
			x_sum /= y_end - y_start;
			if (scan&&x_sum > avg_BV*0.5) {
				right_limit = x;
				break;
			}
			if (x_sum < avg_BV*0.5) {
				scan = true;
				if (edge2 == -1)edge2 = x;
			}
		}
		if (right_limit == 0 || right_limit < roi.cols / 2) {
			QMessageBox msgBox;
			msgBox.setText(tr("PIMA ROI is not full"));
			msgBox.exec();
			return;
		}

		int right_edge_center = (edge2 + right_limit) / 2 + 1;

		for (int y = rowshalf; y < roi.rows; y++) {
			uchar indata = roi.at<uchar>(y, right_edge_center);
			if (indata > avg_BV) {
				if (y > bottom_limit)bottom_limit = y;
				break;
			}
			if (y == roi.rows - 1)bottom_limit = roi.rows - 1;
		}

		for (int y = rowshalf; y >= 0; y--) {
			uchar indata = roi.at<uchar>(y, right_edge_center);
			if (indata > avg_BV) {
				if (y < top_limit)top_limit = y;
				break;
			}
			if (y == 0)top_limit = y;
		}
		int Y_range = bottom_limit - top_limit;
		my_steps = (startV - endV) / 50;
		stepPixels = (right_limit - left_limit - 1.0) / my_steps;

		Mat left_ROI = roi(Rect(roi.cols*0.15, roi.rows*0.15, roi.cols*0.3, roi.rows*0.7));
		Mat right_ROI = roi(Rect(roi.cols*0.55, roi.rows*0.15, roi.cols*0.3, roi.rows*0.7));

		int left_ss = 0, right_ss = 0;
		for (int y = 0; y < left_ROI.rows; ++y) {
			for (int x = 0; x < left_ROI.cols; ++x) {
				if (left_ROI.at<uchar>(y, x) > avg_BV) {
					left_ss++;
				}
				if (right_ROI.at<uchar>(y, x) >avg_BV) {
					right_ss++;
				}
			}
		}
		if (left_ss < right_ss) {
			int n = 0;
			for (float x = right_limit; x >= left_limit; x -= stepPixels) {
				int x_s = x, x_e = x - 4, test_cnt = 0;
				if (x == right_limit) {
					x_s -= stepPixels / 2;
					x_e -= stepPixels / 2;
				}
				int offset = Y_range *(0.05 + 0.15*(x - left_limit) / (right_limit - left_limit));
				int bv_x[256] = { 0 };
				for (int b = x_s; b > x_e; b--) {
					int y = top_limit + offset, end_y = bottom_limit - offset;
					while (roi.at<uchar>(y, b) < th_BV&&y < roi.rows / 2)
						y++;
					for (; y < roi.rows; y++) {
						uchar data = roi.at<uchar>(y, b);
						if (data < th_BV)break;
					}
					while (roi.at<uchar>(end_y, b) < th_BV&&end_y > roi.rows / 2)
						end_y--;

					for (; end_y >0; end_y--) {
						uchar data = roi.at<uchar>(end_y, b);
						if (data < th_BV)break;
					}
					y += (end_y - y) / 6;
					end_y -= (end_y - y) / 6;
					for (int a = y; a <= end_y; a++) {
						bv_x[roi.at<uchar>(a, b)]++;
						test_cnt++;
					}
				}	
				float th = test_cnt * 0.3;
				if (Pos_flag%10 == 3) th = test_cnt *0.25;
				th = test_cnt * 0.3;
				int dark_sum = 0, bright_sum = 0;
				float dark_BV = 0, briht_BV = 0;
				for (int i = 0; i < 256; i++) {
					while (dark_sum < th&&bv_x[i]>0) {
						dark_sum++;
						dark_BV += i;
						bv_x[i]--;
					}
					if (dark_sum >= th)break;
				}

				for (int i = 255; i > 0; i--) {
					while (bright_sum < th&&bv_x[i]>0) {
						bright_sum++;
						briht_BV += i;
						bv_x[i]--;
					}
					if (bright_sum >= th)break;
				}

				dark_BV /= dark_sum;
				briht_BV /= bright_sum;

				float mtf = (briht_BV - dark_BV) / (dark_BV + briht_BV);
				if (mtf > MTF_spec) {
					break;
				}
				PIMA_Result -= 50;
			}
		}
		else {
			int n = 0;
			for (float x = left_limit; x <= right_limit; x += stepPixels) {
				int x_s = x, x_e = x + 4, test_cnt = 0;
				if (x == left_limit) {
					x_s += stepPixels / 2;
					x_e += stepPixels / 2;
				}
				int offset = Y_range *(0.05 + 0.15*(right_limit - x) / (right_limit - left_limit));
				int bv_x[256] = { 0 };
				for (int b = x_s; b < x_e; b++) {
					int y = top_limit + offset, end_y = bottom_limit - offset;
					while (roi.at<uchar>(y, b) < th_BV&&y < roi.rows / 2)
						y++;

					for (; y < roi.rows; y++) {
						uchar data = roi.at<uchar>(y, b);
						if (data < th_BV)break;
					}
					while (roi.at<uchar>(end_y, b) < th_BV&&end_y > roi.rows / 2)
						end_y--;

					for (; end_y >0; end_y--) {
						uchar data = roi.at<uchar>(end_y, b);
						if (data < th_BV)break;
					}
					y += (end_y - y) / 6;
					end_y -= (end_y - y) / 6;

					for (int a = y; a <= end_y; a++) {
						bv_x[roi.at<uchar>(a, b)]++;
						test_cnt++;
					}
				}
				float th = test_cnt * 0.3;
				if (Pos_flag%10 == 3) th = test_cnt * 0.25;
				int dark_sum = 0, bright_sum = 0;
				float dark_BV = 0, briht_BV = 0;
				for (int i = 0; i < 256; i++) {
					while (dark_sum < th&&bv_x[i]>0) {
						dark_sum++;
						dark_BV += i;
						bv_x[i]--;
					}
					if (dark_sum >= th)break;
				}

				for (int i = 255; i > 0; i--) {
					while (bright_sum < th&&bv_x[i]>0) {
						bright_sum++;
						briht_BV += i;
						bv_x[i]--;
					}
					if (bright_sum >= th)break;
				}

				dark_BV /= dark_sum;
				briht_BV /= bright_sum;

				float mtf = (briht_BV - dark_BV) / (dark_BV + briht_BV);
				if (mtf > MTF_spec) {
					break;
				}
				PIMA_Result -= 50;
			}
		}
	}
	else {
		//up
		int up_limit = 0, edge1 = -1, colshalf = roi.cols / 2;
		bool scan = false;
		int left_limit = colshalf, right_limit = colshalf;

		for (int y = 0; y < roi.rows; y++) {
			int y_sum = 0, x_start = roi.cols / 4, x_end = roi.cols * 3 / 4;;
			for (int x = x_start; x < x_end; x++) {
				y_sum += roi.at<uchar>(y, x);
			}
			y_sum /= x_end - x_start;
			if (scan&&y_sum > avg_BV*0.5) {
				up_limit = y;
				break;
			}
			if (y_sum < avg_BV*0.5) {
				scan = true;
				if (edge1 == -1)edge1 = y;
			}
		}
		int up_edge_center = (edge1 + up_limit) / 2 - 1;

		for (int x = colshalf; x < roi.cols; x++) {
			uchar indata = roi.at<uchar>(up_edge_center, x);
			if (indata > avg_BV) {
				if (x > right_limit)right_limit = x;
				break;
			}
			if (x == roi.cols - 1)right_limit = roi.cols - 1;
		}

		for (int x = colshalf; x >= 0; x--) {
			uchar indata = roi.at<uchar>(up_edge_center, x);
			if (indata > avg_BV) {
				if (x < left_limit)left_limit = x;
				break;
			}
			if (x == 0)left_limit = x;
		}

		if (up_limit == 0 || up_limit>roi.cols / 2) {
			QMessageBox msgBox;
			msgBox.setText(tr("PIMA ROI is not full"));
			msgBox.exec();
			return;
		}
		//down
		int down_limit = 0, edge2 = -1; scan = false;
		for (int y = roi.rows - 1; y > 0; y--) {
			int y_sum = 0, x_start = roi.cols / 4, x_end = roi.cols * 3 / 4;
			for (int x = x_start; x < x_end; x++) {
				y_sum += roi.at<uchar>(y, x);
			}
			y_sum /= x_end - x_start;
			if (scan&&y_sum > avg_BV*0.5) {
				down_limit = y;
				break;
			}
			if (y_sum < avg_BV*0.5) {
				scan = true;
				if (edge2 == -1)edge2 = y;
			}
		}
		if (down_limit == 0 || down_limit < roi.rows / 2) {
			QMessageBox msgBox;
			msgBox.setText(tr("PIMA ROI is not full"));
			msgBox.exec();
			return;
		}

		int down_edge_center = (edge2 + down_limit) / 2 + 1;

		for (int x = colshalf; x < roi.cols; x++) {
			uchar indata = roi.at<uchar>(down_edge_center, x);
			if (indata > avg_BV) {
				if (x > right_limit)right_limit = x;
				break;
			}
			if (x == roi.cols - 1)right_limit = roi.cols - 1;
		}

		for (int x = colshalf; x >= 0; x--) {
			uchar indata = roi.at<uchar>(down_edge_center, x);
			if (indata > avg_BV) {
				if (x < left_limit)left_limit = x;
				break;
			}
			if (x == 0)left_limit = x;
		}
		int X_range = right_limit - left_limit;
		my_steps = (startV - endV) / 50;
		stepPixels = (down_limit - up_limit - 1.0) / my_steps;

		Mat up_ROI = roi(Rect(roi.cols*0.15, roi.rows*0.15, roi.cols*0.7, roi.rows*0.3));
		Mat down_ROI = roi(Rect(roi.cols*0.15, roi.rows*0.55, roi.cols*0.7, roi.rows*0.3));

		int up_ss = 0, down_ss = 0;
		for (int y = 0; y < up_ROI.rows; ++y) {
			for (int x = 0; x < up_ROI.cols; ++x) {
				if (up_ROI.at<uchar>(y, x) > avg_BV) {
					up_ss++;
				}
				if (down_ROI.at<uchar>(y, x) >avg_BV) {
					down_ss++;
				}
			}
		}

		if (up_ss < down_ss) {
			int n = 0;
			for (float y = down_limit; y >= up_limit; y -= stepPixels) {
				int y_s = y, y_e = y - 4, test_cnt = 0;
				if (y == down_limit) {
					y_s -= stepPixels / 2;
					y_e -= stepPixels / 2;
				}
				int offset = X_range *(0.05 + 0.15*(y - up_limit) / (down_limit - up_limit));
				int bv_y[256] = { 0 };
				for (int b = y_s; b > y_e; b--) {
					int x = left_limit + offset, end_x = right_limit - offset;

					while (roi.at<uchar>(b, x) < th_BV&&x < roi.cols / 2)
						x++;

					for (; x < roi.cols; x++) {
						uchar data = roi.at<uchar>(b, x);
						if (data < th_BV)break;
					}
					while (roi.at<uchar>(b, end_x) < th_BV&&end_x > roi.cols / 2)
						end_x--;

					for (; end_x >0; end_x--) {
						uchar data = roi.at<uchar>(b, end_x);
						if (data < th_BV)break;
					}
					x += (end_x - x) / 6;
					end_x -= (end_x - x) / 6;
					for (int a = x; a <= end_x; a++) {
						bv_y[roi.at<uchar>(b, a)]++;
						test_cnt++;
					}
				}
				float th = test_cnt * 0.3;
				if (Pos_flag%10 == 3) th = test_cnt * 0.25;
				int dark_sum = 0, bright_sum = 0;
				float dark_BV = 0, briht_BV = 0;
				for (int i = 0; i < 256; i++) {
					while (dark_sum < th&&bv_y[i]>0) {
						dark_sum++;
						dark_BV += i;
						bv_y[i]--;
					}
					if (dark_sum >= th)break;
				}

				for (int i = 255; i > 0; i--) {
					while (bright_sum < th&&bv_y[i]>0) {
						bright_sum++;
						briht_BV += i;
						bv_y[i]--;
					}
					if (bright_sum >= th)break;
				}

				dark_BV /= dark_sum;
				briht_BV /= bright_sum;

				float mtf = (briht_BV - dark_BV) / (dark_BV + briht_BV);
				if (mtf > MTF_spec) {
					break;
				}
				PIMA_Result -= 50;
			}
		}
		else {
			int n = 0;
			for (float y = up_limit; y <= down_limit; y += stepPixels) {
				int y_s = y, y_e = y + 4, test_cnt = 0;
				if (y == up_limit) {
					y_s += stepPixels / 2;
					y_e += stepPixels / 2;
				}
				int offset = X_range *(0.05 + 0.15*(down_limit - y) / (down_limit - up_limit));

				int bv_y[256] = { 0 }, bv_back[256] = { 0 };
				for (int b = y_s; b < y_e; b++) {
					int x = left_limit + offset, end_x = right_limit - offset;

					while (roi.at<uchar>(b, x) < th_BV&&x < roi.cols / 2)
						x++;

					for (; x < roi.cols; x++) {
						uchar data = roi.at<uchar>(b, x);
						if (data < th_BV)break;
					}
					while (roi.at<uchar>(b, end_x) < th_BV&&end_x > roi.cols / 2)
						end_x--;

					for (; end_x >0; end_x--) {
						uchar data = roi.at<uchar>(b, end_x);
						if (data < th_BV)break;
					}

					x += (end_x - x) / 6;
					end_x -= (end_x - x) / 6;

					for (int a = x; a <= end_x; a++) {
						bv_y[roi.at<uchar>(b, a)]++;
						bv_back[roi.at<uchar>(b, a)]++;
						test_cnt++;
					}
				}

				float th = test_cnt * 0.3;
				if (Pos_flag%10 == 3) th = test_cnt *0.25;
				int dark_sum = 0, bright_sum = 0;
				float dark_BV = 0, briht_BV = 0;
				for (int i = 0; i < 256; i++) {
					while (dark_sum < th&&bv_y[i]>0) {					
						dark_sum++;
						dark_BV +=i;
						bv_y[i]--;
					}
					if (dark_sum >= th)break;
				}

				for (int i = 255; i > 0; i--) {
					while (bright_sum < th&&bv_y[i]>0) {
						bright_sum++;
						briht_BV += i;
						bv_y[i]--;
					}
					if (bright_sum >= th)break;
				}

				dark_BV /= dark_sum;
				briht_BV /= bright_sum;

				float mtf = (briht_BV - dark_BV) / (dark_BV + briht_BV);
				if (mtf > MTF_spec) {
					break;
				}
				PIMA_Result -= 50;
			}
		}

	}
	if (ret) {
		string str = "PIMA test Fail!\n";
		ui.log->insertPlainText(str.c_str());
	}
	else {
		string str = "0." + to_string(Pos_flag % 10) + "F_";
		if (Pos_flag / 10 == 1)str += "H ";
		else str += "V ";
		str += " PIMA:	" + to_string(PIMA_Result) + '\n';
		ui.log->insertPlainText(str.c_str());
	}
	gray_image.release();
}

void color_Tester::on_pushButton_PIMA_2_clicked() {

	QString filename = QFileDialog::getOpenFileName(this, tr("Open Image"), "", tr("Image File(*.bmp *.jpg *.jpeg *.png *.pbm *.pgm *.ppm)"));
	QTextCodec *code = QTextCodec::codecForName("gb18030");
	name = code->fromUnicode(filename).data();

	temp_image = cv::imread(name);
	if (!temp_image.data)
	{
		QMessageBox msgBox;
		msgBox.setText(tr("image data is null"));
		msgBox.exec();
	}
	else
	{
		cvtColor(temp_image, gray_image, CV_BGR2GRAY);
		temp_image.release();
	}
	Mat roi; Rect ROIBox;
	ROISelection(gray_image, roi, ROIBox);

	if (roi.empty())
	{
		QMessageBox msgBox;
		msgBox.setText(tr("No roi has been cropped"));
		msgBox.exec();
		return;
	}
	int BV_cnt[256] = { 0 };
	int t_sum = 0, back_BV = 255, black_BV = 0;
	for (int i = 0; i < roi.rows; i++) {
		uchar* indata = roi.ptr<uchar>(i);
		for (int j = 0; j < roi.cols; j++) {
			BV_cnt[indata[j]]++;
			t_sum += indata[j];
		}
	}
	int avg_BV = t_sum / (roi.rows*roi.cols);
	t_sum = 0;
	int t_cnt = roi.rows * roi.cols;
	int t_cnt10 = roi.rows * roi.cols*0.1;
	for (int i = 255; i > 50; i--) {
		t_sum += BV_cnt[i];
		if (t_sum > t_cnt10) {
			back_BV = i;
			break;
		}
	}
	t_sum = 0;
	for (int i = 0; i < 200; i++) {
		t_sum += BV_cnt[i];
		if (t_sum > 32) {
			black_BV = i;
			break;
		}
	}
	int roi_x = ROIBox.x + ROIBox.width / 2;
	int roi_y = ROIBox.y + ROIBox.height / 2;


	cv::imshow("roi", roi);
	cv::waitKey(0);
	cv::destroyAllWindows();

	int ret = 0;
	int Pos_flag = 8;
	long steps = 50;
	long my_steps = 50;

	long stepGap = 4;
	long tvLine = 11;
	float stepVa = 20.0f;
	long startV = PIMA[8][1];//3400;
	long endV = PIMA[8][0];// 1600;
	float MTF_spec = 0.025;
	float roi_d = sqrt(pow(roi_x - gray_image.cols / 2, 2) + pow(roi_y - gray_image.rows / 2, 2));
	float d10 = sqrt(pow(gray_image.cols / 2, 2) + pow(gray_image.rows / 2, 2));

	if (roi_d<d10*0.75) {
		startV = PIMA[7][1];
		endV = PIMA[7][0];;
		MTF_spec = 0.025;
		Pos_flag = 7;
	}

	if (roi_d<d10*0.6) {
		startV = PIMA[5][1];
		endV = PIMA[5][0];;
		MTF_spec = 0.03;
		Pos_flag = 5;
	}

	if (roi_d<d10*0.4) {
		startV = PIMA[3][1];
		endV = PIMA[3][0];;
		MTF_spec = 0.03;
		Pos_flag = 3;
	}

	long thresh = 2;
	float specMtf = 5.0f;
	long slope = 4;
	float mtfRes = 0;
	float stepPixels = 0;
	int pimaTV = 0;
	int pos_x = 0;
	int pos_y = 0;
	int PIMA_Result = startV;

	bool pass_flag[128] = { 0 };
	int pass_cnt = 0;

	imwrite("PIMA.bmp", roi);
	int th_BV = black_BV + (back_BV - black_BV)*0.5;

	if (roi.cols > roi.rows) {
		Pos_flag += 10;
		//left
		int left_limit = 0, edge1 = -1, rowshalf = roi.rows / 2;
		bool scan = false;
		int top_limit = roi.rows / 2, bottom_limit = roi.rows / 2;

		for (int x = 0; x < roi.cols; x++) {
			int x_sum = 0, y_start = roi.rows / 4, y_end = roi.rows * 3 / 4;;
			for (int y = y_start; y < y_end; y++) {
				x_sum += roi.at<uchar>(y, x);
			}
			x_sum /= y_end - y_start;
			if (scan&&x_sum > avg_BV*0.5) {
				left_limit = x;
				break;
			}
			if (x_sum < avg_BV*0.5) {
				scan = true;
				if (edge1 == -1)edge1 = x;
			}
		}
		int left_edge_center = (edge1 + left_limit) / 2 - 1;

		for (int y = rowshalf; y < roi.rows; y++) {
			uchar indata = roi.at<uchar>(y, left_edge_center);
			if (indata > avg_BV) {
				if (y > bottom_limit)bottom_limit = y;
				break;
			}
			if (y == roi.rows - 1)bottom_limit = y;
		}

		for (int y = rowshalf; y > 0; y--) {
			uchar indata = roi.at<uchar>(y, left_edge_center);
			if (indata > avg_BV) {
				if (y < top_limit)top_limit = y;
				break;
			}
			if (y == 0)top_limit = y;
		}

		if (left_limit == 0 || left_limit>roi.cols / 2) {
			QMessageBox msgBox;
			msgBox.setText(tr("PIMA ROI is not full"));
			msgBox.exec();
			return;
		}
		//right
		int right_limit = 0, edge2 = -1; scan = false;
		for (int x = roi.cols - 1; x > 0; x--) {
			int x_sum = 0, y_start = roi.rows / 4, y_end = roi.rows * 3 / 4;
			for (int y = y_start; y < y_end; y++) {
				x_sum += roi.at<uchar>(y, x);
			}
			x_sum /= y_end - y_start;
			if (scan&&x_sum > avg_BV*0.5) {
				right_limit = x;
				break;
			}
			if (x_sum < avg_BV*0.5) {
				scan = true;
				if (edge2 == -1)edge2 = x;
			}
		}
		if (right_limit == 0 || right_limit < roi.cols / 2) {
			QMessageBox msgBox;
			msgBox.setText(tr("PIMA ROI is not full"));
			msgBox.exec();
			return;
		}

		int right_edge_center = (edge2 + right_limit) / 2 + 1;

		for (int y = rowshalf; y < roi.rows; y++) {
			uchar indata = roi.at<uchar>(y, right_edge_center);
			if (indata > avg_BV) {
				if (y > bottom_limit)bottom_limit = y;
				break;
			}
			if (y == roi.rows - 1)bottom_limit = roi.rows - 1;
		}

		for (int y = rowshalf; y >= 0; y--) {
			uchar indata = roi.at<uchar>(y, right_edge_center);
			if (indata > avg_BV) {
				if (y < top_limit)top_limit = y;
				break;
			}
			if (y == 0)top_limit = y;
		}
		int Y_range = bottom_limit - top_limit;
		my_steps = (startV - endV) / 50;
		stepPixels = (right_limit - left_limit - 1.0) / my_steps;

		Mat left_ROI = roi(Rect(roi.cols*0.15, roi.rows*0.15, roi.cols*0.3, roi.rows*0.7));
		Mat right_ROI = roi(Rect(roi.cols*0.55, roi.rows*0.15, roi.cols*0.3, roi.rows*0.7));

		int left_ss = 0, right_ss = 0;
		for (int y = 0; y < left_ROI.rows; ++y) {
			for (int x = 0; x < left_ROI.cols; ++x) {
				if (left_ROI.at<uchar>(y, x) > avg_BV) {
					left_ss++;
				}
				if (right_ROI.at<uchar>(y, x) >avg_BV) {
					right_ss++;
				}
			}
		}
		if (left_ss < right_ss) {
			for (float x = right_limit; x >= left_limit; x -= stepPixels) {
				int x_s = x, x_e = x - 4;
				if (x == right_limit) {
					x_s -= stepPixels / 2;
					x_e -= stepPixels / 2;
				}
				int offset = Y_range *(0.05 + 0.15*(x - left_limit) / (right_limit - left_limit));
				int y = top_limit + offset, end_y = bottom_limit - offset;

				int wave_cnt =0;
				for (int b = x_s; b > x_e; b--) {
					y = top_limit + offset, end_y = bottom_limit - offset;
					while (roi.at<uchar>(y, b) < th_BV&&y < roi.rows / 2)
						y++;
					for (; y < roi.rows; y++) {
						uchar data = roi.at<uchar>(y, b);
						if (data < th_BV)break;
					}
					while (roi.at<uchar>(end_y, b) < th_BV&&end_y > roi.rows / 2)
						end_y--;

					for (; end_y >0; end_y--) {
						uchar data = roi.at<uchar>(end_y, b);
						if (data < th_BV)break;
					}
					y += (end_y - y) / 10;
					end_y -= (end_y - y) / 10;
					int test_cnt = 0;
					int bv_x[256] = { 0 }, bv_y[256] = { 0 };
					for (int a = y; a <= end_y; a++) {
						bv_y[a-y] = roi.at<uchar>(a, b);
						bv_x[roi.at<uchar>(a, b)]++;
						test_cnt++;
					}

					float th = test_cnt * 0.45;
					int dark_sum = 0, bright_sum = 0;
					float dark_BV = 0, briht_BV = 0;
					for (int i = 0; i < 256; i++) {
						while (dark_sum < th&&bv_x[i]>0) {
							dark_sum++;
							dark_BV += i;
							bv_x[i]--;
						}
						if (dark_sum >= th)break;
					}

					for (int i = 255; i > 0; i--) {
						while (bright_sum < th&&bv_x[i]>0) {
							bright_sum++;
							briht_BV += i;
							bv_x[i]--;
						}
						if (bright_sum >= th)break;
					}

					dark_BV /= dark_sum;
					briht_BV /= bright_sum;
					bool peak = false, valley = false;
					int cnt = 0;
					for (int a = y; a <= end_y;a++) {
						uchar indata = roi.at<uchar>(a, b);
						if (indata > briht_BV) peak = true;
						if (indata < dark_BV) valley = true;
						if (peak&&valley) {
							cnt++;
							peak = valley = false;
						}
						if (indata > briht_BV-0.5) peak = true;
						if (indata < dark_BV+0.5) valley = true;
					}
					if (cnt > 11 && briht_BV - dark_BV>3)wave_cnt++;
				}
				if (wave_cnt > 2)
					pass_flag[pass_cnt] = true;

				pass_cnt++;
			}
		}
		else {
			int n = 0;
			for (float x = left_limit; x <= right_limit; x += stepPixels) {
				int x_s = x, x_e = x + 4;
				if (x == left_limit) {
					x_s += stepPixels / 2;
					x_e += stepPixels / 2;
				}
				int offset = Y_range *(0.05 + 0.15*(right_limit - x) / (right_limit - left_limit));
				int wave_cnt = 0;
				for (int b = x_s; b < x_e; b++) {
					int y = top_limit + offset, end_y = bottom_limit - offset;
					while (roi.at<uchar>(y, b) < th_BV&&y < roi.rows / 2)
						y++;

					for (; y < roi.rows; y++) {
						uchar data = roi.at<uchar>(y, b);
						if (data < th_BV)break;
					}
					while (roi.at<uchar>(end_y, b) < th_BV&&end_y > roi.rows / 2)
						end_y--;

					for (; end_y >0; end_y--) {
						uchar data = roi.at<uchar>(end_y, b);
						if (data < th_BV)break;
					}
					y += (end_y - y) / 10;
					end_y -= (end_y - y) / 10;
					int test_cnt = 0;

					int bv_x[256] = { 0 }, bv_y[256] = { 0 };
					for (int a = y; a <= end_y; a++) {
						bv_y[a - y] = roi.at<uchar>(a, b);
						bv_x[roi.at<uchar>(a, b)]++;
						test_cnt++;
					}

					float th = test_cnt * 0.45;
					int dark_sum = 0, bright_sum = 0;
					float dark_BV = 0, briht_BV = 0;
					for (int i = 0; i < 256; i++) {
						while (dark_sum < th&&bv_x[i]>0) {
							dark_sum++;
							dark_BV += i;
							bv_x[i]--;
						}
						if (dark_sum >= th)break;
					}

					for (int i = 255; i > 0; i--) {
						while (bright_sum < th&&bv_x[i]>0) {
							bright_sum++;
							briht_BV += i;
							bv_x[i]--;
						}
						if (bright_sum >= th)break;
					}

					dark_BV /= dark_sum;
					briht_BV /= bright_sum;

					bool peak = false, valley = false;
					int cnt = 0;
					for (int a = y; a <= end_y; a++) {
						uchar indata = roi.at<uchar>(a, b);
						if (indata > briht_BV) peak = true;
						if (indata < dark_BV) valley = true;
						if (peak&&valley) {
							cnt++;
							peak = valley = false;
						}
						if (indata > briht_BV - 0.5) peak = true;
						if (indata < dark_BV + 0.5) valley = true;
					}
					if (cnt > 11&& briht_BV- dark_BV>3)wave_cnt++;
				}
				if (wave_cnt > 2)
					pass_flag[pass_cnt] = true;

				pass_cnt++;
			}
		}
	}
	else {
		//up
		int up_limit = 0, edge1 = -1, colshalf = roi.cols / 2;
		bool scan = false;
		int left_limit = colshalf, right_limit = colshalf;

		for (int y = 0; y < roi.rows; y++) {
			int y_sum = 0, x_start = roi.cols / 4, x_end = roi.cols * 3 / 4;;
			for (int x = x_start; x < x_end; x++) {
				y_sum += roi.at<uchar>(y, x);
			}
			y_sum /= x_end - x_start;
			if (scan&&y_sum > avg_BV*0.5) {
				up_limit = y;
				break;
			}
			if (y_sum < avg_BV*0.5) {
				scan = true;
				if (edge1 == -1)edge1 = y;
			}
		}
		int up_edge_center = (edge1 + up_limit) / 2 - 1;

		for (int x = colshalf; x < roi.cols; x++) {
			uchar indata = roi.at<uchar>(up_edge_center, x);
			if (indata > avg_BV) {
				if (x > right_limit)right_limit = x;
				break;
			}
			if (x == roi.cols - 1)right_limit = roi.cols - 1;
		}

		for (int x = colshalf; x >= 0; x--) {
			uchar indata = roi.at<uchar>(up_edge_center, x);
			if (indata > avg_BV) {
				if (x < left_limit)left_limit = x;
				break;
			}
			if (x == 0)left_limit = x;
		}

		if (up_limit == 0 || up_limit>roi.cols / 2) {
			QMessageBox msgBox;
			msgBox.setText(tr("PIMA ROI is not full"));
			msgBox.exec();
			return;
		}
		//down
		int down_limit = 0, edge2 = -1; scan = false;
		for (int y = roi.rows - 1; y > 0; y--) {
			int y_sum = 0, x_start = roi.cols / 4, x_end = roi.cols * 3 / 4;
			for (int x = x_start; x < x_end; x++) {
				y_sum += roi.at<uchar>(y, x);
			}
			y_sum /= x_end - x_start;
			if (scan&&y_sum > avg_BV*0.45) {
				down_limit = y;
				break;
			}
			if (y_sum < avg_BV*0.5) {
				scan = true;
				if (edge2 == -1)edge2 = y;
			}
		}
		if (down_limit == 0 || down_limit < roi.rows / 2) {
			QMessageBox msgBox;
			msgBox.setText(tr("PIMA ROI is not full"));
			msgBox.exec();
			return;
		}

		int down_edge_center = (edge2 + down_limit) / 2 + 1;

		for (int x = colshalf; x < roi.cols; x++) {
			uchar indata = roi.at<uchar>(down_edge_center, x);
			if (indata > avg_BV) {
				if (x > right_limit)right_limit = x;
				break;
			}
			if (x == roi.cols - 1)right_limit = roi.cols - 1;
		}

		for (int x = colshalf; x >= 0; x--) {
			uchar indata = roi.at<uchar>(down_edge_center, x);
			if (indata > avg_BV) {
				if (x < left_limit)left_limit = x;
				break;
			}
			if (x == 0)left_limit = x;
		}
		int X_range = right_limit - left_limit;
		my_steps = (startV - endV) / 50;
		stepPixels = (down_limit - up_limit - 1.0) / my_steps;

		Mat up_ROI = roi(Rect(roi.cols*0.15, roi.rows*0.15, roi.cols*0.7, roi.rows*0.3));
		Mat down_ROI = roi(Rect(roi.cols*0.15, roi.rows*0.55, roi.cols*0.7, roi.rows*0.3));

		int up_ss = 0, down_ss = 0;
		for (int y = 0; y < up_ROI.rows; ++y) {
			for (int x = 0; x < up_ROI.cols; ++x) {
				if (up_ROI.at<uchar>(y, x) > avg_BV) {
					up_ss++;
				}
				if (down_ROI.at<uchar>(y, x) >avg_BV) {
					down_ss++;
				}
			}
		}

		if (up_ss < down_ss) {
			int n = 0;
			for (float y = down_limit; y >= up_limit; y -= stepPixels) {
				int y_s = y, y_e = y - 4;
				if (y == down_limit) {
					y_s -= stepPixels / 2;
					y_e -= stepPixels / 2;
				}
				int offset = X_range *(0.05 + 0.15*(y - up_limit) / (down_limit - up_limit));
				int wave_cnt = 0;
				for (int b = y_s; b > y_e; b--) {
					int x = left_limit + offset, end_x = right_limit - offset;

					while (roi.at<uchar>(b, x) < th_BV&&x < roi.cols / 2)
						x++;

					for (; x < roi.cols; x++) {
						uchar data = roi.at<uchar>(b, x);
						if (data < th_BV)break;
					}
					while (roi.at<uchar>(b, end_x) < th_BV&&end_x > roi.cols / 2)
						end_x--;

					for (; end_x >0; end_x--) {
						uchar data = roi.at<uchar>(b, end_x);
						if (data < th_BV)break;
					}
					x += (end_x - x) / 10;
					end_x -= (end_x - x) / 10;
					int test_cnt = 0;
					int bv_x[256] = { 0 }, bv_y[256] = { 0 };
					for (int a = x; a <= end_x; a++) {
						bv_x[a - x] = roi.at<uchar>(b, a);
						bv_y[roi.at<uchar>(b, a)]++;
						test_cnt++;
					}

					float th = test_cnt * 0.45;
					int dark_sum = 0, bright_sum = 0;
					float dark_BV = 0, briht_BV = 0;
					for (int i = 0; i < 256; i++) {
						while (dark_sum < th&&bv_y[i]>0) {
							dark_sum++;
							dark_BV += i;
							bv_y[i]--;
						}
						if (dark_sum >= th)break;
					}

					for (int i = 255; i > 0; i--) {
						while (bright_sum < th&&bv_y[i]>0) {
							bright_sum++;
							briht_BV += i;
							bv_y[i]--;
						}
						if (bright_sum >= th)break;
					}

					dark_BV /= dark_sum;
					briht_BV /= bright_sum;
					bool peak = false, valley = false;
					int cnt = 0;
					for (int a = x; a <= end_x; a++) {
						uchar indata = roi.at<uchar>(b, a);
						if (indata > briht_BV) peak = true;
						if (indata < dark_BV) valley = true;
						if (peak&&valley) {
							cnt++;
							peak = valley = false;
						}
						if (indata > briht_BV - 0.5) peak = true;
						if (indata < dark_BV + 0.5) valley = true;
					}
					if (cnt > 11 && briht_BV - dark_BV>3)wave_cnt++;
				}
				if (wave_cnt > 2)
					pass_flag[pass_cnt] = true;

				pass_cnt++;
			}
		}
		else {
			int n = 0;
			for (float y = up_limit; y <= down_limit; y += stepPixels) {
				int y_s = y, y_e = y + 4, test_cnt = 0;
				if (y == up_limit) {
					y_s += stepPixels / 2;
					y_e += stepPixels / 2;
				}
				int offset = X_range *(0.05 + 0.15*(down_limit - y) / (down_limit - up_limit));
				int wave_cnt = 0;
				for (int b = y_s; b < y_e; b++) {
					int x = left_limit + offset, end_x = right_limit - offset;

					while (roi.at<uchar>(b, x) < th_BV&&x < roi.cols / 2)
						x++;

					for (; x < roi.cols; x++) {
						uchar data = roi.at<uchar>(b, x);
						if (data < th_BV)break;
					}
					while (roi.at<uchar>(b, end_x) < th_BV&&end_x > roi.cols / 2)
						end_x--;

					for (; end_x >0; end_x--) {
						uchar data = roi.at<uchar>(b, end_x);
						if (data < th_BV)break;
					}

					x += (end_x - x) / 10;
					end_x -= (end_x - x) / 10;
					int bv_y[256] = { 0 }, bv_x[256] = { 0 };
					for (int a = x; a <= end_x; a++) {
						bv_y[roi.at<uchar>(b, a)]++;
						bv_x[a-x]= roi.at<uchar>(b, a);
						test_cnt++;
					}

					float th = test_cnt * 0.45;
					int dark_sum = 0, bright_sum = 0;
					float dark_BV = 0, briht_BV = 0;
					for (int i = 0; i < 256; i++) {
						while (dark_sum < th&&bv_y[i]>0) {
							dark_sum++;
							dark_BV += i;
							bv_y[i]--;
						}
						if (dark_sum >= th)break;
					}

					for (int i = 255; i > 0; i--) {
						while (bright_sum < th&&bv_y[i]>0) {
							bright_sum++;
							briht_BV += i;
							bv_y[i]--;
						}
						if (bright_sum >= th)break;
					}

					dark_BV /= dark_sum;
					briht_BV /= bright_sum;
					bool peak = false, valley = false;
					int cnt = 0;
					for (int a = x; a <= end_x; a++) {
						uchar indata = roi.at<uchar>(b, a);
						if (indata > briht_BV) peak = true;
						if (indata < dark_BV) valley = true;
						if (peak&&valley) {
							cnt++;
							peak = valley = false;
						}
						if (indata > briht_BV - 0.5) peak = true;
						if (indata < dark_BV + 0.5) valley = true;
					}
					if (cnt > 11 && briht_BV - dark_BV>3)wave_cnt++;
				}
				if (wave_cnt > 2)
					pass_flag[pass_cnt] = true;

				pass_cnt++;
			}
		}
	}

	for (int k = 0; k < pass_cnt; k++) {
		if (pass_flag[k] && pass_flag[k + 1] && pass_flag[k + 2]) {
			PIMA_Result -= 50 * k;
			break;
		}
		if (k == pass_cnt - 1)PIMA_Result = endV;
	}

	if (ret) {
		string str = "PIMA test Fail!\n";
		ui.log->insertPlainText(str.c_str());
	}
	else {
		string str = "0." + to_string(Pos_flag % 10) + "F_";
		if (Pos_flag / 10 == 1)str += "H ";
		else str += "V ";
		str += " PIMA:	" + to_string(PIMA_Result) + '\n';
		ui.log->insertPlainText(str.c_str());
	}
	gray_image.release();
}

void color_Tester::on_pushButton_open_raw_clicked() {

	Width = ui.width->document()->toPlainText().toInt();
	Height = ui.height->document()->toPlainText().toInt();

	QString filename = QFileDialog::getOpenFileName(this, tr("Open Image"), "", tr("Image File(*.raw)"));
	QTextCodec *code = QTextCodec::codecForName("gb18030");
	name = code->fromUnicode(filename).data();
	if (name.length() < 2) return;

	FILE *fp = NULL;

	int ret = 0;
	if (Width*Height == 0) {
		QMessageBox msgBox;
		msgBox.setText(tr("Plz input Img_width and Img_height value"));
		msgBox.exec();
		return;
	}
	unsigned short *pRawData = (unsigned short *)calloc(Width*Height, sizeof(unsigned short));

	if (NULL == pRawData)
	{
		QMessageBox msgBox;
		msgBox.setText(tr("Fail to calloc buf"));
		msgBox.exec();
		return;
	}

	ifstream in(name.c_str());
	in.seekg(0, ios::end); //设置文件指针到文件流的尾部
	streampos ps = in.tellg(); //读取文件指针的位置
	in.close(); //关闭文件流

	if (NULL == (fp = fopen(name.c_str(), "rb")))
	{
		QMessageBox msgBox;
		msgBox.setText(tr("Fail to read"));
		msgBox.exec();
		return;
	}

	if (Width*Height * 2 != ps)
	{
		QMessageBox msgBox;
		msgBox.setText(tr("Width * Height Size does not match Raw Size!"));
		msgBox.exec();
		return;
	}

	ret = fread(pRawData, sizeof(unsigned short)*Width*Height, 1, fp);

	IplImage *pBayerData = cvCreateImage(cvSize(Width, Height), 16, 1);
	IplImage *pRgbDataInt8 = cvCreateImage(cvSize(Width, Height), 8, 1);

	memcpy(pBayerData->imageData, (char *)pRawData, Width*Height*sizeof(unsigned short));

	cvConvertScale(pBayerData, pRgbDataInt8, 0.25, 0);

	temp_image = cvarrToMat(pRgbDataInt8);
	img2 = temp_image.clone();
	gray_image = img2.clone();

	bool map = ui.GB->isChecked();
	if (map) {
		cvtColor(img2, img2, CV_BayerGR2BGR);
	}

	map = ui.GR->isChecked();
	if (map) {
		cvtColor(img2, img2, CV_BayerGB2BGR);
	}

	map = ui.BG->isChecked();
	if (map) {
		cvtColor(img2, img2, CV_BayerRG2BGR);
	}

	map = ui.RG->isChecked();
	if (map) {
		cvtColor(img2, img2, CV_BayerBG2BGR);
	}

	image = img2.clone();
	imageCopy = img2.clone();
	temp_image = img2.clone();

	display_Image();

	ui.pushButton_Xiaomi_SFR->setEnabled(true);

}

void color_Tester::on_pushButton_open_bmp_clicked() {

	QString filename = QFileDialog::getOpenFileName(this, tr("Open Image"), "", tr("Image File(*.bmp *.jpg *.jpeg *.png *.pbm *.pgm *.ppm)"));
	QTextCodec *code = QTextCodec::codecForName("gb18030");
	name = code->fromUnicode(filename).data();

	if (name.length() < 2) return;

	image = imread(name);
	
	imageCopy = image.clone();
	temp_image = image.clone();

	display_Image();

	ui.pushButton_Xiaomi_SFR->setEnabled(true);

}

void color_Tester::on_pushButton_Xiaomi_SFR_clicked() {

	ofstream fout_sfr(".\\Data\\xiaomi_SFR_Result.txt");
	char* pattern = "GRBG";

	if (ui.BG->isChecked())
		pattern = "BGGR";
	if (ui.GB->isChecked())
		pattern = "GBRG";
	if (ui.GR->isChecked())
		pattern = "GRBG";
	if (ui.RG->isChecked())
		pattern = "RGGB";

	int width = ui.width->document()->toPlainText().toInt();
	int height = ui.height->document()->toPlainText().toInt();

	//vector<float> spatialFeq{ 0,0,0,0 }; //nyquist频率
	float spatialFeq[4] = {0};

	int frqQuantity = 4;
	float* gridH = new float(0.0);
	float* gridW = new float(0.0);
	int gridLen = 120;

	TCHAR lpTexts[20];
	GetPrivateProfileString(TEXT("XIAOMISFR"), TEXT("Incline"), TEXT("8.0"), lpTexts, 8, TEXT(".\\Setting\\SFR_NewChart.set"));
	string SFR_setting = CT2A(lpTexts);
	float incline = atof(SFR_setting.c_str());
	GetPrivateProfileString(TEXT("XIAOMISFR"), TEXT("NyquistArray"), TEXT("2,3,6,8"), lpTexts, 18, TEXT(".\\Setting\\SFR_NewChart.set"));
	SFR_setting = CT2A(lpTexts);

	int x = 0, y = 0;

	while (x < SFR_setting.length() && y < 4) {

		string tmp = "";
		while (SFR_setting[x] != ','&&x < SFR_setting.length()) {
			tmp += SFR_setting[x++];
		}
		spatialFeq[y++] = atof(tmp.c_str());
		x++;
	}

	unsigned char* bmpdata = new unsigned char[width * height * 3];
	float* fFovValue = new float(0.0);
	float* fEFL = new float(0.0);

	float iObjectdis = 120; //模拟的chart距离
	float Gridsize = 10.0; //棋盘格边长
	float fMagnif = 1.0; //增距镜放大倍数（实距离/模拟距离），没有增距镜则 1
	float fpixelsize = 0.0064; //单位 pixel size（注意binning）

	int ret = 0;
	int iMode = GetPrivateProfileInt(_T("XIAOMISFR"), TEXT("SFR_Demo"), 1, TEXT(".\\Setting\\SFR_NewChart.set"));

	GetPrivateProfileString(TEXT("XIAOMISFR"), TEXT("GridH"), TEXT("0.0"), lpTexts, 8, TEXT(".\\Setting\\SFR_NewChart.set"));
	string tmp = "";
	tmp = CT2A(lpTexts);
	*gridH = atof(tmp.c_str());

	GetPrivateProfileString(TEXT("XIAOMISFR"), TEXT("GridW"), TEXT("0.0"), lpTexts, 8, TEXT(".\\Setting\\SFR_NewChart.set"));
	tmp = CT2A(lpTexts);
	*gridW = atof(tmp.c_str());

	int nyquist = GetPrivateProfileInt(_T("XIAOMISFR"), TEXT("Nyquist"), 0, TEXT(".\\Setting\\SFR_NewChart.set"));

	vector<SFR_RESULT_LIST> sfr_result;

	INPUTIMG_INFO input_image;
	input_image.bmpData = bmpdata;
	input_image.pattern = pattern;
	input_image.width = width;
	input_image.height = height;

	//FILE *fp = NULL;
	//if (NULL == (fp = fopen("raw8.raw", "rb")))
	//{
	//	QMessageBox msgBox;
	//	msgBox.setText(tr("Fail to read"));
	//	msgBox.exec();
	//	return;
	//}

	//	input_image.rawDate = new unsigned char[input_image.width * input_image.height];
	//	ret = fread(input_image.rawDate, sizeof(unsigned char)*input_image.width*input_image.height, 1, fp);

	//for (int i = 0; i < gray_image.rows; i++) {
	//	uchar* inData = gray_image.ptr<uchar>(i);
	//	for (int j = 0; j < gray_image.cols; j++) {
	//		if (i<gray_image.rows/3&&j<gray_image.cols/3) {
	//			inData[j] = 0;
	//		}
	//	}
	//}

	input_image.rawDate = gray_image.data;
	string tag = "Mode" + to_string(iMode);

	typedef int(*pfnSFRDemoFunc_mode3)(int iMode, std::vector<SFR_RESULT_LIST> &SFRTestResult,
		INPUTIMG_INFO inImageInfo, float* spatialFeq, int frqQuantity,
		float* gridH, float* gridW, int gridLen,
		char* output, char* tag, float incline);

	HINSTANCE m_hDll = LoadLibrary(_T("Sharpness_bilinear_dll.dll"));

	pfnSFRDemoFunc_mode3 SFRDemoFunc_mode3 = (pfnSFRDemoFunc_mode3)GetProcAddress(m_hDll, "?SFRDemoFunc_mode3@@YAHHAEAV?$vector@USFR_RESULT_LIST@@V?$allocator@USFR_RESULT_LIST@@@std@@@std@@UINPUTIMG_INFO@@PEAMH22HPEAD3M@Z");
	ret = SFRDemoFunc_mode3(iMode, sfr_result,input_image, spatialFeq, frqQuantity, gridH, gridW, gridLen, NULL, NULL, incline);

	delete[] bmpdata;
	bmpdata = NULL;

	if (ret) {
		string r_str = to_string(ret);
		r_str = "Xiaomi SFR DLL ret: " + r_str + "\n";
		ui.log->insertPlainText(r_str.c_str());
		return;
	}

	if (nyquist > 3)nyquist = 0;
	if (iMode == 0) {
		for (int i = 0; i < sfr_result.size(); i++) {
			if (i == 0) {
				fout_sfr << "group_1_1" << "\t" << sfr_result[i].SFRValue[nyquist].HValue << endl;
				fout_sfr << "group_1_2" << "\t" << sfr_result[i].SFRValue[nyquist].VValue << endl;
			}
			if (i >= 1 && i <= 4) {
				fout_sfr << "group_2_" << 2 * i - 1 << "\t" << sfr_result[i].SFRValue[nyquist].HValue << endl;
				fout_sfr << "group_2_" << 2 * i - 1 + 1 << "\t" << sfr_result[i].SFRValue[nyquist].VValue << endl;
			}
		}
	}
	if (iMode == 11 || iMode == 12) {
		for (int i = 0; i < sfr_result.size(); i++) {
			if (i == 0) {
				fout_sfr << "group_1_1" << "\t" << sfr_result[i].SFRValue[nyquist].HValue << endl;
				fout_sfr << "group_1_2" << "\t" << sfr_result[i].SFRValue[nyquist].VValue << endl;
			}
			if (i >= 1 && i <= 4) {
				fout_sfr << "group_2_" << 2 * i - 1 << "\t" << sfr_result[i].SFRValue[nyquist].HValue << endl;
				fout_sfr << "group_2_" << 2 * i - 1 + 1 << "\t" << sfr_result[i].SFRValue[nyquist].VValue << endl;
			}
			if (i >= 5 && i <= 8) {
				fout_sfr << "group_3_" << 2 * i - 9 << "\t" << sfr_result[i].SFRValue[nyquist].HValue << endl;
				fout_sfr << "group_3_" << 2 * i - 9 + 1 << "\t" << sfr_result[i].SFRValue[nyquist].VValue << endl;
			}
			if (i >= 9 && i <= 13) {
				fout_sfr << "group_4_" << 2 * i - 17 << "\t" << sfr_result[i].SFRValue[nyquist].HValue << endl;
				fout_sfr << "group_4_" << 2 * i - 17 + 1 << "\t" << sfr_result[i].SFRValue[nyquist].VValue << endl;
			}
		}
	}
	if (iMode == 21) {
		for (int i = 0; i < sfr_result.size(); i++) {
			if (i == 0) {
				fout_sfr << "group_1_1" << "\t" << sfr_result[i].SFRValue[nyquist].HValue << endl;
				fout_sfr << "group_1_2" << "\t" << sfr_result[i].SFRValue[nyquist].VValue << endl;
			}
			if (i >= 1 && i <= 4) {
				fout_sfr << "group_2_" << 2 * i - 1 << "\t" << sfr_result[i].SFRValue[nyquist].HValue << endl;
				fout_sfr << "group_2_" << 2 * i - 1 + 1 << "\t" << sfr_result[i].SFRValue[nyquist].VValue << endl;
			}
			if (i >= 5 && i <= 12) {
				fout_sfr << "group_3_" << 2 * i - 9 << "\t" << sfr_result[i].SFRValue[nyquist].HValue << endl;
				fout_sfr << "group_3_" << 2 * i - 9 + 1 << "\t" << sfr_result[i].SFRValue[nyquist].VValue << endl;
			}
			if (i >= 13 && i <= 18) {
				fout_sfr << "group_4_" << 2 * i - 25 << "\t" << sfr_result[i].SFRValue[nyquist].HValue << endl;
				fout_sfr << "group_4_" << 2 * i - 25 + 1 << "\t" << sfr_result[i].SFRValue[nyquist].VValue << endl;
			}
		}
	}
	if (iMode == 22) {
		for (int i = 0; i < sfr_result.size(); i++) {
			if (i == 0) {
				fout_sfr << "group_1_1" << "\t" << sfr_result[i].SFRValue[nyquist].HValue << endl;
				fout_sfr << "group_1_2" << "\t" << sfr_result[i].SFRValue[nyquist].VValue << endl;
			}
			if (i >= 1 && i <= 4) {
				fout_sfr << "group_2_" << 2 * i - 1 << "\t" << sfr_result[i].SFRValue[nyquist].HValue << endl;
				fout_sfr << "group_2_" << 2 * i - 1 + 1 << "\t" << sfr_result[i].SFRValue[nyquist].VValue << endl;
			}
			if (i >= 5 && i <= 12) {
				fout_sfr << "group_3_" << 2 * i - 9 << "\t" << sfr_result[i].SFRValue[nyquist].HValue << endl;
				fout_sfr << "group_3_" << 2 * i - 9 + 1 << "\t" << sfr_result[i].SFRValue[nyquist].VValue << endl;
			}
			if (i >= 13 && i <= 14) {
				fout_sfr << "group_4_" << 2 * i - 25 << "\t" << sfr_result[i].SFRValue[nyquist].HValue << endl;
				fout_sfr << "group_4_" << 2 * i - 25 + 1 << "\t" << sfr_result[i].SFRValue[nyquist].VValue << endl;
			}
			if (i >= 15 && i <= 18) {
				fout_sfr << "group_5_" << 2 * i - 29 << "\t" << sfr_result[i].SFRValue[nyquist].HValue << endl;
				fout_sfr << "group_5_" << 2 * i - 29 + 1 << "\t" << sfr_result[i].SFRValue[nyquist].VValue << endl;
			}
		}
	}
	fout_sfr << endl;
	fout_sfr.close();

	ifstream in(".\\Data\\xiaomi_SFR_Result.txt");
	ostringstream outStr;
	outStr << in.rdbuf();
	string outContent = outStr.str();
	ui.log->insertPlainText(outContent.c_str());

	string result_img = tag + "Result.bmp";
	image = imread(result_img);
	display_Image();

	ui.log->insertPlainText("Xiaomi SFR Test Done!\n");
	in.close();
	fout_sfr.close();
}

void color_Tester::on_pushButton_Iris_clicked() {
	ui.log->clear();

	TCHAR lpTexts[256] = { 0 };
	string dino_path = "";

	GetPrivateProfileString(TEXT("TEST_OPTION"), TEXT("path"), TEXT(""), lpTexts, 255, TEXT(".\\Setting\\specValue.ini"));
	dino_path = TCHAR2STRING(lpTexts);
	if (dino_path.length() < 5)return;

#ifdef  PYTHON_DLL

	string input_img_path = "D:\\Project\\color_Tester\\color_Tester\\Data\\Segmentation_Multi\\Images\\Test\\Scan_4095_OC1.bmp";
	PyObject* args2 = Py_BuildValue("Os", mBestModel, input_img_path.c_str());//给python函数参数赋值

	PyObject* pRet2 = PyObject_CallObject(pFunc2, args2);//调用函数PyObject_CallFunctionObjArgs 
	if (pRet2 == NULL) {
		PyErr_Print();  // 打印错误信息
		ui.log->setText("Call Python Function pFunc2 failed!\n");
		return;
	}
	if (PyErr_Occurred()) {
		PyErr_Print();
		ui.log->setText("Python error occurred!\n");
		return;
	}
	int res = 0;

	//PyArg_ParseTuple(pRet2, "i", &res);//分析返回的元组值

	if (!PyArg_Parse(pRet2, "i", &res)) {
		ui.log->setText("Failed to parse Python return value to double!\n");
		return;
	}

#endif

	ui.textBrowser_cnt->setFontPointSize(48);
	ui.textBrowser_cnt->setText(QString::number(res));
	ui.textBrowser_cnt->setAlignment(Qt::AlignCenter);
}

int OutputLogFunction(std::string log)
{
	fout << log << endl;
	return 0;
}

int FourDirectionSFR(UINT width, UINT height, BYTE *p_src, BYTE *p_dst3, std::vector<std::pair<double, double>> dect_pos,
	int roi_length, int roi_width, double spatial_freq_denominator, double gamma,
	std::vector<double>& sfr_h_w2b, std::vector<double>& sfr_h_b2w, std::vector<double>& sfr_v_w2b, std::vector<double>& sfr_v_b2w, bool save_image)
{
	int ret = 0;

	sfr_h_w2b.clear();
	sfr_h_b2w.clear();
	sfr_v_w2b.clear();
	sfr_v_b2w.clear();

	cv::Mat gray_img = cv::Mat(height, width, CV_8UC1, (void *)p_src);

	int ret_locate_edge = 0;

	std::vector<cv::Mat> edge_img_h_w2b;
	std::vector<cv::Mat> edge_img_h_b2w;
	std::vector<cv::Mat> edge_img_v_w2b;
	std::vector<cv::Mat> edge_img_v_b2w;

	ChessBoard_CornerDetection detector;

	ret_locate_edge += detector.Init(gray_img, OutputLogFunction,".\\image\\",1);

	ChessBoard_CornerDetection::SfrEdgeDectOption dect_option;
	dect_option.roi_length = roi_length;
	dect_option.roi_width = roi_width;
	dect_option.dect_pos = dect_pos;
	
	ret_locate_edge += detector.ProcessGetChessSfrEdgeImages_4Dir(edge_img_h_w2b, edge_img_h_b2w, edge_img_v_w2b, edge_img_v_b2w, dect_option);

	if (ret_locate_edge) {
		ret += 1;
	}

	cv::Mat marked_img;
	detector.GetMarkedImage(marked_img);

	for (int i = 0; i < marked_img.rows; ++i) {
		memcpy(p_dst3, marked_img.ptr<uchar>(i), 3 * width);
		p_dst3 += width * 3;
	}

	if (save_image) {
		cv::imwrite(".\\image\\SFR_1.bmp", marked_img);
	}

	SFR_Calculator sfr_calc;
	SFR_Calculator::SFR_Option sfr_option;
	sfr_option.spatial_freq_denominator = spatial_freq_denominator;
	sfr_option.gamma = gamma;

	for (int point_cnt = 0; point_cnt < dect_pos.size(); point_cnt++) {

		std::string str_field = std::to_string(dect_pos[point_cnt].first);
		str_field = str_field.substr(0, str_field.find(".") + 2);
		std::string str_degree = std::to_string(dect_pos[point_cnt].second);
		str_degree = str_degree.substr(0, str_degree.find(".") + 2);

		for (int dir = 0; dir < 4; dir++) {

			cv::Mat cur_edge_img;
			std::string edge_name;

			switch (dir) {
			case 0:
				cur_edge_img = edge_img_h_w2b[point_cnt];
				sfr_option.edge_type = (int)SFR_Calculator::EdgeType::kHorizontal;
				edge_name = "Field_" + str_field + "_Degree_" + str_degree + "_Edge_H_WhiteToBlack";
				break;
			case 1:
				cur_edge_img = edge_img_h_b2w[point_cnt];
				sfr_option.edge_type = (int)SFR_Calculator::EdgeType::kHorizontal;
				edge_name = "Field_" + str_field + "_Degree_" + str_degree + "_Edge_H_BlackToWhite";
				break;
			case 2:
				cur_edge_img = edge_img_v_w2b[point_cnt];
				sfr_option.edge_type = (int)SFR_Calculator::EdgeType::kVertical;
				edge_name = "Field_" + str_field + "_Degree_" + str_degree + "_Edge_V_WhiteToBlack";
				break;
			case 3:
				cur_edge_img = edge_img_v_b2w[point_cnt];
				sfr_option.edge_type = (int)SFR_Calculator::EdgeType::kVertical;
				edge_name = "Field_" + str_field + "_Degree_" + str_degree + "_Edge_V_BlackToWhite";
				break;
			default:
				return 1;
			}

			double sfr_result = -1.0;
			if (!cur_edge_img.empty()) {
				int ret_sfr = sfr_calc.Init(cur_edge_img, OutputLogFunction, ".\\image\\", false, false, edge_name);
				ret_sfr += sfr_calc.ProcessSFR(sfr_result, sfr_option);
				if (ret_sfr) {
					sfr_result = -1.0;
					ret += 1;
				}
			}
			else {
				sfr_result = -1.0;
			}

			switch (dir) {
			case 0: sfr_h_w2b.push_back(sfr_result); break;
			case 1: sfr_h_b2w.push_back(sfr_result); break;
			case 2: sfr_v_w2b.push_back(sfr_result); break;
			case 3: sfr_v_b2w.push_back(sfr_result); break;
			default:
				return 1;
			}
		}
	}

	return ret;
}

void color_Tester::on_pushButton_SFR_ROI4_clicked() {

	fout.open("SFR_ROI_4.txt");
	cvtColor(imageCopy, gray_image, CV_BGR2GRAY);
	std::vector<std::pair<double, double>> dect_pos;	//first:field, second:degree

	dect_pos.push_back(std::make_pair(0, 0));
	dect_pos.push_back(std::make_pair(0.3, 37));  // right down
	dect_pos.push_back(std::make_pair(0.3, 143));   // left down
	dect_pos.push_back(std::make_pair(0.3, 217));   // left Top
	dect_pos.push_back(std::make_pair(0.3, 323));   // right Top

	int roi_length = 100;
	int roi_width = 60;
	double spatial_freq_denominator = 3.0;
	double gamma = 0.3;
	bool save_image = true;

	std::vector<double> sfr_h_w2b;
	std::vector<double> sfr_h_b2w;
	std::vector<double> sfr_v_w2b;
	std::vector<double> sfr_v_b2w;

	int return_value = FourDirectionSFR(gray_image.cols, gray_image.rows, gray_image.data, imageCopy.data, dect_pos, roi_length, roi_width, spatial_freq_denominator, gamma,
		sfr_h_w2b, sfr_h_b2w, sfr_v_w2b, sfr_v_b2w, save_image);

	for (int i = 0; i < 5; i++) {	
		fout << "Field_" << i << "_H:	" << sfr_h_w2b[i] << "	" << sfr_h_b2w[i] << "	" << sfr_h_w2b[i] - sfr_h_b2w[i] << endl;
		fout << "Field_" << i << "_V:	" << sfr_v_w2b[i] << "	" << sfr_v_b2w[i] << "	" << sfr_v_w2b[i] - sfr_v_b2w[i] << endl;
	}

	fout.close();
}




