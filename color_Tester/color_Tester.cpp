#include "color_Tester.h"
#include <Windows.h>
#include <direct.h>
#include <String>
#include <direct.h>
#include <stdlib.h>
#include <stdio.h>
#include <io.h>
#include <iostream>     
#include <fstream>   
#include <Python.h>

int NG = 0, Color_Tcnt = 0;
ofstream fout;
PyObject * pModule = NULL;//声明变量
PyObject * pFunc3 = NULL;// 声明变量


color_Tester::color_Tester(QWidget *parent)
	: QWidget(parent)
{
	ui.setupUi(this);
	//connect(ui.pushButton_clear, SIGNAL(clicked()), this, SLOT(on_pushButton_clear_clicked()));
	//connect(ui.pushButton_Color_Diff, SIGNAL(clicked()), this, SLOT(on_pushButton_Color_Diff_clicked()));
	//connect(ui.pushButton_cnt_reduce, SIGNAL(clicked()), this, SLOT(on_pushButton_cnt_reduce_clicked()));
	Py_Initialize();//使用python之前，要调用Py_Initialize();这个函数进行初始化

	if (!Py_IsInitialized())	{
		ui.log->setText("Python Initialize Fail!\n");
	}
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./')");//这一步很重要，修改Python路径
	pModule = PyImport_ImportModule("color_test");//这里是要调用的文件名hello.py
	if (pModule == NULL)	{
		ui.log->setText("Python pModule Initialize Fail!\n");
	}

	pFunc3 = PyObject_GetAttrString(pModule, "color_check");//这里是要调用的函数名

	if (!pFunc3 || !PyCallable_Check(pFunc3)) {
		ui.log->setText("Python PyObject Initialize Fail!\n");
	}
}

void color_Tester::on_pushButton_clear_clicked() {
	ui.log->clear();
}

void color_Tester::displayResult() {

	ui.log->setFontPointSize(48);
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

	displayResult();
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

	PyObject* args2 = Py_BuildValue("ddd", BGR_avg[2], BGR_avg[1], BGR_avg[0]);//给python函数参数赋值
	PyObject* pRet = PyObject_CallObject(pFunc3, args2);//调用函数

	double res = 0;
	PyArg_Parse(pRet, "d", &res);//转换返回类型
	if (res>0.5)NG = 1;
	else NG = 2;

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

	return res;
}

void color_Tester::on_pushButton_cnt_reduce_clicked() {

	Color_Tcnt--;
	if (Color_Tcnt < 0)Color_Tcnt = 0;
	ui.textBrowser_cnt->setFontPointSize(48);
	ui.textBrowser_cnt->setText(QString::number(Color_Tcnt));
	ui.textBrowser_cnt->setAlignment(Qt::AlignCenter);

}

