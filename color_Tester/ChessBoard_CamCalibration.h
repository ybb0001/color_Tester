#pragma once
#include<afx.h>
#include<io.h>
#include<vector>
#include<string>
//#include<Eigen\Dense>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
//#include "opencv2/core/eigen.hpp"

//#include<opencv2\highgui\highgui.hpp>
//#include<opencv2\core\core.hpp>
//#include<opencv2\core\eigen.hpp>
//#include<opencv2\imgproc\imgproc.hpp>
//#include<opencv2\calib3d\calib3d.hpp>
//#ifdef _DEBUG
//#pragma comment(lib,"opencv_core249d.lib")
//#pragma comment(lib,"opencv_highgui249d.lib")
//#pragma comment(lib,"opencv_imgproc249d.lib")
//#pragma comment(lib,"opencv_calib3d249d.lib")
//#else
//#pragma comment(lib,"opencv_core249.lib")
//#pragma comment(lib,"opencv_highgui249.lib")
//#pragma comment(lib,"opencv_imgproc249.lib")
//#pragma comment(lib,"opencv_calib3d249.lib")
//#endif

#define FIX_U0_V0_GAMMA
#define DISTORTION_ACCURACY 0
/*
DISTORTION_ACCURACY == 0: radial dis coeff = (1+k1*r2+k2*r2^2+k3*r2^3)
DISTORTION_ACCURACY == 1: radial dis coeff = (1+k1*r2+k2*r2^2+k3*r2^3)/(1+k4*r2+k5*r2^2+k6*r2^3)
*/

class ChessBoard_CamCalibration {
public:
	ChessBoard_CamCalibration();
	virtual ~ChessBoard_CamCalibration();

private:
	//单目标定
	int GetMatrixH(std::vector<cv::Point2f> src_Points, std::vector<cv::Point2f> dest_Points, cv::Mat& matrix_H);
	int GetInitialIntrinsicParams(std::vector<cv::Mat> matrix_H, int imgWidth, int imgHeight, cv::Mat& matrix_M_init, bool bFix_u0v0Gamma/*, double pixelAspectRatio = -1, bool bUse_pixelAspectRatio = false*/);
	int GetInitialExtrinsicParams(std::vector<cv::Mat> matrix_H, cv::Mat matrix_M, std::vector<cv::Mat>& vec_R_init, std::vector<cv::Mat>& vec_T_init);
	int OptimizeSingleCameraParams_LevMarq(std::vector<std::vector<cv::Point3f>> objectPoints, std::vector<std::vector<cv::Point2f>> imagePoints,
		cv::Mat& matrix_M, bool bFixMatrixM, cv::Mat& matrix_K, cv::Mat& matrix_P, std::vector<cv::Mat>& vec_R, std::vector<cv::Mat>& vec_T, std::vector<double>& error, double& avgError);

	typedef int(*pOutputLog)(std::string log);
	pOutputLog m_OutputLog;

public:
	/**
	* \brief 初始化赋值
	* \param OutputLogFuntion	输入，设置打印 log 函数，默认为 NULL
	* \return 空
	*/
	void Init(pOutputLog OutputLogFuntion = NULL);

	/**
	* \brief 执行单目标定
	* \param objectPoints		输入，每张标定板图片对应的三维坐标
	* \param imagePoints		输入，每张标定板图片中提取到的二维坐标
	* \param imgWidth			输入，像素宽度
	* \param imgHeight			输入，像素高度
	* \param matrix_M			输入/输出，标定结果：内参
	* \param bFixMatrixM		输入，是否使用输入的内参
	* \param matrix_K			输出，标定结果：径向畸变系数
	* \param matrix_P			输出，标定结果：切向畸变系数
	* \param vec_R				输出，标定结果：每张图像对应的旋转向量
	* \param vec_T				输出，标定结果：每张图像对应的平移向量
	* \param error				输出，每张图像的重投影误差
	* \param avgError			输出，总体平均误差
	* \return 执行无误返回 0，执行错误返回 -1
	*/
	int ProcessSingleCamCal(std::vector<std::vector<cv::Point3f>> objectPoints, std::vector<std::vector<cv::Point2f>> imagePoints, int imgWidth, int imgHeight,
		cv::Mat& matrix_M, bool bFixMatrixM, cv::Mat& matrix_K, cv::Mat& matrix_P, std::vector<cv::Mat>& vec_R, std::vector<cv::Mat>& vec_T, std::vector<double>& error, double& avgError);
	/**
	* \brief 获取单应性矩阵
	* \param points_src			输入，源图像特征点坐标
	* \param points_dest		输入，变换后图像特征点坐标
	* \param matrix_H			输出，单应性矩阵
	* \return 执行无误返回 0，执行错误返回 -1
	*/
	int ProcessGetHomography(std::vector<cv::Point2f> points_src, std::vector<cv::Point2f> points_dest, cv::Mat& matrix_H);
	/**
	* \brief 根据相机参数，重投影物体三维坐标至画面二维坐标，可选择是否输出关于相机各参数的一阶微分矩阵
	* \param objectPoints		输入，物体三维坐标
	* \param matrix_M			输入，内参
	* \param matrix_K			输入，径向畸变系数
	* \param matrix_P			输入，切向畸变系数
	* \param vec_R				输入，旋转向量
	* \param vec_T				输入，平移向量
	* \param projectionPoints	输出，重投影后的画面二维坐标
	* \param dpdf				输出，关于内参fx/fy的一阶微分矩阵
	* \param dpdc				输出，关于内参u0/v0的一阶微分矩阵
	* \param dpdk				输出，关于径向畸变系数的一阶微分矩阵
	* \param dpdp				输出，关于径向畸变系数的一阶微分矩阵
	* \param dpdr				输出，关于旋转向量的一阶微分矩阵
	* \param dpdt				输出，关于平移向量的一阶微分矩阵
	* \return 执行无误返回 0，执行错误返回 -1
	*/
	int ProjectPoints(std::vector<cv::Point3f> objectPoints, cv::Mat matrix_M, cv::Mat matrix_K, cv::Mat matrix_P, cv::Mat vec_R, cv::Mat vec_T, std::vector<cv::Point2f>& projectionPoints,
		cv::Mat* dpdf = NULL, cv::Mat* dpdc = NULL, cv::Mat* dpdk = NULL, cv::Mat* dpdp = NULL, cv::Mat* dpdr = NULL, cv::Mat* dpdt = NULL);

	/**
	* \brief 旋转矩阵转换为旋转角度
	* \param rotationMatrix	输入，旋转矩阵
	* \param pitch				输出，俯仰角
	* \param yaw				输出，偏航角
	* \param roll				输出，滚转角
	* \return 执行无误返回 0，执行错误返回 -1
	*/
	int RotationMatrixToEulerAngles(cv::Mat rotationMatrix, double& pitch, double& yaw, double& roll);
};