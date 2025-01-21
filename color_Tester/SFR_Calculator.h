#pragma once
#include<afx.h>
#include<io.h>
#include<vector>
#include<string>
#include<fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>


#include"fftw3.h"
#pragma comment(lib, "libfftw3-3.lib")

#define GLOBAL_LINEAR_FIT_RMSE_THRESHOLD 1.0
#define SPLIT_LINEAR_FIT_RMSE_THRESHOLD 0.1
#define SPLIT_LINEAR_MIN_AREA_HEIGHT 8
#define LSF_DERIVATIVE_FILTER 1	// 0 = [-1 1], 1 = [-1/2 0 1/2]

class SFR_Calculator {
public:
	SFR_Calculator();
	virtual ~SFR_Calculator();

	/** @brief 边缘类型 */
	enum class EdgeType {
		kNotDefinded,	///< 未指定
		kHorizontal,	///< 水平边缘
		kVertical,	///< 垂直边缘
	};

	/** @brief 傅里叶变换方式 */
	enum class DFT_Type {
		kNormalDFT,
		kFFTW3,
	};

	/** @brief SFR 计算参数结构体 */
	struct SFR_Option {
		double spatial_freq_denominator;	// 空间频率，1/x * Nyquist Freq
		double gamma;	// 输入图的 Gamma 编码系数
		int dft_type;	// 傅里叶变换方式：0:NORMAL_DFT，1:FFTW3_DFT，默认为 0
		int edge_type;	// 指定边缘类型：0:不指定（自动判定），1:水平边缘，2:垂直边缘，默认为 0
		bool use_user_defined_edge_position;	// 如果已经计算好了边缘确切位置，可选择使用计算好的坐标（边缘类型为 不指定 时不生效）
		std::vector<double> edge_pos;	// 计算好的边缘确切位置坐标（边缘类型为 水平边缘 时输入 y 轴坐标，边缘类型为 垂直边缘 时输入 x 轴坐标）

		SFR_Option() :gamma(1.0), spatial_freq_denominator(1.0), dft_type((int)DFT_Type::kFFTW3), edge_type((int)EdgeType::kNotDefinded),
			use_user_defined_edge_position(false) {
			edge_pos.clear();
		}
	};

private:
	/** @brief 边缘梯度类型 */
	enum class EdgeBrightnessDirection {
		kNotDefinded,	///< 未指定
		kBlackToWhite,	///< 左黑右白
		kWhiteToBlack,	///< 左白右黑
	};

	//SFR 计算
	int checkEdgeTypeAndRotateRoi();
	void imageNormalization();
	void deGamma(double gamma);
	int calcEdgeLine();
	void calcEdgePostion();
	int overSampling4x();
	void calcLsf();
	void centeringLsf(bool calc_lsf_centroid_method);
	void hammingWindow(std::vector<double>& data, double cen_offset = 0.0);
	void runDft();
	void runDft_fftw3();
	void calcSfrArray();
	double getSfrWithSpatialFreq(double spatial_freq);
	int calcRowCentroids(bool use_hamming, bool is_linear);
	void findMaxDiffInRow();
	int fitLineLeastSquares(std::vector<double> data, double& slope, double& intercept, int idx_offset = 0, bool output_log = true);
	int calcFitErr(std::vector<double> data, double slope, double intercept, double& rmse, int idx_offset = 0, bool output_log = true);
	int splitLinearAreaDP_CheckLinearAreaHeightLeastOnce();
	int checkSlope(double edge_slope, bool output_log = true);
	int checkHeightAndReduceHeight(double edge_slope, int* edge_height, bool output_log = true, int* reduce_height = nullptr);
	int checkFitErr(double err, double err_boundary, bool output_log = true);

	//画图存数据
	void SaveGrayImg(std::string folder, std::string img_name, cv::Mat gray_img, int width, int height);
	void SaveRGBImg(std::string folder, std::string img_name, cv::Mat rgb_img, int width, int height);
	void drawCentroidsOnImg(cv::Mat& img, std::vector<double> row_centroids, double centroid_offset = 0.0);
	void drawFitLineOnImg(cv::Mat& img, double slope, double intercept);
	void drawFitLineOnImg(cv::Mat& img, std::vector<double> slope, std::vector<double> intercept, std::vector<bool> is_linear_area);
	void saveFitLineData(std::string folder, std::string file_name, double slope, double intercept, double err);
	void saveFitLineData(std::string folder, std::string file_name, std::vector<double> slope, std::vector<double> intercept, std::vector<double> err, std::vector<bool> is_linear_area);
	void saveArrayData(std::string folder, std::string file_name, std::vector<double> arr);
	void saveArrayData(std::string folder, std::string file_name, std::vector<double> arr1, std::vector<double> arr2);
	void saveArrayData(std::string folder, std::string file_name, std::vector<std::complex<double>> arr);

	cv::Mat edge_img_;
	cv::Mat edge_normalized_;
	int edge_width_;
	int edge_height_;

	EdgeType edge_type_;
	EdgeBrightnessDirection edge_brightness_direction_;
	DFT_Type dft_type_;

	std::vector<int> row_max_diff_pos_;
	std::vector<double> row_centroids_;

	double slope_;
	double intercept_;
	double err_;
	bool split_;
	std::vector<double> split_slope_;
	std::vector<double> split_intercept_;
	std::vector<double> split_err_;
	std::vector<bool> split_is_linear_area_;
	std::vector<double> calc_edge_pos_;
	int far_left_calc_edge_pos_, far_right_calc_edge_pos_, avg_calc_edge_pos_;
	bool use_user_defined_edge_position_;

	std::vector<double> esf_;
	std::vector<int> esf_add_cnt_;
	std::vector<double> lsf_;

	std::vector<std::complex<double>> spectrum_;
	std::vector<double> sfr_;
	std::vector<double> freq_;


	typedef int (*pOutputLog)(std::string log);
	pOutputLog OutputLog;
	std::string result_folder_path_;
	bool save_img_;
	bool save_data_;
	std::string test_process_prefix_;

public:
	/**
	 * @brief 初始化赋值
	 * @param [in] edge_gray_img				用于计算 SFR 的刀边灰图
	 * @param [in] output_log_funtion		设置打印 log 函数，默认为 NULL
	 * @param [in] result_folder_path		输出 Debug 用数据及图片的文件夹，默认为 空
	 * @param [in] save_img							是否输出过程 Debug 用图片，默认为 false
	 * @param [in] save_data						是否输出过程 Debug 用数据，默认为 false
	 * @param [in] test_process_prefix	当前图像的子名称（用于打印异常 log 时标识用），默认为 空
	 * @return 0												成功
	 * @return -1												失败
	*/
	int Init(cv::Mat& edge_gray_img, pOutputLog output_log_funtion = NULL, std::string result_folder_path = "", bool save_img = false, bool save_data = false, std::string test_process_prefix = "");

	/**
	 * @brief 执行 SFR 计算
	 * @param [out] sfr_result					SFR 计算结果
	 * @param [in] sfr_option						SFR 计算参数
	 * @return 0												成功
	 * @return -1												失败
	*/
	int ProcessSFR(double& sfr_result, SFR_Option sfr_option);
};