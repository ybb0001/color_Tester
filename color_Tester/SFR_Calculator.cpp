#include"SFR_Calculator.h"

SFR_Calculator::SFR_Calculator() {
	edge_width_ = 0;
	edge_height_ = 0;

	edge_type_ = EdgeType::kNotDefinded;
	edge_brightness_direction_ = EdgeBrightnessDirection::kNotDefinded;
	dft_type_ = DFT_Type::kNormalDFT;

	slope_ = 0.0;
	intercept_ = 0.0;
	err_ = 0.0;
	split_ = false;
	far_left_calc_edge_pos_ = 0;
	far_right_calc_edge_pos_ = 0;
	avg_calc_edge_pos_ = 0;
	use_user_defined_edge_position_ = false;

	OutputLog = NULL;
	result_folder_path_ = "";
	save_img_ = false;
	save_data_ = false;
	test_process_prefix_ = "";
}

SFR_Calculator::~SFR_Calculator() {
	if (!edge_img_.empty()) {
		edge_img_.release();
	}
	if (!edge_normalized_.empty()) {
		edge_normalized_.release();
	}
}

int SFR_Calculator::Init(cv::Mat& edge_gray_img, pOutputLog output_log_funtion, std::string result_folder_path, bool save_img, bool save_data, std::string test_process_prefix) {

	OutputLog = output_log_funtion;
	result_folder_path_ = result_folder_path;
	save_img_ = save_img;
	save_data_ = save_data;
	test_process_prefix_ = test_process_prefix;

	if (edge_gray_img.type() != CV_8UC1) {
		if (OutputLog) {
			OutputLog("Image Type error");
		}
		return -1;
	}

	if (!edge_img_.empty()) {
		edge_img_.release();
	}
	edge_img_ = edge_gray_img.clone();
	edge_width_ = edge_img_.cols;
	edge_height_ = edge_img_.rows;

	edge_type_ = EdgeType::kNotDefinded;
	edge_brightness_direction_ = EdgeBrightnessDirection::kNotDefinded;

	row_max_diff_pos_.clear();
	row_centroids_.clear();

	slope_ = 0.0;
	intercept_ = 0.0;
	err_ = 0.0;
	split_ = false;
	split_slope_.clear();
	split_intercept_.clear();
	split_err_.clear();
	split_is_linear_area_.clear();
	calc_edge_pos_.clear();
	far_left_calc_edge_pos_ = 0;
	far_right_calc_edge_pos_ = 0;
	avg_calc_edge_pos_ = 0;

	esf_.clear();
	esf_add_cnt_.clear();
	lsf_.clear();

	spectrum_.clear();
	sfr_.clear();
	freq_.clear();

	return 0;
}

int SFR_Calculator::ProcessSFR(double& sfr_result, SFR_Option sfr_option) {

	sfr_result = -1.0;

	if (edge_img_.empty()) {
		if (OutputLog) {
			OutputLog("Empty Image");
		}
		return -1;
	}

	dft_type_ = (DFT_Type)sfr_option.dft_type;

	edge_type_ = (EdgeType)sfr_option.edge_type;
	if (edge_type_ != EdgeType::kNotDefinded && edge_type_ != EdgeType::kHorizontal && edge_type_ != EdgeType::kVertical) {
		edge_type_ = EdgeType::kNotDefinded;
	}

	use_user_defined_edge_position_ = sfr_option.use_user_defined_edge_position;
	if (use_user_defined_edge_position_ == true) {
		if (edge_type_ == EdgeType::kNotDefinded) {
			if (OutputLog)	OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "if need to define edge position, define edge type first");
			return -1;
		} else {
			calc_edge_pos_ = sfr_option.edge_pos;
			if (edge_type_ == EdgeType::kHorizontal) {
				if (calc_edge_pos_.size() != edge_width_) {
					if (OutputLog)	OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "for horizontal edge, if need to define edge position, the input edge pos vector should be Y coordinate, size should same as image width");
					return -1;
				}
			} else if (edge_type_ == EdgeType::kVertical) {
				if (calc_edge_pos_.size() != edge_height_) {
					if (OutputLog)	OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "for vertical edge, if need to define edge position, the input edge pos vector should be X coordinate, size should same as image height");
					return -1;
				}
			}
		}
	}


	if (save_img_) {
		SaveGrayImg(result_folder_path_, "0_Img_Edge", edge_img_, edge_img_.cols, edge_img_.rows);
	}

	//1. 统一为 V 方向刀边
	if (checkEdgeTypeAndRotateRoi()) {
		return -1;
	}
	if (save_img_) {
		SaveGrayImg(result_folder_path_, "1_Img_JudgeAndRotate", edge_img_, edge_img_.cols, edge_img_.rows);
	}

	//2. 归一化
	imageNormalization();

	//3. Gamma 恢复
	if (sfr_option.gamma != 1.0) {
		deGamma(sfr_option.gamma);
		if (save_img_) {
			SaveGrayImg(result_folder_path_, "2_Img_DeGamma", edge_img_, edge_img_.cols, edge_img_.rows);
		}
	}

	//4. 计算每行矩心，线性拟合，获取 edge line
	if (use_user_defined_edge_position_ == false) {
		if (calcEdgeLine()) {
			return -1;
		}
	}

	//5. 计算每行 edge 确切坐标
	calcEdgePostion();

	//6. 四倍超采样，获取 ESF
	if (overSampling4x()) {
		if (save_data_) {
			saveArrayData(result_folder_path_, "11_ESF_OverSampling4x_NG", esf_);
		}
		return -1;
	}
	if (save_data_) {
		saveArrayData(result_folder_path_, "11_ESF", esf_);
	}

	//7. 一阶微分，获取 LSF
	calcLsf();
	if (save_data_) {
		saveArrayData(result_folder_path_, "12_LSF", lsf_);
	}

	//8. 获取的 LSF 曲线置中
	centeringLsf(true);
	if (save_data_) {
		saveArrayData(result_folder_path_, "13_LSF_Centering", lsf_);
	}

	//9. 应用汉明窗
	hammingWindow(lsf_);
	if (save_data_) {
		saveArrayData(result_folder_path_, "14_LSF_Hamming", lsf_);
	}

	//10. DFT
	if (dft_type_ == DFT_Type::kFFTW3) {
		runDft_fftw3();
	} else {
		runDft();
	}
	if (save_data_) {
		saveArrayData(result_folder_path_, "15_DFT", spectrum_);
	}

	//11. 整理 DFT 结果，获取 SFR 曲线
	calcSfrArray();
	if (save_data_) {
		saveArrayData(result_folder_path_, "16_SFR", freq_, sfr_);
	}

	//12. 计算指定空间频率下的 SFR 值
	double spatial_freq = 1.0 / sfr_option.spatial_freq_denominator;
	sfr_result = getSfrWithSpatialFreq(spatial_freq);

	return 0;
}

void SFR_Calculator::imageNormalization(){
	
	// TODO: 归一化
	if (!edge_normalized_.empty()){
		edge_normalized_.release();
	}
	edge_normalized_.create(edge_height_, edge_width_, CV_64FC1);

	for (int y = 0; y < edge_height_; y++) {
		for (int x = 0; x < edge_width_; x++) {
			edge_normalized_.at<double>(y, x) = (double)edge_img_.at<uchar>(y, x) / 255.0;
		}
	}

	return;
}

void SFR_Calculator::deGamma(double gamma) {

	// TODO: Gamma 矫正恢复

	for (int y = 0; y < edge_height_; y++) {
		for (int x = 0; x < edge_width_; x++) {
			edge_normalized_.at<double>(y, x) = pow(edge_normalized_.at<double>(y, x), 1.0 / gamma);
			if (save_img_){
				edge_img_.at<uchar>(y, x) = 255 * pow((double)edge_img_.at<uchar>(y, x) / 255.0f, 1.0 / gamma);
			}
		}
	}
	return;
}

int SFR_Calculator::checkEdgeTypeAndRotateRoi() {

	// TODO: 判断刀边方向，统一调整为垂直刀边（刀边方向不清晰时裁剪边缘）

	if (edge_width_ < 20 || edge_height_ < 20) {
		if (OutputLog)	OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "Edge Image Size too small");
		return -1;
	}

	std::vector<double> corner_value(4, 0);

	bool need_check = true;
	const int corner_area_size = 2;
	int clip_cnt = 0;

	while (need_check) {

		bool clip_lr = false, clip_tb = false;
		need_check = false;
		std::fill_n(corner_value.begin(), corner_value.size(), 0);

		for (int y = 0; y < corner_area_size; y++) {
			for (int x = 0; x < corner_area_size; x++) {
				corner_value[0] += (double)edge_img_.at<uchar>(y, x);
			}
		}
		for (int y = 0; y < corner_area_size; y++) {
			for (int x = edge_width_ - corner_area_size; x < edge_width_; x++) {
				corner_value[1] += (double)edge_img_.at<uchar>(y, x);
			}
		}
		for (int y = edge_height_ - corner_area_size; y < edge_height_; y++) {
			for (int x = 0; x < corner_area_size; x++) {
				corner_value[2] += (double)edge_img_.at<uchar>(y, x);
			}
		}
		for (int y = edge_height_ - corner_area_size; y < edge_height_; y++) {
			for (int x = edge_width_ - corner_area_size; x < edge_width_; x++) {
				corner_value[3] += (double)edge_img_.at<uchar>(y, x);
			}
		}

		double hor_diff = fabs(corner_value[0] + corner_value[2] - corner_value[1] - corner_value[3]) / 2.0f / (double)(corner_area_size * corner_area_size);
		double ver_diff = fabs(corner_value[0] + corner_value[1] - corner_value[2] - corner_value[3]) / 2.0f / (double)(corner_area_size * corner_area_size);

		if (edge_type_ == EdgeType::kNotDefinded) {
			if (ver_diff > 2.0 * hor_diff) {
				edge_type_ = EdgeType::kHorizontal;
			} else if (hor_diff > 2.0 * ver_diff) {
				edge_type_ = EdgeType::kVertical;
			} else {
				if (OutputLog) OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "Edge is not clear!");
				clip_lr = true;
				clip_tb = true;
			}
		}

		if (edge_type_ == EdgeType::kHorizontal) {
			if (hor_diff > 20.0) {
				if (OutputLog) OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "Horizontal Edge is not clear! left & right corner brightness diff > 20");
				clip_lr = true;
			}
		} else if (edge_type_ == EdgeType::kVertical) {
			if (ver_diff > 20.0) {
				if (OutputLog) OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "Vertical Edge is not clear! up & down corner brightness diff > 20");
				clip_tb = true;
			}
		}

		if (clip_lr || clip_tb) {
			if (edge_width_ < 20 + 4 || edge_height_ < 20 + 4) {
				if (OutputLog) OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "no more clip");
				return -1;
			} else {
				if (OutputLog) OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "clipping..., clip count:" + std::to_string(clip_cnt + 1));
				if (clip_lr) {
					edge_img_ = edge_img_(cv::Rect(corner_area_size, 0, edge_width_ - corner_area_size * 2, edge_height_)).clone();
					edge_width_ -= 4;
					if (use_user_defined_edge_position_ && edge_type_ == EdgeType::kHorizontal) {
						std::vector<double> temp_vec(calc_edge_pos_.begin() + corner_area_size, calc_edge_pos_.end() - 1 - corner_area_size);
						calc_edge_pos_ = temp_vec;
					}
				}
				if (clip_tb) {
					edge_img_ = edge_img_(cv::Rect(0, corner_area_size, edge_width_, edge_height_ - corner_area_size * 2)).clone();
					edge_height_ -= 4;
					if (use_user_defined_edge_position_ && edge_type_ == EdgeType::kHorizontal) {
						std::vector<double> temp_vec(calc_edge_pos_.begin() + corner_area_size, calc_edge_pos_.end() - 1 - corner_area_size);
						calc_edge_pos_ = temp_vec;
					}
				}
				clip_cnt++;
				need_check = true;
			}
		}
	}

	if (edge_width_ < 20 || edge_height_ < 20) {
		if (OutputLog) OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "Edge Image Size after clipping too small");
		return -1;
	}
	
	if (edge_type_ == EdgeType::kHorizontal) {
		cv::Mat temp_src = edge_img_.clone();
		edge_img_.release();
		std::swap(edge_width_, edge_height_);
		edge_img_.create(edge_height_, edge_width_, CV_8UC1);
		for (int y = 0; y < edge_height_; y++) {
			for (int x = 0; x < edge_width_; x++) {
				//edge_img_.at<uchar>(y, x) = temp_src.at<uchar>(edge_width_ - 1 - x, y);	//90度旋转
				edge_img_.at<uchar>(y, x) = temp_src.at<uchar>(x, y);
			}
		}
	}

	double left_sum = 0, right_sum = 0;
	int sum_cnt = 0;

	for (int y = 0; y < edge_height_; y++) {
		left_sum += (double)edge_img_.at<uchar>(y, 0);
		right_sum += (double)edge_img_.at<uchar>(y, edge_width_ - 1);
		sum_cnt++;
	}
	left_sum /= (double)sum_cnt;
	right_sum /= (double)sum_cnt;

	if (left_sum < right_sum) {
		edge_brightness_direction_ = EdgeBrightnessDirection::kBlackToWhite;
	} else if (right_sum < left_sum) {
		edge_brightness_direction_ = EdgeBrightnessDirection::kWhiteToBlack;
	} else {
		if (OutputLog) OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "Edge Direction is not clear!");
		return -1;
	}

	return 0;
}

int SFR_Calculator::calcEdgeLine() {

	// TODO: 计算 edge 线性


	//1. 计算每行刀边矩心
	if (calcRowCentroids(true, true)) {
		if (save_img_) {
			cv::Mat centroids_img = cv::Mat::zeros(edge_img_.size(), CV_8UC3);
			cv::cvtColor(edge_img_, centroids_img, CV_GRAY2RGB);
			drawCentroidsOnImg(centroids_img, row_centroids_);
			SaveRGBImg(result_folder_path_, "7_Img_CalcCentroids_NG", centroids_img, centroids_img.cols, centroids_img.rows);
		}
		return -1;
	}

	if (save_img_) {
		cv::Mat centroids_img = cv::Mat::zeros(edge_img_.size(), CV_8UC3);
		cv::cvtColor(edge_img_, centroids_img, CV_GRAY2RGB);
		drawCentroidsOnImg(centroids_img, row_centroids_);
		SaveRGBImg(result_folder_path_, "7_Img_CalcCentroids", centroids_img, centroids_img.cols, centroids_img.rows);
	}


	//2. 线性拟合
	
	//默认整个 edge 是线性的，直接拟合
	split_ = false;
	slope_ = 0, intercept_ = 0, err_ = 0;
	if (fitLineLeastSquares(row_centroids_, slope_, intercept_)) {
		return -1;
	}
	if (calcFitErr(row_centroids_, slope_, intercept_, err_)) {
		return -1;
	}

	int temp_edge_height = edge_height_;
	if (checkFitErr(err_, GLOBAL_LINEAR_FIT_RMSE_THRESHOLD) || checkSlope(slope_) || checkHeightAndReduceHeight(slope_, &temp_edge_height)) {
		if (save_img_) {
			cv::Mat fit_line_img = cv::Mat::zeros(edge_img_.size(), CV_8UC3);
			cv::cvtColor(edge_img_, fit_line_img, CV_GRAY2RGB);
			drawFitLineOnImg(fit_line_img, slope_, intercept_);
			SaveRGBImg(result_folder_path_, "8_Img_FitLine_NG", fit_line_img, fit_line_img.cols, fit_line_img.rows);
		}
		if (save_data_) {
			saveFitLineData(result_folder_path_, "8_FitLine_NG", slope_, intercept_, err_);
		}

		if (calcRowCentroids(true, false)) {
			if (save_img_) {
				cv::Mat centroids_img = cv::Mat::zeros(edge_img_.size(), CV_8UC3);
				cv::cvtColor(edge_img_, centroids_img, CV_GRAY2RGB);
				drawCentroidsOnImg(centroids_img, row_centroids_);
				SaveRGBImg(result_folder_path_, "9_Img_CalcCentroids_ForSplit_NG", centroids_img, centroids_img.cols, centroids_img.rows);
			}
			return -1;
		}
		if (save_img_) {
			cv::Mat centroids_img = cv::Mat::zeros(edge_img_.size(), CV_8UC3);
			cv::cvtColor(edge_img_, centroids_img, CV_GRAY2RGB);
			drawCentroidsOnImg(centroids_img, row_centroids_);
			SaveRGBImg(result_folder_path_, "9_Img_CalcCentroids_ForSplit", centroids_img, centroids_img.cols, centroids_img.rows);
		}

		//分割为多个线性区域
		if (OutputLog) OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "splitting linear area...");
		if (splitLinearAreaDP_CheckLinearAreaHeightLeastOnce()) {
			if (OutputLog) OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "failed to split linear area");
			return -1;
		}
		split_ = true;
	}
	else{
		edge_height_ = temp_edge_height;
	}

	if (save_img_) {
		cv::Mat fit_line_img = cv::Mat::zeros(edge_img_.size(), CV_8UC3);
		cv::cvtColor(edge_img_, fit_line_img, CV_GRAY2RGB);
		if (split_) {
			drawFitLineOnImg(fit_line_img, split_slope_, split_intercept_, split_is_linear_area_);
			SaveRGBImg(result_folder_path_, "10_Img_FitLine_Split", fit_line_img, fit_line_img.cols, fit_line_img.rows);
		} else {
			drawFitLineOnImg(fit_line_img, slope_, intercept_);
			SaveRGBImg(result_folder_path_, "10_Img_FitLine", fit_line_img, fit_line_img.cols, fit_line_img.rows);
		}
	}
	if (save_data_) {
		if (split_)
			saveFitLineData(result_folder_path_, "10_FitLine_Split", split_slope_, split_intercept_, split_err_, split_is_linear_area_);
		else
			saveFitLineData(result_folder_path_, "10_FitLine", slope_, intercept_, err_);
	}

	return 0;
}

int SFR_Calculator::calcRowCentroids(bool use_hamming, bool is_linear) {

	// TODO: 计算矩心

	row_centroids_.clear();
	row_centroids_.resize(edge_height_);

	cv::Mat temp_normalized = edge_normalized_.clone();

	double derivative_filter_left = 0, derivative_filter_right = 0;

	if (edge_brightness_direction_ == EdgeBrightnessDirection::kBlackToWhite) {
		derivative_filter_left = -1.0;
		derivative_filter_right = 1.0;
	}
	else {
		derivative_filter_left = 1.0;
		derivative_filter_right = -1.0;
	}

	findMaxDiffInRow();

	if (use_hamming) {
		double slope_first, intercept_first;
		bool fit_valid = true;
		double line_center = (double)(edge_width_ - 1) / 2.0;

		if (is_linear == true) {
			std::vector<double> row_max_diff_pos_dbl(edge_height_, 0);
			for (int y = 0; y < edge_height_; y++) {
				row_max_diff_pos_dbl[y] = row_max_diff_pos_[y];
			}
			fit_valid = true;
			if (fitLineLeastSquares(row_max_diff_pos_dbl, slope_first, intercept_first, 0, false)) {
				fit_valid = false;
			}

			std::vector<double> temp_row_centroids(edge_height_, 0);
			for (int y = 0; y < edge_height_; y++) {
				std::vector<double> temp_line_diff(edge_width_ - 1);
				for (int x = std::max(row_max_diff_pos_[y] - 10, 0); x < std::min(row_max_diff_pos_[y] + 10, edge_width_ - 1); x++) {
					temp_line_diff[x] = temp_normalized.at<double>(y, x) * derivative_filter_left + temp_normalized.at<double>(y, x + 1) * derivative_filter_right;
				}

				if (fit_valid == true) {
					double cur_centroid = slope_first * (double)y + intercept_first;
					double cen_offset = cur_centroid - line_center;
					hammingWindow(temp_line_diff, cen_offset);
				}
				else {
					double cen_offset = row_max_diff_pos_[y] - line_center;
					hammingWindow(temp_line_diff, cen_offset);
				}

				double dt = 0, dt1 = 0;
				for (int x = std::max(row_max_diff_pos_[y] - 10, 0); x < std::min(row_max_diff_pos_[y] + 10, edge_width_ - 1); x++) {
					dt1 += temp_line_diff[x];
					dt += temp_line_diff[x] * (double)x;
				}
				temp_row_centroids[y] = dt / dt1;
			}

			fit_valid = true;
			if (fitLineLeastSquares(temp_row_centroids, slope_first, intercept_first, 0, false)) {
				fit_valid = false;
			}
		}

		for (int y = 0; y < edge_height_; y++) {
			std::vector<double> temp_line_diff(edge_width_ - 1);
			for (int x = std::max(row_max_diff_pos_[y] - 10, 0); x < std::min(row_max_diff_pos_[y] + 10, edge_width_ - 1); x++) {
				temp_line_diff[x] = temp_normalized.at<double>(y, x) * derivative_filter_left + temp_normalized.at<double>(y, x + 1) * derivative_filter_right;
			}

			if (is_linear == true && fit_valid == true) {
				double cur_centroid = slope_first * (double)y + intercept_first;
				double cen_offset = cur_centroid - line_center;
				hammingWindow(temp_line_diff, cen_offset);
			}
			else {
				double cen_offset = row_max_diff_pos_[y] - line_center;
				hammingWindow(temp_line_diff, cen_offset);
			}

			double dt = 0, dt1 = 0;
			for (int x = std::max(row_max_diff_pos_[y] - 10, 0); x < std::min(row_max_diff_pos_[y] + 10, edge_width_ - 1); x++) {
				dt1 += temp_line_diff[x];
				dt += temp_line_diff[x] * (double)x;
			}
			row_centroids_[y] = dt / dt1;
		}
	}
	else {
		for (int y = 0; y < edge_height_; y++) {
			double dt = 0, dt1 = 0;
			for (int x = std::max(row_max_diff_pos_[y] - 10, 0); x < std::min(row_max_diff_pos_[y] + 10, edge_width_ - 1); x++) {
				double temp = temp_normalized.at<double>(y, x) * derivative_filter_left + temp_normalized.at<double>(y, x + 1) * derivative_filter_right;
				dt1 += temp;
				dt += temp * (double)x;
			}
			row_centroids_[y] = dt / dt1;
		}
	}

	int close_pixel_cnt = 0;
	for (int y = 0; y < edge_height_; y++) {
		if (row_centroids_[y] < 10.0 || edge_width_ - 1 - row_centroids_[y] < 10.0) {
			close_pixel_cnt++;
		}
	}
	if ((double)close_pixel_cnt / (double)edge_height_ > 0.2) {
		if (OutputLog) OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "The edge in ROI is too close to the image corners");
		return -1;
	}

	return 0;
}

void SFR_Calculator::findMaxDiffInRow() {

	// TODO: 找到每行梯度变化最大的点

	row_max_diff_pos_.clear();
	row_max_diff_pos_.resize(edge_height_);

	double max_diff = 0;
	int max_diff_pos = 0;

	for (int y = 0; y < edge_height_; y++) {
		max_diff = 0;
		max_diff_pos = 0;
		for (int x = 2; x < edge_width_ - 2; x++) {
			double cur_diff = abs(
				1 * edge_normalized_.at<double>(y, x + 2)
				+ 2 * edge_normalized_.at<double>(y, x + 1)
				- 2 * edge_normalized_.at<double>(y, x - 1)
				- 1 * edge_normalized_.at<double>(y, x - 2));
			if (max_diff < cur_diff) {
				max_diff = cur_diff;
				max_diff_pos = x;
			}
		}
		row_max_diff_pos_[y] = max_diff_pos;
	}

	return;
}

int SFR_Calculator::fitLineLeastSquares(std::vector<double> data, double& slope, double& intercrpt, int idx_offset, bool output_log) {

	// TODO: 线性拟合

	double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;

	int n = data.size();
	if (n < 2) {
		if (output_log && OutputLog) {
			OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") +
				"The data used for fitting is empty");
		}
		slope = 0.0;
		intercrpt = 0.0;
		return -1;
	}

	for (int i = 0; i < n; i++) {
		double x = i + idx_offset;
		sum_x += (double)x;
		sum_y += data[i];
		sum_xx += ((double)x * (double)x);
		sum_xy += data[i] * (double)x;
	}

	intercrpt = (sum_xx * sum_y - sum_xy * sum_x) / ((double)n * sum_xx - sum_x * sum_x);
	slope = ((double)n * sum_xy - sum_y * sum_x) / ((double)n * sum_xx - sum_x * sum_x);

	if (isnan(slope) || isnan(intercrpt)) {
		if (output_log && OutputLog) {
			OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") +
				"Invalid fitting result");
		}
		slope = 0.0;
		intercrpt = 0.0;
		return -1;
	}

	return 0;
}

int SFR_Calculator::calcFitErr(std::vector<double> data, double slope, double intercept, double& rmse, int idx_offset, bool output_log) {

	// TODO: 线性拟合误差

	int n = data.size();
	if (n < 2) {
		if (output_log && OutputLog) {
			OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") +
				"The data used for calc fit error is empty");
		}
		rmse = 1.0;
		return -1;
	}

	double sse = 0;
	for (int i = 0; i < n; i++) {
		double x = i + idx_offset;
		double y = data[i];
		double y_expect = slope * x + intercept;
		sse += (y - y_expect) * (y - y_expect);
	}
	rmse = sqrt(sse / (double)n);

	if (isnan(rmse)) {
		if (output_log && OutputLog) {
			OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") +
				"Invalid fit error result");
		}
		rmse = 1.0;
		return -1;
	}

	return 0;
}

int SFR_Calculator::splitLinearAreaDP_CheckLinearAreaHeightLeastOnce() {

	// TODO: 把整个 edge 区域分割，获取多个满足线性要求的区域

	split_slope_.clear();
	split_slope_.resize(edge_height_);
	split_intercept_.clear();
	split_intercept_.resize(edge_height_);
	split_err_.clear();
	split_err_.resize(edge_height_);
	split_is_linear_area_.clear();
	split_is_linear_area_.resize(edge_height_);

	bool found = false;

	const int min_height = SPLIT_LINEAR_MIN_AREA_HEIGHT;

	std::vector<std::vector<int>> dp(edge_height_, std::vector<int>(2, 0));
	std::vector<std::vector<std::vector<std::pair<int, int>>>> area_start_and_height(edge_height_, std::vector<std::vector<std::pair<int, int>>>(2));
	std::vector<std::vector<std::vector<std::tuple<double, double, double>>>> slope_intercept_err(edge_height_, std::vector<std::vector<std::tuple<double, double, double>>>(2));

	for (int end_row = 1; end_row < edge_height_; end_row++) {
		dp[end_row] = dp[end_row - 1];
		area_start_and_height[end_row] = area_start_and_height[end_row - 1];
		slope_intercept_err[end_row] = slope_intercept_err[end_row - 1];

		for (int start_row = 0; start_row < end_row - min_height + 1; start_row++) {
			int temp_height = end_row - start_row + 1;
			std::vector<double> temp_centroids(row_centroids_.begin() + start_row, row_centroids_.begin() + start_row + temp_height);
			double temp_slope = 0, temp_intercept = 0, temp_err = 0;
			if (fitLineLeastSquares(temp_centroids, temp_slope, temp_intercept, start_row, false)) {
				continue;
			}
			if (calcFitErr(temp_centroids, temp_slope, temp_intercept, temp_err, start_row, false)) {
				continue;
			}

			if (checkFitErr(temp_err, SPLIT_LINEAR_FIT_RMSE_THRESHOLD, false) == 0 && checkSlope(temp_slope, false) == 0){

				if (dp[end_row][0] < (start_row > 0 ? (dp[start_row - 1][0] + temp_height) : temp_height)) {
					dp[end_row][0] = start_row > 0 ? (dp[start_row - 1][0] + temp_height) : temp_height;
					if (start_row > 0) {
						area_start_and_height[end_row][0] = area_start_and_height[start_row - 1][0];
						area_start_and_height[end_row][0].push_back(std::make_pair(start_row, temp_height));
						slope_intercept_err[end_row][0] = slope_intercept_err[start_row - 1][0];
						slope_intercept_err[end_row][0].push_back(std::make_tuple(temp_slope, temp_intercept, temp_err));
					} else {
						area_start_and_height[end_row][0] = std::vector<std::pair<int, int>>(1, std::make_pair(start_row, temp_height));
						slope_intercept_err[end_row][0] = std::vector<std::tuple<double, double, double>>(1, std::make_tuple(temp_slope, temp_intercept, temp_err));
					}
				}

				if (dp[end_row][1] < (start_row > 0 ? (dp[start_row - 1][1] + temp_height) : temp_height)) {
					dp[end_row][1] = start_row > 0 ? (dp[start_row - 1][1] + temp_height) : temp_height;
					if (start_row > 0) {
						area_start_and_height[end_row][1] = area_start_and_height[start_row - 1][1];
						area_start_and_height[end_row][1].push_back(std::make_pair(start_row, temp_height));
						slope_intercept_err[end_row][1] = slope_intercept_err[start_row - 1][1];
						slope_intercept_err[end_row][1].push_back(std::make_tuple(temp_slope, temp_intercept, temp_err));
					} else {
						area_start_and_height[end_row][1] = std::vector<std::pair<int, int>>(1, std::make_pair(start_row, temp_height));
						slope_intercept_err[end_row][1] = std::vector<std::tuple<double, double, double>>(1, std::make_tuple(temp_slope, temp_intercept, temp_err));
					}
				}

				int reduce_height = 0;
				if (checkHeightAndReduceHeight(temp_slope, &temp_height, false, &reduce_height) == 0) {
					int new_start_row = start_row + reduce_height;
					if (dp[end_row][1] < (new_start_row > 0 ? (dp[new_start_row - 1][0] + temp_height) : temp_height)) {
						dp[end_row][1] = new_start_row > 0 ? (dp[new_start_row - 1][0] + temp_height) : temp_height;
						if (new_start_row > 0) {
							area_start_and_height[end_row][1] = area_start_and_height[new_start_row - 1][0];
							area_start_and_height[end_row][1].push_back(std::make_pair(new_start_row, temp_height));
							slope_intercept_err[end_row][1] = slope_intercept_err[new_start_row - 1][0];
							slope_intercept_err[end_row][1].push_back(std::make_tuple(temp_slope, temp_intercept, temp_err));
						} else {
							area_start_and_height[end_row][1] = std::vector<std::pair<int, int>>(1, std::make_pair(new_start_row, temp_height));
							slope_intercept_err[end_row][1] = std::vector<std::tuple<double, double, double>>(1, std::make_tuple(temp_slope, temp_intercept, temp_err));
						}
					}
					if (dp[end_row][1] < (new_start_row > 0 ? (dp[new_start_row - 1][1] + temp_height) : temp_height)) {
						dp[end_row][1] = new_start_row > 0 ? (dp[new_start_row - 1][1] + temp_height) : temp_height;
						if (new_start_row > 0) {
							area_start_and_height[end_row][1] = area_start_and_height[new_start_row - 1][1];
							area_start_and_height[end_row][1].push_back(std::make_pair(new_start_row, temp_height));
							slope_intercept_err[end_row][1] = slope_intercept_err[new_start_row - 1][1];
							slope_intercept_err[end_row][1].push_back(std::make_tuple(temp_slope, temp_intercept, temp_err));
						} else {
							area_start_and_height[end_row][1] = std::vector<std::pair<int, int>>(1, std::make_pair(new_start_row, temp_height));
							slope_intercept_err[end_row][1] = std::vector<std::tuple<double, double, double>>(1, std::make_tuple(temp_slope, temp_intercept, temp_err));
						}
					}
				}
			}
		}
	}

	if (area_start_and_height[edge_height_ - 1][1].size() != 0)
		found = true;

	if (found == false)
		return -1;

	std::vector<std::pair<int, int>> best_area_start_and_height = area_start_and_height[edge_height_ - 1][1];
	std::vector<std::tuple<double, double, double>> best_slope_intercept_err = slope_intercept_err[edge_height_ - 1][1];

	for (int i = 0; i < best_area_start_and_height.size(); i++) {
		int cur_start = best_area_start_and_height[i].first;
		int cur_height = best_area_start_and_height[i].second;
		double cur_slope = std::get<0>(best_slope_intercept_err[i]);
		double cur_intercept = std::get<1>(best_slope_intercept_err[i]);
		double cur_err = std::get<2>(best_slope_intercept_err[i]);
		std::fill_n(split_slope_.begin() + cur_start, cur_height, cur_slope);
		std::fill_n(split_intercept_.begin() + cur_start, cur_height, cur_intercept);
		std::fill_n(split_err_.begin() + cur_start, cur_height, cur_err);
		std::fill_n(split_is_linear_area_.begin() + cur_start, cur_height, true);
	}

	return 0;
}

int SFR_Calculator::checkSlope(double edge_slope, bool output_log) {

	// TODO: 根据iso要求判断斜率

	if (!isnormal(edge_slope)) {
		if (output_log && OutputLog) {
			OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") +
				"Invalid edge slope");
		}
		return -1;
	}

	//1.边缘斜率不能高于1/4（保证4倍超采样不会跳档）
	if (fabs(edge_slope) > 0.25) {
		if (output_log && OutputLog) {
			OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") +
				"Slope of the edge must be less than 1/4, edge tilt angle should be less than 14 degrees");
		}
		return -1;
	}

	return 0;
}

int SFR_Calculator::checkHeightAndReduceHeight(double edge_slope, int* edge_height, bool output_log, int* reduce_height) {

	// TODO: 根据iso要求判断高度并调整roi高度

	if (*edge_height <= 0 || !isnormal(edge_slope)) {
		if (output_log && OutputLog) {
			OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") +
				"Invalid edge slope & edge height");
		}
		return -1;
	}

	//1.计算数据高度应调整为1/slope整数倍（保证4倍超采样各档样本数均匀）
	int cycs = (int)((double)*edge_height * fabs(edge_slope));
	if (reduce_height)
		*reduce_height = 0;
	if (cycs / fabs(edge_slope) < *edge_height) {
		if (reduce_height)
			*reduce_height = *edge_height - cycs / fabs(edge_slope);
		*edge_height = cycs / fabs(edge_slope);
	}

	//2.边缘斜率不能低于1/图像高度h（保证4倍超采样周期完整，不会缺失档位）（h不能为0）
	if (*edge_height == 0 || fabs(edge_slope) < (1.0f / (double)*edge_height)) {
		if (output_log && OutputLog) {
			OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") +
				"Slope of the edge must be larger than 1/RoiHeight, need to rotate the chart or increase the edge image size");
		}
		return -1;
	}

	return 0;
}

int SFR_Calculator::checkFitErr(double err, double err_boundary, bool output_log) {

	// TODO: 判断线性拟合误差

	if (!isnormal(err) || !isnormal(err_boundary)) {
		if (output_log && OutputLog) {
			OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") +
				"Invalid error & error boundary");
		}
		return -1;
	}

	if (err > err_boundary) {
		if (output_log && OutputLog) {
			OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") +
				"Fit Line RMSE error is out of boundary, edge shape may not be a line");
		}
		return -1;
	}

	return 0;
}

void SFR_Calculator::calcEdgePostion() {

	// TODO: 计算 edge 在每行的确切位置

	if (use_user_defined_edge_position_ == false) {
		calc_edge_pos_.clear();
		calc_edge_pos_.resize(edge_height_);
	}

	double min_edge_pos = edge_width_;
	double max_edge_pos = 0;
	double edge_pos_sum = 0.0;
	int sum_cnt = 0;

	for (int y = 0; y < edge_height_; y++) {
		if (use_user_defined_edge_position_ == false) {
			if (split_) {
				if (split_is_linear_area_[y]) {
					calc_edge_pos_[y] = split_slope_[y] * (double)y + split_intercept_[y];
				} else {
					continue;
				}
			} else {
				calc_edge_pos_[y] = slope_ * (double)y + intercept_;
			}
		}
		min_edge_pos = std::min(min_edge_pos, calc_edge_pos_[y]);
		max_edge_pos = std::max(max_edge_pos, calc_edge_pos_[y]);
		edge_pos_sum += calc_edge_pos_[y];
		sum_cnt++;
	}

	far_left_calc_edge_pos_ = min_edge_pos;
	far_right_calc_edge_pos_ = max_edge_pos;
	avg_calc_edge_pos_ = edge_pos_sum / (double)sum_cnt;

	return;
}

int SFR_Calculator::overSampling4x() {

	// TODO: 四倍超采样

	esf_.clear();
	esf_.resize(edge_width_ * 4);
	std::fill_n(esf_.begin(), esf_.size(), 0);
	esf_add_cnt_.clear();
	esf_add_cnt_.resize(edge_width_ * 4);
	std::fill_n(esf_add_cnt_.begin(), esf_add_cnt_.size(), 0);

	int num = 0;
	double cur_edge_pos = 0;
	double base_edge_pos = 0;

	base_edge_pos = avg_calc_edge_pos_;

	for (int y = 0; y < edge_height_; y++) {
		if (split_) {
			if (split_is_linear_area_[y] == false) {
				continue;
			}
		}
		cur_edge_pos = calc_edge_pos_[y];
		for (int x = 0; x < edge_width_; x++) {
			num = (int)((((double)x + base_edge_pos - cur_edge_pos) - 0.125f) / 0.25f + 0.5f);
			if (num >= 0 && num < 4 * edge_width_) {
				esf_[num] += edge_normalized_.at<double>(y, x);
				esf_add_cnt_[num]++;
			}
		}
	}

	std::vector<int> index_empty;
	for (int index = 0; index < esf_.size(); index++) {
		if (esf_add_cnt_[index]) {
			esf_[index] /= (double)esf_add_cnt_[index];
		}
		else {
			index_empty.push_back(index);
		}
	}

	//填充空白数据（空白数据会使得 ESF 存在突变区域，SFR 结果变高）
	for (int i = 0; i < index_empty.size(); i++) {
		int index = index_empty[i];
		int neigh_left = index - 1, neigh_right = index + 1;
		bool found_left = false, found_right = false;
		if (index == 0) {
			while (find(index_empty.begin(), index_empty.end(), neigh_right) != index_empty.end()) {
				neigh_right++;
			}

			if (neigh_right > esf_.size() - 1) {
				if (OutputLog) {
					OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "Empty ESF Array");
				}
				return -1;
			}
			else {
				found_right = true;
			}
		}
		else if (index == esf_.size() - 1) {
			while (find(index_empty.begin(), index_empty.end(), neigh_left) != index_empty.end()) {
				neigh_left--;
			}

			if (neigh_left < 0) {
				if (OutputLog) {
					OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "Empty ESF Array");
				}
				return -1;
			}
			else {
				found_left = true;
			}
		}
		else {
			while (find(index_empty.begin(), index_empty.end(), neigh_left) != index_empty.end()) {
				neigh_left--;
			}
			while (find(index_empty.begin(), index_empty.end(), neigh_right) != index_empty.end()) {
				neigh_right++;
			}

			if (neigh_left < 0 && neigh_right > esf_.size() - 1) {
				if (OutputLog) {
					OutputLog(test_process_prefix_ + (test_process_prefix_ == "" ? "" : ": ") + "Empty ESF Array");
				}
				return -1;
			}
			else if (neigh_left < 0) {
				found_right = true;
			}
			else if (neigh_right > esf_.size() - 1) {
				found_left = true;
			}
			else {
				found_left = true;
				found_right = true;
			}
		}

		if (found_left == true && found_right == true) {
			esf_[index] = (esf_[neigh_right] - esf_[neigh_left]) / (double)(neigh_right - neigh_left) * (double)(index - neigh_left) + esf_[neigh_left];
		}
		else if (found_left == true) {
			esf_[index] = esf_[neigh_left];
		}
		else if (found_right == true) {
			esf_[index] = esf_[neigh_right];
		}
	}

	return 0;
}

void SFR_Calculator::calcLsf() {

	// TODO: 计算LSF

	lsf_.clear();
	lsf_.resize(esf_.size());
	std::fill_n(lsf_.begin(), lsf_.size(), 0);

	double derivative_filter_left = 0, derivative_filter_right = 0;

#if LSF_DERIVATIVE_FILTER == 1
	if (edge_brightness_direction_ == EdgeBrightnessDirection::kBlackToWhite) {
		derivative_filter_left = -0.5;
		derivative_filter_right = 0.5;
	}
	else {
		derivative_filter_left = 0.5;
		derivative_filter_right = -0.5;
	}
#else
	if (edge_brightness_direction_ == EdgeBrightnessDirection::kBlackToWhite) {
		derivative_filter_left = -1.0;
		derivative_filter_right = 1.0;
	}
	else {
		derivative_filter_left = 1.0;
		derivative_filter_right = -1.0;
	}
#endif

	double pre_value, next_value;
	for (int i = 0; i < lsf_.size(); i++) {
		if (i == 0) {
			pre_value = esf_[0];
		}
		else {
			pre_value = esf_[i - 1];
		}

#if LSF_DERIVATIVE_FILTER == 1
		if (i == lsf_.size() - 1) {
			next_value = esf_[lsf_.size() - 1];
		}
		else {
			next_value = esf_[i + 1];
		}
#else
		next_value = esf_[i];
#endif

		lsf_[i] = pre_value * derivative_filter_left + next_value * derivative_filter_right;
	}

	//两侧的值用临近值填充
	lsf_[0] = lsf_[1];
#if LSF_DERIVATIVE_FILTER == 1
	lsf_[lsf_.size() - 1] = lsf_[lsf_.size() - 2];
#endif

	return;
}

void SFR_Calculator::centeringLsf(bool calc_lsf_centroid_method) {

	// TODO: 调整LSF峰值位置至中心（计算曲线矩心 或 根据峰值位置 移动曲线）

	int offset = 0;

	if (calc_lsf_centroid_method) {	//计算曲线矩心
		double peak_num = -1;
		double dt = 0, dt1 = 0;
		for (int i = 0; i < lsf_.size(); i++) {
			double temp = lsf_[i];
			dt1 += temp;
			dt += temp * (double)i;
		}
		peak_num = dt / dt1;

		offset = (int)(peak_num + 0.5) - lsf_.size() / 2;
	}
	else {	//根据峰值位置
		//此处考虑可能存在双峰情况
		int left_peak_num = -1, right_peak_num = -1;
		double max_lsf = -1;
		for (int i = 0; i < lsf_.size(); i++) {
			if (fabs(lsf_[i]) == max_lsf) {
				right_peak_num = i;
			}
			if (fabs(lsf_[i]) > max_lsf) {
				max_lsf = fabs(lsf_[i]);
				left_peak_num = i;
				right_peak_num = i;
			}
		}
		offset = (left_peak_num + right_peak_num) / 2 - lsf_.size() / 2;
	}

	std::vector<double> lsf_array_before(lsf_);

	for (int i = 0; i < lsf_.size(); i++) {
		double temp;
		if ((i + offset < 0) || (i + offset > lsf_.size() - 1)) {
			temp = 0;
		}
		else {
			temp = lsf_array_before[i + offset];
		}
		lsf_[i] = temp;
	}

	return;
}

void SFR_Calculator::hammingWindow(std::vector<double>& data, double cen_offset) {

	// TODO: 汉明窗

	double n = data.size();
	double denominator = n + 2 * fabs(cen_offset);
	double numerator;
	if (cen_offset >= 0) {
		numerator = 0;
	}
	else {
		numerator = -2 * cen_offset;
	}

	for (int i = 0; i < n; i++) {
		data[i] *= 0.54 - 0.46 * cos(2 * CV_PI * (double)numerator / (double)(denominator - 1));
		numerator += 1.0;
	}

	return;
}

void SFR_Calculator::runDft() {

	// TODO: DFT

	int size = lsf_.size();
	spectrum_.clear();
	spectrum_.resize(size);

	for (int k = 0; k < size; k++) {
		std::complex<double> sum(0, 0);
		for (int n = 0; n < size; n++) {
			sum += lsf_[n] * exp(-2.0 * CV_PI * std::complex<double>(0, 1) * (double)k * (double)n / (double)size);
		}
		spectrum_[k] = sum;
	}

	return;
}

void SFR_Calculator::runDft_fftw3() {

	// TODO: DFT (use fftw3)

	int size = lsf_.size();
	spectrum_.clear();
	spectrum_.resize(size);

	fftw_plan p_fftw3;
	double* fftw3_in_r;
	fftw_complex* fftw3_out_c;

	fftw3_in_r = (double*)fftw_malloc(sizeof(double) * size);
	fftw3_out_c = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * size);
	p_fftw3 = fftw_plan_dft_r2c_1d(size, fftw3_in_r, fftw3_out_c, FFTW_ESTIMATE);

	for (int n = 0; n < size; n++) {
		fftw3_in_r[n] = lsf_[n];
	}

	fftw_execute(p_fftw3);

	double* p_complex = (double*)fftw3_out_c;

	for (int k = 0; k < size; k++) {
		spectrum_[k] = std::complex<double>(p_complex[2 * k], p_complex[2 * k + 1]);
	}

	fftw_destroy_plan(p_fftw3);

	if (fftw3_in_r != NULL) {
		fftw_free(fftw3_in_r);
		fftw3_in_r = NULL;
	}
	if (fftw3_out_c != NULL) {
		fftw_free(fftw3_out_c);
		fftw3_out_c = NULL;
	}

	return;
}

void SFR_Calculator::calcSfrArray() {

	// TODO: 整理 DFT 结果，获取 SFR 数组

	//int size = edge_width_;
	int size = spectrum_.size() / 4;
	sfr_.clear();
	sfr_.resize(size + 1);
	freq_.clear();
	freq_.resize(size + 1);

	double dc_value = abs(spectrum_[0]);

	for (int k = 0; k <= size; k++) {
		sfr_[k] = abs(spectrum_[k]) / dc_value;
		freq_[k] = (double)k / (double)size;

		// 校正离散差分对信号频域放大失真的影响
		if (k > 0) {
			double wk = 2 * CV_PI * (double)k / (double)(size * 4);
#if LSF_DERIVATIVE_FILTER == 1
			sfr_[k] = sfr_[k] / abs(sin(wk)) * wk;
#else
			sfr_[k] = sfr_[k] / abs(2 * sin(wk / 2)) * wk;
#endif
		}
	}

	return;
}

double SFR_Calculator::getSfrWithSpatialFreq(double spatial_freq) {

	// TODO: 获取指定空间频率下的 SFR
	double sfr_Area = 0.0;
	double freq_before = 0.0, freq_next = 1.0;
	double sfr_before = 0.0, sfr_next = 0.0;
	for (int k = 0; k < freq_.size(); k++) {
		if (freq_[k] <= spatial_freq * 0.5f) {
			freq_before = freq_[k];
			sfr_before = sfr_[k];
		} else {
			freq_next = freq_[k];
			sfr_next = sfr_[k];
			break;
		}
		if (freq_[k] < 0.5f)sfr_Area += sfr_[k];
	}
	double sfr_result = (sfr_before - sfr_next) / (freq_before - freq_next) * (spatial_freq * 0.5f - freq_next) + sfr_next;

	if (isnan(sfr_result)) {
		sfr_result = -1.0;
	}
	return sfr_Area;

	return sfr_result;
}

void SFR_Calculator::drawCentroidsOnImg(cv::Mat& img, std::vector<double> row_centroids, double centroid_offset) {

	int rows = std::min(img.rows, (int)row_centroids.size());

	for (int y = 0; y < rows; y++) {
		int centroid_x = (int)(row_centroids[y] + centroid_offset + 0.5f);
		centroid_x = std::max(centroid_x, 0);
		centroid_x = std::min(centroid_x, img.cols - 1);
		img.at<cv::Vec3b>(y, centroid_x)[0] = 0;
		img.at<cv::Vec3b>(y, centroid_x)[1] = 0;
		img.at<cv::Vec3b>(y, centroid_x)[2] = 255;
	}

	return;
}

void SFR_Calculator::drawFitLineOnImg(cv::Mat& img, double slope, double intercept) {

	for (int y = 0; y < img.rows; y++) {
		int fit_line_x = (int)(slope * (double)y + intercept + 0.5f);
		fit_line_x = std::max(fit_line_x, 0);
		fit_line_x = std::min(fit_line_x, img.cols - 1);
		img.at<cv::Vec3b>(y, fit_line_x)[0] = 0;
		img.at<cv::Vec3b>(y, fit_line_x)[1] = 0;
		img.at<cv::Vec3b>(y, fit_line_x)[2] = 255;
	}

	return;
}

void SFR_Calculator::drawFitLineOnImg(cv::Mat& img, std::vector<double> slope, std::vector<double> intercept, std::vector<bool> is_linear_area) {

	int rows = std::min(img.rows, (int)is_linear_area.size());

	for (int y = 0; y < rows; y++) {
		if (is_linear_area[y]) {
			int fit_line_x = (int)(slope[y] * (double)y + intercept[y] + 0.5f);
			fit_line_x = std::max(fit_line_x, 0);
			fit_line_x = std::min(fit_line_x, img.cols - 1);
			img.at<cv::Vec3b>(y, fit_line_x)[0] = 0;
			img.at<cv::Vec3b>(y, fit_line_x)[1] = 0;
			img.at<cv::Vec3b>(y, fit_line_x)[2] = 255;
		}
	}

	return;
}

void SFR_Calculator::saveFitLineData(std::string folder, std::string file_name, double slope, double intercept, double err) {

	std::fstream fout;
	std::string file_path = folder + "\\" + file_name + ".txt";
	fout.open(file_path, std::ios::out);
	fout << "Slope: " << slope << std::endl;
	fout << "Intercept: " << intercept << std::endl;
	fout << "Err: " << err << std::endl;
	fout.close();

	return;
}

void SFR_Calculator::saveFitLineData(std::string folder, std::string file_name, std::vector<double> slope, std::vector<double> intercept, std::vector<double> err, std::vector<bool> is_linear_area) {

	std::fstream fout;
	std::string file_path = folder + "\\" + file_name + ".txt";
	fout.open(file_path, std::ios::out);
	fout << "StartRow\tEndRow\tSlope\tIntercept\tErr" << std::endl;

	int start = 0;
	for (int y = 0; y < is_linear_area.size(); y++) {
		if (is_linear_area[y]) {
			if (y == 0) {
				start = y;
			} else {
				if (slope[y] != slope[y - 1] || intercept[y] != intercept[y - 1] || err[y] != err[y - 1] || is_linear_area[y - 1] == false) {
					start = y;
				}
			}
			if (y == is_linear_area.size() - 1) {
				fout << start << "\t" << y << "\t" << slope[y] << "\t" << intercept[y] << "\t" << err[y] << std::endl;
			} else {
				if (slope[y] != slope[y + 1] || intercept[y] != intercept[y + 1] || err[y] != err[y + 1] || is_linear_area[y + 1] == false) {
					fout << start << "\t" << y << "\t" << slope[y] << "\t" << intercept[y] << "\t" << err[y] << std::endl;
				}
			}
		}
	}
	fout.close();

	return;
}

void SFR_Calculator::saveArrayData(std::string folder, std::string file_name, std::vector<double> arr) {

	std::fstream fout;
	std::string file_path = folder + "\\" + file_name + ".txt";
	fout.open(file_path, std::ios::out);
	for (int i = 0; i < arr.size(); i++) {
		fout << arr[i] << std::endl;
	}
	fout.close();

	return;
}

void SFR_Calculator::saveArrayData(std::string folder, std::string file_name, std::vector<std::complex<double>> arr) {

	std::fstream fout;
	std::string file_path = folder + "\\" + file_name + ".txt";
	fout.open(file_path, std::ios::out);
	for (int i = 0; i < arr.size(); i++) {
		fout << arr[i].real() << "\t" << arr[i].imag() << std::endl;
	}
	fout.close();

	return;
}

void SFR_Calculator::saveArrayData(std::string folder, std::string file_name, std::vector<double> arr1, std::vector<double> arr2) {

	std::fstream fout;
	std::string file_path = folder + "\\" + file_name + ".txt";
	fout.open(file_path, std::ios::out);
	for (int i = 0; i < sfr_.size(); i++) {
		fout << freq_[i] << "\t" << sfr_[i] << std::endl;
	}
	fout.close();

	return;
}

void SFR_Calculator::SaveGrayImg(std::string folder, std::string img_name, cv::Mat gray_img, int width, int height) {

	cv::imwrite(folder + "\\" + img_name + ".bmp", gray_img);

	return;
}

void SFR_Calculator::SaveRGBImg(std::string folder, std::string img_name, cv::Mat rgb_img, int width, int height) {

	cv::imwrite(folder + "\\" + img_name + ".bmp", rgb_img);

	return;
}