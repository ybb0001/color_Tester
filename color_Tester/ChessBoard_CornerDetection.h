#pragma once
#include"ChessBoard_CamCalibration.h"

#define EDGE_SIDE_CHK_COLOR_OVERLAP_RATIO_THRES 0.005 //0.001
//#define SAVE_RAW

class ChessBoard_CornerDetection {
public:
	/** \brief 检测棋盘格角点参数结构体 */
	struct CornerDectOption {
		int chessDectType;	//棋盘格检测模式：0:简化搜索检测，1:全参数搜索检测，-1:非自动检测
		int startCountPos_X;	//角点行列计数起始位置（将最靠近该位置的角点设置为 行列位置原点）
		int startCountPos_Y;	//角点行列计数起始位置（将最靠近该位置的角点设置为 行列位置原点）
		int minCornerNum;	//角点个数下限，默认为 10
		int thresholdValue;	//非自动检测时，二值化阈值，设置为 -1 时 OTSU，默认为 -1
		int morphType;	//非自动检测时，形态学操作类型：0:膨胀，1:腐蚀，-1:不指定，默认为 0
		int morphIter;	//非自动检测时，腐蚀或膨胀 迭代次数，默认为 5
		int approxPolygonError;	//非自动检测时，多边形近似最大误差，默认为 10

		CornerDectOption() :chessDectType(0), startCountPos_X(0), startCountPos_Y(0), minCornerNum(10),
			thresholdValue(-1), morphType(0), morphIter(5), approxPolygonError(10) {}
	};

	/** @brief 检测棋盘格刀边参数结构体 */
	struct SfrEdgeDectOption {
		std::vector<std::pair<double, double>> dect_pos; // 点位位置（Vec2d[0]：field，范围0<=field<1；Vec2d[1]：degree，范围0<=degree<360，3点钟方向为0度，顺时针旋转）
		int roi_length;	// 截取刀边时的 Roi 长度，默认为 50
		int roi_width;	// 截取刀边时的 Roi 宽度，默认为 50

		SfrEdgeDectOption() :roi_length(50), roi_width(50) {}
	};

private:
	struct MyChessBlock {
		cv::Point2d centerPoint;
		cv::Point2d cornerPoint[4];	//LT,RT,RB,LB
		double edgeSlope_H[2];	//UP,DOWN
		double edgeSlope_V[2];	//LEFT,RIGHT
		double edgeLength_H[2];	//UP,DOWN
		double edgeLength_V[2];	//LEFT,RIGHT
		MyChessBlock* connection_out[4];	//LT,RT,RB,LB
		bool bConnected[4];	//LT,RT,RB,LB
		int inCount;
		int blockIndex;
		bool bIsValid;

		double angle_axisRotate;	//block坐标上施加的坐标系旋转角度

		MyChessBlock() {
			centerPoint = cv::Point2d(0, 0);
			for (int i = 0; i < 2; i++) {
				edgeSlope_H[i] = 0;
				edgeSlope_V[i] = 0;
				edgeLength_H[i] = 0;
				edgeLength_V[i] = 0;
			}
			for (int i = 0; i < 4; i++) {
				cornerPoint[i] = cv::Point2d(0, 0);
				connection_out[i] = NULL;
				bConnected[i] = false;
			}

			inCount = 0;
			blockIndex = -1;
			bIsValid = false;

			angle_axisRotate = 0.0;
		}
	};

public:
	ChessBoard_CornerDetection();
	virtual ~ChessBoard_CornerDetection();

private:
	enum EdgeType
	{
		EDGE_NOT_DEFINDED,
		EDGE_HORIZONTAL,
		EDGE_VERTICAL,
	};
	/** @brief 边缘检测错误类型 */
	enum class EdgeDetectError {
		kHorWhiteToBlackFailed = 0x0001,
		kHorBlackToWhiteFailed = 0x0010,
		kVerWhiteToBlackFailed = 0x0100,
		kVerBlackToWhiteFailed = 0x1000,
	};

	//解决 cv::findContours 崩溃的重写函数
	void findContours2(const cv::Mat image, std::vector<std::vector<cv::Point>>& contours, int mode, int method, cv::Point offset = cv::Point());
	void findContours2(const cv::Mat image, std::vector<cv::Mat>& contours, std::vector<cv::Vec4i>& hierarchy, int mode, int method, cv::Point offset = cv::Point());
	void findContours2(const cv::Mat image, std::vector<cv::Mat>& contours, int mode, int method, cv::Point offset = cv::Point());

	//通用
	double GetDistanceOfTwoPoint(double pt1_x, double pt1_y, double pt2_x, double pt2_y);
	double GetAngleOfTwoVector(double pt1_x, double pt1_y, double pt2_x, double pt2_y, double center_x, double center_y);
	double GetAngleOfTwoVector_0to360(double pt1_x, double pt1_y, double pt2_x, double pt2_y, double center_x, double center_y);
	void RotateVector(double pt_before_x, double pt_before_y, double center_x, double center_y, double angle, double& pt_after_x, double& pt_after_y);
	double GetEdgeSlope(double pt1_x, double pt1_y, double pt2_x, double pt2_y, EdgeType& edgeType);

	//检测直角边
	int getChessEdgeImage_4Dir(cv::Mat gray_img, cv::Mat& gray_img_marked, cv::Mat& edge_img_h_w2b, cv::Mat& edge_img_h_b2w, cv::Mat& edge_img_v_w2b, cv::Mat& edge_img_v_b2w,
		cv::Rect& edge_pos_h_w2b, cv::Rect& edge_pos_h_b2w, cv::Rect& edge_pos_v_w2b, cv::Rect& edge_pos_v_b2w,
		int cen_offset_x, int cen_offset_y, int edge_roi_length, int edge_roi_width, std::string result_folder, int* cur_block_size = nullptr);
	void findRightAngleEdges(std::vector<cv::Point> contour, std::vector<cv::Vec4i>& edges, int img_width, int img_height, int approx_polygon_error, double angle_diff_usl);
	void filterEdgesBySlope(std::vector<cv::Vec4i>& edges, std::vector<cv::Vec4i>& edges_h, std::vector<cv::Vec4i>& edges_v,
		double& avg_slope_h, double& avg_slope_v, double slope_diff_usl);
	void filterEdgesByLength(std::vector<cv::Vec4i>& edges, double length_diff_ratio_usl, double length_usl);
	void relocateEdgesAndCheckSides_4Dir(const cv::Mat& gray_img_threshold, std::vector<cv::Vec4i> edges_h, std::vector<cv::Vec4i> edges_v,
		std::vector<cv::Vec4i>& edges_h_w2b, std::vector<cv::Vec4i>& edges_h_b2w, std::vector<cv::Vec4i>& edges_v_w2b, std::vector<cv::Vec4i>& edges_v_b2w,
		double avg_slope_h, double avg_slope_v, int search_range, double slope_diff_usl, double required_roi_length, double required_roi_width);
	bool checkEdgeSides_AndBrightnessDir(const cv::Mat& gray_img_threshold, cv::Vec4i& edge, EdgeType edge_type, int required_roi_length, int required_roi_width,
		bool& is_w2b);
	void findClosestEdge(cv::Point point, std::vector<cv::Vec4i> edges, cv::Vec4i& closest_edge);

	//检测角点
	MyChessBlock* GetChessGraph_SimplifiedAuto(cv::Mat grayImg, int& cornerNum, double& meanBlockSize, int imgWidth, int imgHeight,
		std::string resultFolderPath, bool bOnlyDectBlackBlock = false, double angleOffset_axisRotate = 0.0, double* curAngle_axisRotate = NULL);
	MyChessBlock* GetChessGraph_Auto(cv::Mat grayImg, int& cornerNum, double& meanBlockSize, int imgWidth, int imgHeight,
		std::string resultFolderPath, bool bOnlyDectBlackBlock = false, double angleOffset_axisRotate = 0.0, double* curAngle_axisRotate = NULL);
	MyChessBlock* GetChessGraph(cv::Mat grayImg, int& cornerNum, double& meanBlockSize, int imgWidth, int imgHeight, CornerDectOption cornerDectOption,
		std::string resultFolderPath, double angleOffset_axisRotate = 0.0, double* curAngle_axisRotate = NULL);
	bool IsParallelogram(cv::Mat contour, cv::Mat& contour_parallelogram, int roiWidth, int roiHeight, int approxPolygonError, double lengthDiff_Ratio_USL, double angleDiff_USL);
	double CheckParallelogramsRotationAndRotateAxis(std::vector<cv::Mat>& contours_parallelogram);
	MyChessBlock ParallelogramContourToChessBlock(cv::Mat contour_parallelogram, int blockIndex, double angle_axisRotate = 0.0);
	bool IsBlackChessBlock(MyChessBlock chessblock, const cv::Mat& grayImg_threshold);
	int ConnectChessBlocks(MyChessBlock* startBlock, std::vector<MyChessBlock>& chessBlocks, std::vector<int>& bTraversed,
		std::vector<const MyChessBlock*>& unitPionter, double distance_LSL, double distance_USL, double slopeDiff_USL, double lengthDiff_Ratio_USL,
		double& sumBlockDis, int& connectionCount);
	void DeleteChessGraph(MyChessBlock* startBlock, std::vector<int>& bTraversed);
	int GetChessGraphConnerPixelAndIndexes(MyChessBlock* startBlock, std::vector<int>& bTraversed, cv::Mat grayImg,
		std::vector<cv::Point2f>& cornerPoints, bool bGetSubPixel, int subPixelWinSize, std::vector<cv::Point2i>& cornerIndexes, int curIndex_X = 0, int curIndex_Y = 0);
	int GetChessGraphBlockPositionAndIndexes(MyChessBlock* startBlock, std::vector<int>& bTraversed,
		std::vector<cv::Point2f>& blockPositions, std::vector<cv::Point2i>& blockIndexes, int curIndex_X = 0, int curIndex_Y = 0);
	bool ChangeChessGraphStartBlock_WithBlockIndex(MyChessBlock* startBlock, int blockIndex, std::vector<int>& bTraversed,
		MyChessBlock** startBlock_new, int fromWhichCorner = -1, MyChessBlock* beforeBlock = NULL);
	int TraverseChessGraph(MyChessBlock* startBlock, std::vector<int>& bTraversed);
	int SetChessGraphAxisRotationAngle(MyChessBlock* startBlock, double angle_axisRotate_new, std::vector<int>& bTraversed);
	MyChessBlock* FindClosestBlockInChessGraph(MyChessBlock* startBlock, cv::Point2d point, std::vector<int>& bTraversed, double& minDis);

	//画图存图
	int SaveGrayImg(std::string folderPath, std::string imgName, cv::Mat grayImg, int width, int height);
	int SaveRGBImg(std::string folderPath, std::string imgName, cv::Mat rgbImg, int width, int height);
	void DrawLinesOnImg(cv::Mat& img, std::vector<cv::Vec4i> lines, cv::Scalar color, int line_width);
	int DrawCountersOnImg(cv::Mat& contoursImg, std::vector<cv::Mat> contours, std::vector<bool> bContoursIsValid, unsigned char colorR, unsigned char colorG, unsigned char colorB);
	int ApproxPolyAndDrawOnImg(cv::Mat& contoursImg_polygon, std::vector<cv::Mat> contours, std::vector<bool> bContoursIsValid, int approxPolygonError, bool bClose, cv::Scalar color, int lineWidth);
	int DrawClosePolygonOnImg(cv::Mat& contoursImg_closePolygon, std::vector<cv::Mat> contours_closePolygon, cv::Scalar color, int lineWidth);
	int DrawEdgesOnImg(cv::Mat& edgesImg, std::vector<cv::Vec4i> edges, cv::Scalar color, int lineWidth);
	int DrawChessBlocksOnImg(cv::Mat& chessBlocksImg, std::vector<MyChessBlock> chessblocks, cv::Scalar color, int lineWidth);
	int DrawChessGraphOnImg(cv::Mat& chessGraphImg, MyChessBlock* startBlock, std::vector<int>& bTraversed, cv::Scalar lineColor, int lineWidth, cv::Scalar connetCornerColor);

	cv::Mat m_GrayImage;
	int m_ImgWidth;
	int m_ImgHeight;
	int m_ImgHalfWidth;
	int m_ImgHalfHeight;

	cv::Mat m_MarkedImage;

	typedef int(*pOutputLog)(std::string log);
	pOutputLog m_OutputLog;
	std::string m_ResultFolderPath;
	bool m_bSaveImg;

public:
	/**
	* \brief 初始化赋值
	* \param grayImage			输入，用于检测的灰图（Type：CV_8UC1）
	* \param OutputLogFuntion	输入，设置打印 log 函数，默认为 NULL
	* \param resultFolderPath	输入，Debug 用图片输出文件夹，默认为 空
	* \param bSaveImg			输入，是否输出过程 Debug 用图片，默认为 false
	* \return 执行无误返回 0，执行错误返回 -1
	*/
	int Init(cv::Mat& grayImage, pOutputLog OutputLogFuntion = NULL, std::string resultFolderPath = "", bool bSaveImg = false);

	/**
	* \brief 获取单张棋盘格图片的角点
	* \param cornerPoints		输出，找到的角点坐标
	* \param cornerIndexes		输出，找到的角点所在的行列位置（与角点坐标一一对应）
	* \param blockPositions	输出，找到的 block 坐标
	* \param blockIndexes		输出，找到的 block 所在的行列位置（与 block 坐标一一对应）
	* \param cornerDectOption	输入，角点检测参数
	* \return 执行无误返回 0，执行错误返回 -1
	*/
	int ProcessGetChessCornerPointsAndIndex_SingleImage(std::vector<cv::Point2f>& cornerPoints, std::vector<cv::Point2i>& cornerIndexes,
		CornerDectOption cornerDectOption, std::vector<cv::Point2f>* blockPositions = NULL, std::vector<cv::Point2i>* blockIndexes = NULL);

	/**
	* @brief 获取棋盘格图片的多个 SFR 检测点位四周最近的刀边图
	* @param [out] edge_img_h_w2b					找到的各个点位最近的水平刀边图（颜色-白到黑）（未找到的点位会被设置为空 Mat）
	* @param [out] edge_img_h_b2w					找到的各个点位最近的水平刀边图（颜色-黑到白）（未找到的点位会被设置为空 Mat）
	* @param [out] edge_img_v_w2b					找到的各个点位最近的垂直刀边图（颜色-白到黑）（未找到的点位会被设置为空 Mat）
	* @param [out] edge_img_v_b2w					找到的各个点位最近的垂直刀边图（颜色-黑到白）（未找到的点位会被设置为空 Mat）
	* @param [in] sfr_edge_dect_option				刀边检测参数
	* @param [out] edge_pos_up						找到的最近水平刀边图（颜色-白到黑）在原图像位置（与水平刀边图一一对应，未找到的点位会被设置为 (x:-1,y:-1,width:-1,height:-1)）
	* @param [out] edge_pos_down					找到的最近水平刀边图（颜色-黑到白）在原图像位置（与垂直刀边图一一对应，未找到的点位会被设置为 (x:-1,y:-1,width:-1,height:-1)）
	* @param [out] edge_pos_left					找到的最近垂直刀边图（颜色-白到黑）在原图像位置（与水平刀边图一一对应，未找到的点位会被设置为 (x:-1,y:-1,width:-1,height:-1)）
	* @param [out] edge_pos_right					找到的最近垂直刀边图（颜色-黑到白）在原图像位置（与垂直刀边图一一对应，未找到的点位会被设置为 (x:-1,y:-1,width:-1,height:-1)）
	* @return 0										成功
	* @return >0									失败
	*/
	int ProcessGetChessSfrEdgeImages_4Dir(std::vector<cv::Mat>& edge_img_h_w2b, std::vector<cv::Mat>& edge_img_h_b2w,
		std::vector<cv::Mat>& edge_img_v_w2b, std::vector<cv::Mat>& edge_img_v_b2w,
		SfrEdgeDectOption sfr_edge_dect_option,
		std::vector<cv::Rect>* edge_pos_h_w2b = nullptr, std::vector<cv::Rect>* edge_pos_h_b2w = nullptr,
		std::vector<cv::Rect>* edge_pos_v_w2b = nullptr, std::vector<cv::Rect>* edge_pos_v_b2w = nullptr);

	/**
	* \brief 获取检测结果标识图片
	* \param markedImg		输出，用于接收结果标识图片
	* \return 空
	*/
	void GetMarkedImage(cv::Mat& markedImg);
};