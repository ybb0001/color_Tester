#include"ChessBoard_CornerDetection.h"

ChessBoard_CornerDetection::ChessBoard_CornerDetection()
{
	m_ImgWidth = 0;
	m_ImgHeight = 0;
	m_ImgHalfWidth = 0;
	m_ImgHalfHeight = 0;

	m_OutputLog = NULL;
	m_ResultFolderPath = "";
	m_bSaveImg = false;
}

ChessBoard_CornerDetection::~ChessBoard_CornerDetection()
{
	if (!m_GrayImage.empty()) {
		m_GrayImage.release();
	}

	if (!m_MarkedImage.empty()) {
		m_MarkedImage.release();
	}
}

int ChessBoard_CornerDetection::Init(cv::Mat& grayImage, pOutputLog OutputLogFuntion, std::string resultFolderPath, bool bSaveImg)
{
	m_OutputLog = OutputLogFuntion;
	m_ResultFolderPath = resultFolderPath;
	m_bSaveImg = bSaveImg;

	if (grayImage.type() != CV_8UC1) {
		if (m_OutputLog) {
			m_OutputLog("Image Type error");
		}
		return -1;
	}

	if (!m_GrayImage.empty()) {
		m_GrayImage.release();
	}
	m_GrayImage = grayImage.clone();
	m_ImgWidth = m_GrayImage.cols;
	m_ImgHeight = m_GrayImage.rows;
	m_ImgHalfWidth = m_ImgWidth / 2;
	m_ImgHalfHeight = m_ImgHeight / 2;

	if (!m_MarkedImage.empty()) {
		m_MarkedImage.release();
	}
	m_MarkedImage = cv::Mat::zeros(grayImage.size(), CV_8UC3);
	cv::cvtColor(m_GrayImage, m_MarkedImage, CV_GRAY2RGB);

	return 0;
}

int ChessBoard_CornerDetection::ProcessGetChessCornerPointsAndIndex_SingleImage(std::vector<cv::Point2f>& cornerPoints, std::vector<cv::Point2i>& cornerIndexes,
	CornerDectOption cornerDectOption, std::vector<cv::Point2f>* blockPositions, std::vector<cv::Point2i>* blockIndexes)
{
	if (m_GrayImage.empty()) {
		if (m_OutputLog) {
			m_OutputLog("Empty Image");
		}
		return -1;
	}

	if (!cornerPoints.empty()) {
		cornerPoints.clear();
	}
	if (!cornerIndexes.empty()) {
		cornerIndexes.clear();
	}
	if (blockPositions) {
		if (!(*blockPositions).empty()) {
			(*blockPositions).clear();
		}
	}
	if (blockIndexes) {
		if (!(*blockIndexes).empty()) {
			(*blockIndexes).clear();
		}
	}

	int ret = 0;
	std::vector<int> bTraversed;

	// 1.寻找棋盘格方块并建立关系
	int cornerNum = 0;
	double meanBlockSize = 0;
	MyChessBlock* startBlock = NULL;
	if (cornerDectOption.chessDectType == 0) {
		startBlock = GetChessGraph_SimplifiedAuto(m_GrayImage, cornerNum, meanBlockSize, m_ImgWidth, m_ImgHeight, m_ResultFolderPath);
	}
	else if (cornerDectOption.chessDectType == 1) {
		startBlock = GetChessGraph_Auto(m_GrayImage, cornerNum, meanBlockSize, m_ImgWidth, m_ImgHeight, m_ResultFolderPath, true);	//全参数搜索检测时，设置为只检测黑块
	}
	else {
		startBlock = GetChessGraph(m_GrayImage, cornerNum, meanBlockSize, m_ImgWidth, m_ImgHeight, cornerDectOption, m_ResultFolderPath);
	}
	if (!startBlock) {
		m_OutputLog("No Blocks detected!");
		return -1;
	}
	if (cornerNum < cornerDectOption.minCornerNum) {
		m_OutputLog("Not enough Blocks detected!");

		bTraversed.clear();
		DeleteChessGraph(startBlock, bTraversed);
		startBlock = NULL;
		return -1;
	}

	// 2.精确化角点位置
	std::vector<cv::Point2f> tempCornerPoints;
	std::vector<cv::Point2i> tempCornerIndexes;
	int subPixelWinSize = (meanBlockSize == 0) ? 5 : ((int)(meanBlockSize / 8.0 + 0.5));
	bTraversed.clear();
	GetChessGraphConnerPixelAndIndexes(startBlock, bTraversed, m_GrayImage, tempCornerPoints, true, subPixelWinSize, tempCornerIndexes);

	// 3.角点index(0,0)修正
	double minDis = FLT_MAX;
	int indexOffsetX = 0, indexOffsetY = 0;
	for (int i = 0; i < tempCornerPoints.size(); i++) {
		double curDis = GetDistanceOfTwoPoint(tempCornerPoints[i].x, tempCornerPoints[i].y, cornerDectOption.startCountPos_X, cornerDectOption.startCountPos_Y);
		if (curDis < minDis) {
			minDis = curDis;
			indexOffsetX = tempCornerIndexes[i].x;
			indexOffsetY = tempCornerIndexes[i].y;
		}
	}
	for (int i = 0; i < tempCornerIndexes.size(); i++) {
		tempCornerIndexes[i].x -= indexOffsetX;
		tempCornerIndexes[i].y -= indexOffsetY;
	}

	// 4.角点结果赋值
	cornerPoints = tempCornerPoints;
	cornerIndexes = tempCornerIndexes;

	if (blockPositions || blockIndexes) {
		// 5.获取 block 位置
		std::vector<cv::Point2f> tempBlockPositions;
		std::vector<cv::Point2i> tempBlockIndexes;
		bTraversed.clear();
		GetChessGraphBlockPositionAndIndexes(startBlock, bTraversed, tempBlockPositions, tempBlockIndexes);

		// 6.block index(0,0)修正
		minDis = FLT_MAX;
		indexOffsetX = 0, indexOffsetY = 0;
		for (int i = 0; i < tempBlockPositions.size(); i++) {
			double curDis = GetDistanceOfTwoPoint(tempBlockPositions[i].x, tempBlockPositions[i].y, cornerDectOption.startCountPos_X, cornerDectOption.startCountPos_Y);
			if (curDis < minDis) {
				minDis = curDis;
				indexOffsetX = tempBlockIndexes[i].x;
				indexOffsetY = tempBlockIndexes[i].y;
			}
		}
		for (int i = 0; i < tempBlockIndexes.size(); i++) {
			tempBlockIndexes[i].x -= indexOffsetX;
			tempBlockIndexes[i].y -= indexOffsetY;
		}

		// 7.block 结果赋值
		if (blockPositions) {
			*blockPositions = tempBlockPositions;
		}
		if (blockIndexes) {
			*blockIndexes = tempBlockIndexes;
		}
	}

	// 8.画图，标识检测结果
	bTraversed.clear();
	DrawChessGraphOnImg(m_MarkedImage, startBlock, bTraversed, cv::Scalar(0, 255, 0), 2, cv::Scalar(0, 255, 255));
	cv::line(m_MarkedImage, startBlock->centerPoint, startBlock->centerPoint, cv::Scalar(255, 0, 0), 25);
	cv::Point2d corner_origin(-1, -1);
	std::vector<cv::Point2i>::iterator findCornerOriginPos = find(cornerIndexes.begin(), cornerIndexes.end(), cv::Point2i(0, 0));
	if (findCornerOriginPos != cornerIndexes.end()) {
		int j = findCornerOriginPos - cornerIndexes.begin();
		corner_origin = cornerPoints[j];
	}
	cv::line(m_MarkedImage, corner_origin, corner_origin, cv::Scalar(255, 255, 0), 25);
	if (blockPositions && blockIndexes) {
		cv::Point2d block_origin(-1, -1);
		std::vector<cv::Point2i>::iterator findBlockOriginPos = find((*blockIndexes).begin(), (*blockIndexes).end(), cv::Point2i(0, 0));
		if (findBlockOriginPos != (*blockIndexes).end()) {
			int j = findBlockOriginPos - (*blockIndexes).begin();
			block_origin = (*blockPositions)[j];
		}
		cv::line(m_MarkedImage, block_origin, block_origin, cv::Scalar(255, 255, 0), 25);
	}
	if (m_bSaveImg) {
		SaveRGBImg(m_ResultFolderPath, "Img_FindCorner", m_MarkedImage, m_ImgWidth, m_ImgHeight);
	}

	// 9.清空graph
	bTraversed.clear();
	DeleteChessGraph(startBlock, bTraversed);
	startBlock = NULL;

	return 0;
}

ChessBoard_CornerDetection::MyChessBlock* ChessBoard_CornerDetection::GetChessGraph(cv::Mat grayImg, int& cornerNum, double& meanBlockSize, int imgWidth, int imgHeight, CornerDectOption cornerDectOption,
	std::string resultFolderPath, double angleOffset_axisRotate, double* curAngle_axisRotate)
{
	// TODO: 识别棋盘格，建立block间关系（非自动检测，全部参数需指定）

	int ret = 0;
	std::vector<int> bTraversed;

	int startCountPos_X = cornerDectOption.startCountPos_X;
	int startCountPos_Y = cornerDectOption.startCountPos_Y;

	// 0.直方图均衡化
	cv::Mat grayImg_hist;
	cv::equalizeHist(grayImg, grayImg_hist);

	// 1.二值化
	cv::Mat grayImg_threshold;
	int threshold_value = cornerDectOption.thresholdValue;
	if (threshold_value == -1) {
		cv::threshold(grayImg_hist, grayImg_threshold, 0, 255, cv::THRESH_OTSU);
	}
	else {
		cv::threshold(grayImg_hist, grayImg_threshold, threshold_value, 255, cv::THRESH_BINARY);
	}

	if (m_bSaveImg) {
		SaveGrayImg(resultFolderPath, "Img_Threshold", grayImg_threshold, imgWidth, imgHeight);
	}

	// 2.膨胀或腐蚀
	cv::Mat element/* = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize))*/;
	cv::Mat grayImg_morph;

	if (cornerDectOption.morphIter <= 0) {
		grayImg_morph = grayImg_threshold.clone();
	}
	else {
		if (cornerDectOption.morphType == 0) {
			cv::dilate(grayImg_threshold, grayImg_morph, element, cv::Point(1, 1), cornerDectOption.morphIter);	// 膨胀
		}
		else if (cornerDectOption.morphType == 1) {
			cv::erode(grayImg_threshold, grayImg_morph, element, cv::Point(1, 1), cornerDectOption.morphIter);	// 腐蚀
		}
		else {
			int pointRefX = imgWidth / 2;
			int pointRefY = imgHeight / 2;
			uchar pixelValueRef = grayImg_threshold.at<uchar>(pointRefX, pointRefY);
			if (pixelValueRef) {
				cv::erode(grayImg_threshold, grayImg_morph, element, cv::Point(1, 1), cornerDectOption.morphIter);	// 腐蚀
			}
			else {
				cv::dilate(grayImg_threshold, grayImg_morph, element, cv::Point(1, 1), cornerDectOption.morphIter);	// 膨胀
			}
		}
	}

	if (m_bSaveImg) {
		SaveGrayImg(resultFolderPath, "Img_Morph", grayImg_morph, imgWidth, imgHeight);
	}

	// 3.寻找边界
	//std::vector<cv::Mat> contours(100000);
	//cv::findContours(grayImg_morph, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	std::vector<cv::Mat> contours;
	findContours2(grayImg_morph, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

#if 0
	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Rect rect = cv::boundingRect(contours[i]);

		if (rect.height == grayImg.rows - 2)	// 此处排掉占据整个图像大小的边界
		{
			contours.erase(contours.begin() + i);
			i--;
			continue;
		}
		else if (rect.width <= 15 || rect.height <= 15)	// 此处排掉太小的边界
		{
			contours.erase(contours.begin() + i);
			i--;
			continue;
		}
	}
#else
	std::vector<bool> bContoursIsValid(contours.size(), true);

	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Rect rect = cv::boundingRect(contours[i]);

		if (rect.height == grayImg.rows - 2)	// 此处排掉占据整个图像大小的边界
		{
			bContoursIsValid[i] = false;
		}
		else if (rect.width <= 15 || rect.height <= 15)	// 此处排掉太小的边界
		{
			bContoursIsValid[i] = false;
		}
	}
#endif

	if (m_bSaveImg) {
		cv::Mat contoursImg = cv::Mat::zeros(grayImg.size(), CV_8UC3);
		DrawCountersOnImg(contoursImg, contours, bContoursIsValid, 255, 0, 0);
		SaveRGBImg(resultFolderPath, "Img_Contours", contoursImg, imgWidth, imgHeight);
	}

	// 4.多边形近似（闭合近似）并寻找四边形
	std::vector<cv::Mat> contours_parallelogram;
	double lengthDiff_Ratio_USL_parallelogram = 0.5;
	double angleDiff_USL_parallelogram = 15.0/*8.0*//*5.0*/;
	for (int i = 0; i < contours.size(); i++) {
		if (bContoursIsValid[i]) {
			cv::Mat temp_contour;
			if (IsParallelogram(contours[i], temp_contour, imgWidth, imgHeight, cornerDectOption.approxPolygonError, lengthDiff_Ratio_USL_parallelogram, angleDiff_USL_parallelogram)) {
				contours_parallelogram.push_back(temp_contour);
			}
		}
	}
	if (contours_parallelogram.size() == 0) {
		return NULL;
	}

	if (m_bSaveImg) {
		cv::Mat contoursImg_polygon = cv::Mat::zeros(grayImg.size(), CV_8UC3);
		ApproxPolyAndDrawOnImg(contoursImg_polygon, contours, bContoursIsValid, cornerDectOption.approxPolygonError, true, cv::Scalar(0, 0, 255), 2);
		SaveRGBImg(resultFolderPath, "Img_ApproxPolygonClose", contoursImg_polygon, imgWidth, imgHeight);
		cv::Mat contoursImg_parallelogram = cv::Mat::zeros(grayImg.size(), CV_8UC3);
		DrawClosePolygonOnImg(contoursImg_parallelogram, contours_parallelogram, cv::Scalar(0, 0, 255), 2);
		SaveRGBImg(resultFolderPath, "Img_Parallelograms", contoursImg_parallelogram, imgWidth, imgHeight);
	}

	// 5.判断四边形在画面中的旋转角度，旋转坐标系
	if (angleOffset_axisRotate != 0.0) {
		for (int i = 0; i < contours_parallelogram.size(); i++) {
			for (int j = 0; j < contours_parallelogram[i].rows; j++) {
				double x_before = contours_parallelogram[i].at<cv::Vec2i>(j, 0)[0];
				double y_before = contours_parallelogram[i].at<cv::Vec2i>(j, 0)[1];
				double x_new, y_new;
				RotateVector(x_before, y_before, 0, 0, -angleOffset_axisRotate, x_new, y_new);
				contours_parallelogram[i].at<cv::Vec2i>(j, 0)[0] = x_new;
				contours_parallelogram[i].at<cv::Vec2i>(j, 0)[1] = y_new;
			}
		}
	}
	double angle_axisRotate = CheckParallelogramsRotationAndRotateAxis(contours_parallelogram);

	// 6.创建block数组
	std::vector<MyChessBlock> chessBlocks_found;
	std::vector<double> blocksPos_x, blocksPos_y;
	for (int i = 0; i < contours_parallelogram.size(); i++) {
		MyChessBlock tempBlock = ParallelogramContourToChessBlock(contours_parallelogram[i], i, angle_axisRotate + angleOffset_axisRotate);
		if (tempBlock.bIsValid) {
			chessBlocks_found.push_back(tempBlock);
			blocksPos_x.push_back(tempBlock.centerPoint.x);
			blocksPos_y.push_back(tempBlock.centerPoint.y);
		}
	}
	if (m_bSaveImg) {
		cv::Mat chessBlocksImg = cv::Mat::zeros(grayImg.size(), CV_8UC3);
		DrawChessBlocksOnImg(chessBlocksImg, chessBlocks_found, cv::Scalar(0, 0, 255), 2);
		SaveRGBImg(resultFolderPath, "Img_ChessBlocks", chessBlocksImg, imgWidth, imgHeight);
	}

	if (chessBlocks_found.size() == 0) {
		return NULL;
	}

	// 7.设置5个起始搜索的点，从离这5个点最近的block开始搜索
	std::sort(blocksPos_x.begin(), blocksPos_x.end());
	std::sort(blocksPos_y.begin(), blocksPos_y.end());

	double startFindPos_X[5];
	double startFindPos_Y[5];
	for (int ptCnt = 0; ptCnt < 5; ptCnt++) {
		int tempIndex_x, tempIndex_y;
		switch (ptCnt) {
		case 0:
		case 3:
			tempIndex_x = (int)((double)blocksPos_x.size() / 4.0); break;
		case 1:
		case 4:
			tempIndex_x = (int)((double)blocksPos_x.size() / 4.0 * 3.0); break;
		default:
			tempIndex_x = (int)((double)blocksPos_x.size() / 2.0); break;
		}
		switch (ptCnt) {
		case 0:
		case 1:
			tempIndex_y = (int)((double)blocksPos_y.size() / 4.0); break;
		case 3:
		case 4:
			tempIndex_y = (int)((double)blocksPos_y.size() / 4.0 * 3.0); break;
		default:
			tempIndex_y = (int)((double)blocksPos_y.size() / 2.0); break;
		}
		startFindPos_X[ptCnt] = blocksPos_x[tempIndex_x];
		startFindPos_Y[ptCnt] = blocksPos_y[tempIndex_y];
	}

	std::vector<std::vector<std::pair<double, int>>> closestChessBlocksIndex(5);
	for (int blockCnt = 0; blockCnt < chessBlocks_found.size(); blockCnt++) {
		for (int ptCnt = 0; ptCnt < 5; ptCnt++) {
			double curDis = GetDistanceOfTwoPoint(startFindPos_X[ptCnt], startFindPos_Y[ptCnt], chessBlocks_found[blockCnt].centerPoint.x, chessBlocks_found[blockCnt].centerPoint.y);
			closestChessBlocksIndex[ptCnt].push_back(std::pair<double, int>(curDis, blockCnt));
		}
	}

	for (int ptCnt = 0; ptCnt < 5; ptCnt++) {
		std::sort(closestChessBlocksIndex[ptCnt].begin(), closestChessBlocksIndex[ptCnt].end());
	}

	// 8.变更起始block，循环搜索，获取block位置关系并找到可相连block数量最大时的最适起始block
	MyChessBlock* startBlock = NULL;
	double distance_LSL;
	double distance_USL;
	if (cornerDectOption.morphIter <= 0) {
		distance_LSL = 0;
		distance_USL = 4.0 + cornerDectOption.approxPolygonError * 2.0;/*2.5;*/
	}
	else {
		distance_LSL = cornerDectOption.morphIter * 0;
		distance_USL = cornerDectOption.morphIter * 4.0 + cornerDectOption.approxPolygonError * 2.0;/*2.5;*/
	}
	double slopeDiff_USL_connectEdge = 0.15;
	double lengthDiff_Ratio_USL_connectEdge = 0.25;

	int curCornerNum_blockLoop = 0;
	int maxCornerNum_blockLoop = 0;
	int foundCnt_blockLoop = 0;
	int bestBlockIndex = -1;

	for (int blockCnt = 0; blockCnt < chessBlocks_found.size(); blockCnt++)
	{
		for (int ptCnt = 0; ptCnt < 5; ptCnt++)
		{
			startBlock = new MyChessBlock;

			int curStartBlockIndex = closestChessBlocksIndex[ptCnt][blockCnt].second;

			*startBlock = chessBlocks_found[curStartBlockIndex];

			std::vector<MyChessBlock> chessBlocks_forGetGraph = chessBlocks_found;	//ConnectChessBlocks递归过程中删除4角连满的block，所以复制一份
			std::vector<const MyChessBlock*> unitPionter(chessBlocks_forGetGraph.size(), NULL);	//记录graph中block的地址
			double sumBlockDis = 0;
			int connectionCount = 0;	//记录总角点个数
			bTraversed.clear();
			ConnectChessBlocks(startBlock, chessBlocks_forGetGraph, bTraversed, unitPionter, distance_LSL, distance_USL, slopeDiff_USL_connectEdge, lengthDiff_Ratio_USL_connectEdge,
				sumBlockDis, connectionCount);

			curCornerNum_blockLoop = connectionCount;

			if (curCornerNum_blockLoop > maxCornerNum_blockLoop) {
				bestBlockIndex = curStartBlockIndex;
				maxCornerNum_blockLoop = curCornerNum_blockLoop;
				foundCnt_blockLoop = 0;
			}
			else if (curCornerNum_blockLoop == maxCornerNum_blockLoop) {
				if (maxCornerNum_blockLoop) {
					foundCnt_blockLoop++;
				}
			}

			bTraversed.clear();
			DeleteChessGraph(startBlock, bTraversed);
			startBlock = NULL;

			if (foundCnt_blockLoop >= 5) {
				break;
			}
		}

		if (foundCnt_blockLoop >= 5) {
			break;
		}
	}

	// 9.根据搜索到的最适结果，创建block位置关系
	startBlock = new MyChessBlock;
	*startBlock = chessBlocks_found[bestBlockIndex];

	std::vector<MyChessBlock> chessBlocks_forGetGraph = chessBlocks_found;	//ConnectChessBlocks递归过程中删除4角连满的block，所以复制一份
	std::vector<const MyChessBlock*> unitPionter(chessBlocks_forGetGraph.size(), NULL);	//记录graph中block的地址
	double sumBlockDis = 0;
	int connectionCount = 0;	//记录总角点个数
	bTraversed.clear();
	ConnectChessBlocks(startBlock, chessBlocks_forGetGraph, bTraversed, unitPionter, distance_LSL, distance_USL, slopeDiff_USL_connectEdge, lengthDiff_Ratio_USL_connectEdge,
		sumBlockDis, connectionCount);
	cornerNum = connectionCount;
	meanBlockSize = connectionCount ? (sumBlockDis / (double)connectionCount / sqrt(2)) : 0;

	// 10.恢复坐标系旋转
	if (curAngle_axisRotate) {
		*curAngle_axisRotate = startBlock->angle_axisRotate;
	}
	bTraversed.clear();
	SetChessGraphAxisRotationAngle(startBlock, 0, bTraversed);

	if (m_bSaveImg) {
		cv::Mat chessGraphImg = cv::Mat::zeros(grayImg.size(), CV_8UC3);
		DrawChessBlocksOnImg(chessGraphImg, chessBlocks_found, cv::Scalar(0, 0, 255), 2);
		cv::line(chessGraphImg, startBlock->centerPoint, startBlock->centerPoint, cv::Scalar(255, 0, 0), 25);
		bTraversed.clear();
		DrawChessGraphOnImg(chessGraphImg, startBlock, bTraversed, cv::Scalar(0, 255, 0), 2, cv::Scalar(0, 255, 255));
		SaveRGBImg(resultFolderPath, "Img_ChessGraph", chessGraphImg, imgWidth, imgHeight);
	}

	return startBlock;
}

ChessBoard_CornerDetection::MyChessBlock* ChessBoard_CornerDetection::GetChessGraph_SimplifiedAuto(cv::Mat grayImg, int& cornerNum, double& meanBlockSize, int imgWidth, int imgHeight,
	std::string resultFolderPath, bool bOnlyDectBlackBlock, double angleOffset_axisRotate, double* curAngle_axisRotate)
{
	// TODO: 识别棋盘格，建立block间关系（仅二值化参数循环搜索，节省些许时间），可指定是否只检测黑块

	int ret = 0;
	std::vector<int> bTraversed;

	MyChessBlock* startBlock = NULL;
	std::vector<MyChessBlock> chessBlocks_found;
	int bestBlockIndex = -1;
	double distance_LSL;
	double distance_USL;
	double slopeDiff_USL_connectEdge = 0.15;
	double lengthDiff_Ratio_USL_connectEdge = 0.25;

	cv::Mat grayImg_threshold;
	cv::Mat grayImg_morph;

	// 0.直方图均衡化
	cv::Mat grayImg_hist;
	cv::equalizeHist(grayImg, grayImg_hist);

	// 1.二值化，同时自适应二值化均值窗大小根据上次找到的block大小变化
	double meanBestBlockSize = 0;

	for (int thresholdCnt = 0; thresholdCnt < 2; thresholdCnt++)
	{
		if (thresholdCnt == 1 && meanBestBlockSize == 0) {
			break;
		}

		int thresholdBlockSize = meanBestBlockSize == 0 ? ((int)((double)imgWidth * 0.2) | 1) : ((int)(meanBestBlockSize * 2/*4*/) | 1);
		int thresholdOffset = 0;
		cv::adaptiveThreshold(grayImg_hist, grayImg_threshold, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, thresholdBlockSize, 0);

		if (m_bSaveImg) {
			SaveGrayImg(resultFolderPath, "Img_Threshold", grayImg_threshold, imgWidth, imgHeight);
		}

		// 2.膨胀或腐蚀，识别到的块颜色不对时重新膨胀或腐蚀（bOnlyDectBlackBlock为true时）
		int morphIter_max = 9/*9*/;	//膨胀或腐蚀迭代次数参数 最大值
		int morphIter_step = 2;	//膨胀或腐蚀迭代次数参数 变更步长
		bool bReMorph = true;

		cv::Mat element;
		int morphIter = 5;

		while (bReMorph) {
			if (morphIter <= 0) {
				grayImg_morph = grayImg_threshold.clone();
			}
			else {
				cv::dilate(grayImg_threshold, grayImg_morph, element, cv::Point(1, 1), morphIter);	// 膨胀
			}

			if (m_bSaveImg) {
				SaveGrayImg(resultFolderPath, "Img_Morph", grayImg_morph, imgWidth, imgHeight);
			}

			// 3.寻找边界
			//std::vector<cv::Mat> contours(100000);
			//cv::findContours(grayImg_morph, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
			std::vector<cv::Mat> contours;
			findContours2(grayImg_morph, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

#if 0
			for (size_t i = 0; i < contours.size(); i++)
			{
				cv::Rect rect = cv::boundingRect(contours[i]);

				if (rect.height == grayImg.rows - 2)	// 此处排掉占据整个图像大小的边界
				{
					contours.erase(contours.begin() + i);
					i--;
					continue;
				}
				else if (rect.width <= 15 || rect.height <= 15)	// 此处排掉太小的边界
				{
					contours.erase(contours.begin() + i);
					i--;
					continue;
				}
			}
#else
			std::vector<bool> bContoursIsValid(contours.size(), true);

			for (size_t i = 0; i < contours.size(); i++)
			{
				cv::Rect rect = cv::boundingRect(contours[i]);

				if (rect.height == grayImg.rows - 2)	// 此处排掉占据整个图像大小的边界
				{
					bContoursIsValid[i] = false;
				}
				else if (rect.width <= 15 || rect.height <= 15)	// 此处排掉太小的边界
				{
					bContoursIsValid[i] = false;
				}
			}
#endif

			if (m_bSaveImg) {
				cv::Mat contoursImg = cv::Mat::zeros(grayImg.size(), CV_8UC3);
				DrawCountersOnImg(contoursImg, contours, bContoursIsValid, 255, 0, 0);
				SaveRGBImg(resultFolderPath, "Img_Contours", contoursImg, imgWidth, imgHeight);
			}

			// 4.多边形近似（闭合近似）并寻找四边形
			int approxPolygonError = meanBestBlockSize == 0 ? 10/*20*/ : (int)(meanBestBlockSize / 10.0/*5.0*/);
			std::vector<cv::Mat> contours_parallelogram;
			double lengthDiff_Ratio_USL_parallelogram = 0.5;
			double angleDiff_USL_parallelogram = 15.0/*8.0*//*5.0*/;
			for (int i = 0; i < contours.size(); i++) {
				if (bContoursIsValid[i]) {
					cv::Mat temp_contour;
					if (IsParallelogram(contours[i], temp_contour, imgWidth, imgHeight, approxPolygonError, lengthDiff_Ratio_USL_parallelogram, angleDiff_USL_parallelogram)) {
						contours_parallelogram.push_back(temp_contour);
					}
				}
			}
			if (contours_parallelogram.size() == 0) {
				continue;
			}

			if (m_bSaveImg) {
				cv::Mat contoursImg_polygon = cv::Mat::zeros(grayImg.size(), CV_8UC3);
				ApproxPolyAndDrawOnImg(contoursImg_polygon, contours, bContoursIsValid, approxPolygonError, true, cv::Scalar(0, 0, 255), 2);
				SaveRGBImg(resultFolderPath, "Img_ApproxPolygonClose", contoursImg_polygon, imgWidth, imgHeight);
				cv::Mat contoursImg_parallelogram = cv::Mat::zeros(grayImg.size(), CV_8UC3);
				DrawClosePolygonOnImg(contoursImg_parallelogram, contours_parallelogram, cv::Scalar(0, 0, 255), 2);
				SaveRGBImg(resultFolderPath, "Img_Parallelograms", contoursImg_parallelogram, imgWidth, imgHeight);
			}

			// 5.判断四边形在画面中的旋转角度，旋转坐标系
			if (angleOffset_axisRotate != 0.0) {
				for (int i = 0; i < contours_parallelogram.size(); i++) {
					for (int j = 0; j < contours_parallelogram[i].rows; j++) {
						double x_before = contours_parallelogram[i].at<cv::Vec2i>(j, 0)[0];
						double y_before = contours_parallelogram[i].at<cv::Vec2i>(j, 0)[1];
						double x_new, y_new;
						RotateVector(x_before, y_before, 0, 0, -angleOffset_axisRotate, x_new, y_new);
						contours_parallelogram[i].at<cv::Vec2i>(j, 0)[0] = x_new;
						contours_parallelogram[i].at<cv::Vec2i>(j, 0)[1] = y_new;
					}
				}
			}
			double angle_axisRotate = CheckParallelogramsRotationAndRotateAxis(contours_parallelogram);

			// 6.创建block数组
			chessBlocks_found.clear();
			std::vector<double> blocksPos_x, blocksPos_y;
			for (int i = 0; i < contours_parallelogram.size(); i++) {
				MyChessBlock tempBlock = ParallelogramContourToChessBlock(contours_parallelogram[i], i, angle_axisRotate + angleOffset_axisRotate);
				if (tempBlock.bIsValid) {
					chessBlocks_found.push_back(tempBlock);
					blocksPos_x.push_back(tempBlock.centerPoint.x);
					blocksPos_y.push_back(tempBlock.centerPoint.y);
				}
			}
			if (m_bSaveImg) {
				cv::Mat chessBlocksImg = cv::Mat::zeros(grayImg.size(), CV_8UC3);
				DrawChessBlocksOnImg(chessBlocksImg, chessBlocks_found, cv::Scalar(0, 0, 255), 2);
				SaveRGBImg(resultFolderPath, "Img_ChessBlocks", chessBlocksImg, imgWidth, imgHeight);
			}

			if (chessBlocks_found.size() == 0) {
				continue;
			}

			// 7.设置5个起始搜索的点，从离这5个点最近的block开始搜索
			std::sort(blocksPos_x.begin(), blocksPos_x.end());
			std::sort(blocksPos_y.begin(), blocksPos_y.end());

			double startFindPos_X[5];
			double startFindPos_Y[5];
			for (int ptCnt = 0; ptCnt < 5; ptCnt++) {
				int tempIndex_x, tempIndex_y;
				switch (ptCnt) {
				case 0:
				case 3:
					tempIndex_x = (int)((double)blocksPos_x.size() / 4.0); break;
				case 1:
				case 4:
					tempIndex_x = (int)((double)blocksPos_x.size() / 4.0 * 3.0); break;
				default:
					tempIndex_x = (int)((double)blocksPos_x.size() / 2.0); break;
				}
				switch (ptCnt) {
				case 0:
				case 1:
					tempIndex_y = (int)((double)blocksPos_y.size() / 4.0); break;
				case 3:
				case 4:
					tempIndex_y = (int)((double)blocksPos_y.size() / 4.0 * 3.0); break;
				default:
					tempIndex_y = (int)((double)blocksPos_y.size() / 2.0); break;
				}
				startFindPos_X[ptCnt] = blocksPos_x[tempIndex_x];
				startFindPos_Y[ptCnt] = blocksPos_y[tempIndex_y];
			}

			std::vector<std::vector<std::pair<double, int>>> closestChessBlocksIndex(5);
			for (int blockCnt = 0; blockCnt < chessBlocks_found.size(); blockCnt++) {
				for (int ptCnt = 0; ptCnt < 5; ptCnt++) {
					double curDis = GetDistanceOfTwoPoint(startFindPos_X[ptCnt], startFindPos_Y[ptCnt], chessBlocks_found[blockCnt].centerPoint.x, chessBlocks_found[blockCnt].centerPoint.y);
					closestChessBlocksIndex[ptCnt].push_back(std::pair<double, int>(curDis, blockCnt));
				}
			}

			for (int ptCnt = 0; ptCnt < 5; ptCnt++) {
				std::sort(closestChessBlocksIndex[ptCnt].begin(), closestChessBlocksIndex[ptCnt].end());
			}

			// 8.变更起始block，循环搜索，获取block位置关系并找到可相连block数量最大时的最适起始block
			if (morphIter <= 0) {
				distance_LSL = 0;
				distance_USL = 4.0 + approxPolygonError * 2.0;/*2.5;*/
			}
			else {
				distance_LSL = morphIter * 0;
				distance_USL = morphIter * 4.0 + approxPolygonError * 2.0;/*2.5;*/
			}

			double meanCurBlockSize = 0;

			int curCornerNum_blockLoop = 0;
			int maxCornerNum_blockLoop = 0;
			int foundCnt_blockLoop = 0;
			bestBlockIndex = -1;

			for (int blockCnt = 0; blockCnt < chessBlocks_found.size(); blockCnt++)
			{
				for (int ptCnt = 0; ptCnt < 5; ptCnt++)
				{
					startBlock = new MyChessBlock;

					int curStartBlockIndex = closestChessBlocksIndex[ptCnt][blockCnt].second;

					*startBlock = chessBlocks_found[curStartBlockIndex];

					std::vector<MyChessBlock> chessBlocks_forGetGraph = chessBlocks_found;	//ConnectChessBlocks递归过程中删除4角连满的block，所以复制一份
					std::vector<const MyChessBlock*> unitPionter(chessBlocks_forGetGraph.size(), NULL);	//记录graph中block的地址
					double sumBlockDis = 0;
					int connectionCount = 0;	//记录总角点个数
					bTraversed.clear();
					ConnectChessBlocks(startBlock, chessBlocks_forGetGraph, bTraversed, unitPionter, distance_LSL, distance_USL, slopeDiff_USL_connectEdge, lengthDiff_Ratio_USL_connectEdge,
						sumBlockDis, connectionCount);

					curCornerNum_blockLoop = connectionCount;

					if (curCornerNum_blockLoop > maxCornerNum_blockLoop) {
						bestBlockIndex = curStartBlockIndex;
						maxCornerNum_blockLoop = curCornerNum_blockLoop;
						meanCurBlockSize = connectionCount ? (sumBlockDis / (double)connectionCount / sqrt(2)) : 0;
						//meanCurBlockSize = meanCurBlockSize * cos(fabs((angle_axisRotate + angleOffset_axisRotate) / 180.0 * CV_PI)) + meanCurBlockSize * sin(fabs((angle_axisRotate + angleOffset_axisRotate) / 180.0 * CV_PI));
						meanBestBlockSize = meanCurBlockSize;
						foundCnt_blockLoop = 0;
					}
					else if (curCornerNum_blockLoop == maxCornerNum_blockLoop) {
						if (maxCornerNum_blockLoop) {
							foundCnt_blockLoop++;
						}
					}

					bTraversed.clear();
					DeleteChessGraph(startBlock, bTraversed);
					startBlock = NULL;

					if (foundCnt_blockLoop >= 5) {
						break;
					}
				}

				if (foundCnt_blockLoop >= 5) {
					break;
				}
			}

			if (!bOnlyDectBlackBlock) {
				break;
			}
			else {
				if (bestBlockIndex == -1) {
					break;
				}

				if (IsBlackChessBlock(chessBlocks_found[bestBlockIndex], grayImg_threshold)) {
					bReMorph = false;
				}
				else {
					morphIter += morphIter_step;
				}

				if (morphIter > morphIter_max) {
					break;
				}
			}
		}
	}

	if (bestBlockIndex == -1) {
		return NULL;
	}

	if (bOnlyDectBlackBlock) {
		if (!IsBlackChessBlock(chessBlocks_found[bestBlockIndex], grayImg_threshold)) {
			return NULL;
		}
	}

	// 9.根据搜索到的最适结果，创建block位置关系
	startBlock = new MyChessBlock;
	*startBlock = chessBlocks_found[bestBlockIndex];

	std::vector<MyChessBlock> chessBlocks_forGetGraph = chessBlocks_found;	//ConnectChessBlocks递归过程中删除4角连满的block，所以复制一份
	std::vector<const MyChessBlock*> unitPionter(chessBlocks_forGetGraph.size(), NULL);	//记录graph中block的地址
	double sumBlockDis = 0;
	int connectionCount = 0;	//记录总角点个数
	bTraversed.clear();
	ConnectChessBlocks(startBlock, chessBlocks_forGetGraph, bTraversed, unitPionter, distance_LSL, distance_USL, slopeDiff_USL_connectEdge, lengthDiff_Ratio_USL_connectEdge,
		sumBlockDis, connectionCount);
	cornerNum = connectionCount;
	meanBlockSize = connectionCount ? (sumBlockDis / (double)connectionCount / sqrt(2)) : 0;

	// 10.恢复坐标系旋转
	if (curAngle_axisRotate) {
		*curAngle_axisRotate = startBlock->angle_axisRotate;
	}
	bTraversed.clear();
	SetChessGraphAxisRotationAngle(startBlock, 0, bTraversed);

	if (m_bSaveImg) {
		cv::Mat chessGraphImg = cv::Mat::zeros(grayImg.size(), CV_8UC3);
		DrawChessBlocksOnImg(chessGraphImg, chessBlocks_found, cv::Scalar(0, 0, 255), 2);
		cv::line(chessGraphImg, startBlock->centerPoint, startBlock->centerPoint, cv::Scalar(255, 0, 0), 25);
		bTraversed.clear();
		DrawChessGraphOnImg(chessGraphImg, startBlock, bTraversed, cv::Scalar(0, 255, 0), 2, cv::Scalar(0, 255, 255));
		SaveRGBImg(resultFolderPath, "Img_ChessGraph", chessGraphImg, imgWidth, imgHeight);
	}

	return startBlock;
}

ChessBoard_CornerDetection::MyChessBlock* ChessBoard_CornerDetection::GetChessGraph_Auto(cv::Mat grayImg, int& cornerNum, double& meanBlockSize, int imgWidth, int imgHeight,
	std::string resultFolderPath, bool bOnlyDectBlackBlock, double angleOffset_axisRotate, double* curAngle_axisRotate)
{
	// TODO: 识别棋盘格，建立block间关系（全参数搜索检测，二值化及腐蚀膨胀参数全部循环搜索），可指定是否只检测黑块

	int ret = 0;
	std::vector<int> bTraversed;

	MyChessBlock* startBlock = NULL;
	std::vector<MyChessBlock> chessBlocks_found;
	int bestBlockIndex = -1;
	double distance_LSL;
	double distance_USL;
	double slopeDiff_USL_connectEdge = 0.15;
	double lengthDiff_Ratio_USL_connectEdge = 0.25;

	cv::Mat grayImg_threshold;
	cv::Mat grayImg_morph;

	// 0.直方图均衡化
	cv::Mat grayImg_hist;
	cv::equalizeHist(grayImg, grayImg_hist);

	// 1.变更二值化offset参数，循环二值化，同时自适应二值化均值窗大小根据上次找到的block大小变化
	double beforeBestBlockSize = 0;
	double meanBestBlockSize = 0;

	int thresholdOffset_start = 0/*0*/;	//二值化offset参数搜索起始值
	int thresholdOffset_end = 20/*20*/;	//二值化offset参数搜索结束值
	int thresholdOffset_step = 10;	//二值化offset参数搜索步长

	int curCornerNum_thresholdLoop = 0;
	int maxCornerNum_thresholdLoop = 0;
	int bestThresholdOffset = (thresholdOffset_start + thresholdOffset_end) / 2;
	int foundCnt_thresholdLoop = 0;
	bool bEndThresholdOffsetLoop = false;

	if (thresholdOffset_start > thresholdOffset_end) {
		thresholdOffset_end = thresholdOffset_start;
	}

	for (int thresholdOffset = thresholdOffset_start; thresholdOffset <= thresholdOffset_end + thresholdOffset_step; thresholdOffset += thresholdOffset_step)
	{
		if (beforeBestBlockSize == 0 && meanBestBlockSize != 0) {
			beforeBestBlockSize = meanBestBlockSize;
			thresholdOffset = thresholdOffset_start;
		}

		if (thresholdOffset > thresholdOffset_end || foundCnt_thresholdLoop >= 1) {
			if (thresholdOffset_end == thresholdOffset_start) {
				break;
			}
			thresholdOffset = bestThresholdOffset;
			bEndThresholdOffsetLoop = true;
		}

		int thresholdBlockSize = meanBestBlockSize == 0 ? ((int)((double)imgWidth * 0.2) | 1) : ((int)(meanBestBlockSize * 2/*4*/) | 1);
		cv::adaptiveThreshold(grayImg_hist, grayImg_threshold, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, thresholdBlockSize, thresholdOffset);

		if (m_bSaveImg) {
			SaveGrayImg(resultFolderPath, "Img_Threshold", grayImg_threshold, imgWidth, imgHeight);
		}

		// 2.变更膨胀或腐蚀迭代次数参数，循环膨胀或腐蚀
		int morphIter_start = -1/*3*/;	//膨胀或腐蚀迭代次数参数 搜索起始值
		int morphIter_end = 11/*9*/;	//膨胀或腐蚀迭代次数参数 搜索结束值
		int morphIter_step = 2;	//膨胀或腐蚀迭代次数参数 搜索步长

		int curCornerNum_morphLoop = 0;
		int maxCornerNum_morphLoop = 0;
		int bestMorphIter = (morphIter_start + morphIter_end) / 2;
		int foundCnt_morphLoop = 0;
		bool bEndMorphIterLoop = false;

		if (morphIter_start > morphIter_end) {
			morphIter_end = morphIter_start;
		}

		for (int morphIter = morphIter_start; morphIter <= morphIter_end + morphIter_step; morphIter += morphIter_step)
		{
			if (morphIter > morphIter_end || foundCnt_morphLoop >= 3/*1*/) {
				if (morphIter_end == morphIter_start) {
					break;
				}
				morphIter = bestMorphIter;
				bEndMorphIterLoop = true;
			}

			cv::Mat element;
			if (morphIter <= 0) {
				grayImg_morph = grayImg_threshold.clone();
			}
			else {
				cv::dilate(grayImg_threshold, grayImg_morph, element, cv::Point(1, 1), morphIter);	// 膨胀
			}

			if (m_bSaveImg) {
				SaveGrayImg(resultFolderPath, "Img_Morph", grayImg_morph, imgWidth, imgHeight);
			}

			// 3.寻找边界
			//std::vector<cv::Mat> contours(100000);
			//cv::findContours(grayImg_morph, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
			std::vector<cv::Mat> contours;
			findContours2(grayImg_morph, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

#if 0
			for (size_t i = 0; i < contours.size(); i++)
			{
				cv::Rect rect = cv::boundingRect(contours[i]);

				if (rect.height == grayImg.rows - 2)	// 此处排掉占据整个图像大小的边界
				{
					contours.erase(contours.begin() + i);
					i--;
					continue;
				}
				else if (rect.width <= 15 || rect.height <= 15)	// 此处排掉太小的边界
				{
					contours.erase(contours.begin() + i);
					i--;
					continue;
				}
			}
#else
			std::vector<bool> bContoursIsValid(contours.size(), true);

			for (size_t i = 0; i < contours.size(); i++)
			{
				cv::Rect rect = cv::boundingRect(contours[i]);

				if (rect.height == grayImg.rows - 2)	// 此处排掉占据整个图像大小的边界
				{
					bContoursIsValid[i] = false;
				}
				else if (rect.width <= 15 || rect.height <= 15)	// 此处排掉太小的边界
				{
					bContoursIsValid[i] = false;
				}
			}
#endif

			if (m_bSaveImg) {
				cv::Mat contoursImg = cv::Mat::zeros(grayImg.size(), CV_8UC3);
				DrawCountersOnImg(contoursImg, contours, bContoursIsValid, 255, 0, 0);
				SaveRGBImg(resultFolderPath, "Img_Contours", contoursImg, imgWidth, imgHeight);
			}

			// 4.多边形近似（闭合近似）并寻找四边形
			int approxPolygonError = meanBestBlockSize == 0 ? 10/*20*/ : (int)(meanBestBlockSize / 10.0/*5.0*/);
			std::vector<cv::Mat> contours_parallelogram;
			double lengthDiff_Ratio_USL_parallelogram = 0.5;
			double angleDiff_USL_parallelogram = 15.0/*8.0*//*5.0*/;
			for (int i = 0; i < contours.size(); i++) {
				if (bContoursIsValid[i]) {
					cv::Mat temp_contour;
					if (IsParallelogram(contours[i], temp_contour, imgWidth, imgHeight, approxPolygonError, lengthDiff_Ratio_USL_parallelogram, angleDiff_USL_parallelogram)) {
						contours_parallelogram.push_back(temp_contour);
					}
				}
			}
			if (contours_parallelogram.size() == 0) {
				continue;
			}

			if (m_bSaveImg) {
				cv::Mat contoursImg_polygon = cv::Mat::zeros(grayImg.size(), CV_8UC3);
				ApproxPolyAndDrawOnImg(contoursImg_polygon, contours, bContoursIsValid, approxPolygonError, true, cv::Scalar(0, 0, 255), 2);
				SaveRGBImg(resultFolderPath, "Img_ApproxPolygonClose", contoursImg_polygon, imgWidth, imgHeight);
				cv::Mat contoursImg_parallelogram = cv::Mat::zeros(grayImg.size(), CV_8UC3);
				DrawClosePolygonOnImg(contoursImg_parallelogram, contours_parallelogram, cv::Scalar(0, 0, 255), 2);
				SaveRGBImg(resultFolderPath, "Img_Parallelograms", contoursImg_parallelogram, imgWidth, imgHeight);
			}

			// 5.判断四边形在画面中的旋转角度，旋转坐标系
			if (angleOffset_axisRotate != 0.0) {
				for (int i = 0; i < contours_parallelogram.size(); i++) {
					for (int j = 0; j < contours_parallelogram[i].rows; j++) {
						double x_before = contours_parallelogram[i].at<cv::Vec2i>(j, 0)[0];
						double y_before = contours_parallelogram[i].at<cv::Vec2i>(j, 0)[1];
						double x_new, y_new;
						RotateVector(x_before, y_before, 0, 0, -angleOffset_axisRotate, x_new, y_new);
						contours_parallelogram[i].at<cv::Vec2i>(j, 0)[0] = x_new;
						contours_parallelogram[i].at<cv::Vec2i>(j, 0)[1] = y_new;
					}
				}
			}
			double angle_axisRotate = CheckParallelogramsRotationAndRotateAxis(contours_parallelogram);

			// 6.创建block数组
			chessBlocks_found.clear();
			std::vector<double> blocksPos_x, blocksPos_y;
			for (int i = 0; i < contours_parallelogram.size(); i++) {
				MyChessBlock tempBlock = ParallelogramContourToChessBlock(contours_parallelogram[i], i, angle_axisRotate + angleOffset_axisRotate);
				if (tempBlock.bIsValid) {
					chessBlocks_found.push_back(tempBlock);
					blocksPos_x.push_back(tempBlock.centerPoint.x);
					blocksPos_y.push_back(tempBlock.centerPoint.y);
				}
			}
			if (m_bSaveImg) {
				cv::Mat chessBlocksImg = cv::Mat::zeros(grayImg.size(), CV_8UC3);
				DrawChessBlocksOnImg(chessBlocksImg, chessBlocks_found, cv::Scalar(0, 0, 255), 2);
				SaveRGBImg(resultFolderPath, "Img_ChessBlocks", chessBlocksImg, imgWidth, imgHeight);
			}

			if (chessBlocks_found.size() == 0) {
				continue;
			}

			// 7.设置5个起始搜索的点，从离这5个点最近的block开始搜索
			std::sort(blocksPos_x.begin(), blocksPos_x.end());
			std::sort(blocksPos_y.begin(), blocksPos_y.end());

			double startFindPos_X[5];
			double startFindPos_Y[5];
			for (int ptCnt = 0; ptCnt < 5; ptCnt++) {
				int tempIndex_x, tempIndex_y;
				switch (ptCnt) {
				case 0:
				case 3:
					tempIndex_x = (int)((double)blocksPos_x.size() / 4.0); break;
				case 1:
				case 4:
					tempIndex_x = (int)((double)blocksPos_x.size() / 4.0 * 3.0); break;
				default:
					tempIndex_x = (int)((double)blocksPos_x.size() / 2.0); break;
				}
				switch (ptCnt) {
				case 0:
				case 1:
					tempIndex_y = (int)((double)blocksPos_y.size() / 4.0); break;
				case 3:
				case 4:
					tempIndex_y = (int)((double)blocksPos_y.size() / 4.0 * 3.0); break;
				default:
					tempIndex_y = (int)((double)blocksPos_y.size() / 2.0); break;
				}
				startFindPos_X[ptCnt] = blocksPos_x[tempIndex_x];
				startFindPos_Y[ptCnt] = blocksPos_y[tempIndex_y];
			}

			std::vector<std::vector<std::pair<double, int>>> closestChessBlocksIndex(5);
			for (int blockCnt = 0; blockCnt < chessBlocks_found.size(); blockCnt++) {
				for (int ptCnt = 0; ptCnt < 5; ptCnt++) {
					double curDis = GetDistanceOfTwoPoint(startFindPos_X[ptCnt], startFindPos_Y[ptCnt], chessBlocks_found[blockCnt].centerPoint.x, chessBlocks_found[blockCnt].centerPoint.y);
					closestChessBlocksIndex[ptCnt].push_back(std::pair<double, int>(curDis, blockCnt));
				}
			}

			for (int ptCnt = 0; ptCnt < 5; ptCnt++) {
				std::sort(closestChessBlocksIndex[ptCnt].begin(), closestChessBlocksIndex[ptCnt].end());
			}

			// 8.变更起始block，循环搜索，获取block位置关系并找到可相连block数量最大时的最适起始block
			if (morphIter <= 0) {
				distance_LSL = 0;
				distance_USL = 4.0 + approxPolygonError * 2.0;/*2.5;*/
			}
			else {
				distance_LSL = morphIter * 0;
				distance_USL = morphIter * 4.0 + approxPolygonError * 2.0;/*2.5;*/
			}

			double meanCurBlockSize = 0;

			int curCornerNum_blockLoop = 0;
			int maxCornerNum_blockLoop = 0;
			int foundCnt_blockLoop = 0;
			bestBlockIndex = -1;

			for (int blockCnt = 0; blockCnt < chessBlocks_found.size(); blockCnt++)
			{
				for (int ptCnt = 0; ptCnt < 5; ptCnt++)
				{
					startBlock = new MyChessBlock;

					int curStartBlockIndex = closestChessBlocksIndex[ptCnt][blockCnt].second;

					*startBlock = chessBlocks_found[curStartBlockIndex];

					std::vector<MyChessBlock> chessBlocks_forGetGraph = chessBlocks_found;	//ConnectChessBlocks递归过程中删除4角连满的block，所以复制一份
					std::vector<const MyChessBlock*> unitPionter(chessBlocks_forGetGraph.size(), NULL);	//记录graph中block的地址
					double sumBlockDis = 0;
					int connectionCount = 0;	//记录总角点个数
					bTraversed.clear();
					ConnectChessBlocks(startBlock, chessBlocks_forGetGraph, bTraversed, unitPionter, distance_LSL, distance_USL, slopeDiff_USL_connectEdge, lengthDiff_Ratio_USL_connectEdge,
						sumBlockDis, connectionCount);

					curCornerNum_blockLoop = connectionCount;

					if (curCornerNum_blockLoop > maxCornerNum_blockLoop) {
						bestBlockIndex = curStartBlockIndex;
						maxCornerNum_blockLoop = curCornerNum_blockLoop;
						meanCurBlockSize = connectionCount ? (sumBlockDis / (double)connectionCount / sqrt(2)) : 0;
						//meanCurBlockSize = meanCurBlockSize * cos(fabs((angle_axisRotate + angleOffset_axisRotate) / 180.0 * CV_PI)) + meanCurBlockSize * sin(fabs((angle_axisRotate + angleOffset_axisRotate) / 180.0 * CV_PI));
						foundCnt_blockLoop = 0;
					}
					else if (curCornerNum_blockLoop == maxCornerNum_blockLoop) {
						if (maxCornerNum_blockLoop) {
							foundCnt_blockLoop++;
						}
					}

					bTraversed.clear();
					DeleteChessGraph(startBlock, bTraversed);
					startBlock = NULL;

					if (foundCnt_blockLoop >= 5) {
						break;
					}
				}

				if (foundCnt_blockLoop >= 5) {
					break;
				}
			}

			curCornerNum_morphLoop = maxCornerNum_blockLoop;

			if (curCornerNum_morphLoop > maxCornerNum_morphLoop) {
				if (!bOnlyDectBlackBlock) {
					bestMorphIter = morphIter;
					maxCornerNum_morphLoop = curCornerNum_morphLoop;
					meanBestBlockSize = meanCurBlockSize;
					foundCnt_morphLoop = 0;
				}
				else {
					if (bestBlockIndex != -1) {
						if (IsBlackChessBlock(chessBlocks_found[bestBlockIndex], grayImg_threshold)) {
							bestMorphIter = morphIter;
							maxCornerNum_morphLoop = curCornerNum_morphLoop;
							meanBestBlockSize = meanCurBlockSize;
							foundCnt_morphLoop = 0;
						}
					}
				}
			}
			else {
				if (maxCornerNum_morphLoop) {
					if (!bOnlyDectBlackBlock) {
						foundCnt_morphLoop++;
					}
					else {
						if (bestBlockIndex != -1) {
							if (IsBlackChessBlock(chessBlocks_found[bestBlockIndex], grayImg_threshold)) {
								foundCnt_morphLoop++;
							}
						}
					}
				}
			}

			if (bEndMorphIterLoop) {
				break;
			}
		}

		curCornerNum_thresholdLoop = maxCornerNum_morphLoop;

		if (curCornerNum_thresholdLoop > maxCornerNum_thresholdLoop) {
			bestThresholdOffset = thresholdOffset;
			maxCornerNum_thresholdLoop = curCornerNum_thresholdLoop;
			foundCnt_thresholdLoop = 0;
		}
		else {
			if (maxCornerNum_thresholdLoop) {
				foundCnt_thresholdLoop++;
			}
		}

		if (bEndThresholdOffsetLoop) {
			break;
		}
	}

	if (bestBlockIndex == -1) {
		return NULL;
	}

	if (bOnlyDectBlackBlock) {
		if (!IsBlackChessBlock(chessBlocks_found[bestBlockIndex], grayImg_threshold)) {
			return NULL;
		}
	}

	// 9.根据搜索到的最适结果，创建block位置关系
	startBlock = new MyChessBlock;
	*startBlock = chessBlocks_found[bestBlockIndex];

	std::vector<MyChessBlock> chessBlocks_forGetGraph = chessBlocks_found;	//ConnectChessBlocks递归过程中删除4角连满的block，所以复制一份
	std::vector<const MyChessBlock*> unitPionter(chessBlocks_forGetGraph.size(), NULL);	//记录graph中block的地址
	double sumBlockDis = 0;
	int connectionCount = 0;	//记录总角点个数
	bTraversed.clear();
	ConnectChessBlocks(startBlock, chessBlocks_forGetGraph, bTraversed, unitPionter, distance_LSL, distance_USL, slopeDiff_USL_connectEdge, lengthDiff_Ratio_USL_connectEdge,
		sumBlockDis, connectionCount);
	cornerNum = connectionCount;
	meanBlockSize = connectionCount ? (sumBlockDis / (double)connectionCount / sqrt(2)) : 0;

	// 10.恢复坐标系旋转
	if (curAngle_axisRotate) {
		*curAngle_axisRotate = startBlock->angle_axisRotate;
	}
	bTraversed.clear();
	SetChessGraphAxisRotationAngle(startBlock, 0, bTraversed);

	if (m_bSaveImg) {
		cv::Mat chessGraphImg = cv::Mat::zeros(grayImg.size(), CV_8UC3);
		DrawChessBlocksOnImg(chessGraphImg, chessBlocks_found, cv::Scalar(0, 0, 255), 2);
		cv::line(chessGraphImg, startBlock->centerPoint, startBlock->centerPoint, cv::Scalar(255, 0, 0), 25);
		bTraversed.clear();
		DrawChessGraphOnImg(chessGraphImg, startBlock, bTraversed, cv::Scalar(0, 255, 0), 2, cv::Scalar(0, 255, 255));
		SaveRGBImg(resultFolderPath, "Img_ChessGraph", chessGraphImg, imgWidth, imgHeight);
	}

	return startBlock;
}

void ChessBoard_CornerDetection::GetMarkedImage(cv::Mat& markedImg)
{
	if (!markedImg.empty()) {
		markedImg.release();
	}
	markedImg = m_MarkedImage.clone();

	return;
}

double ChessBoard_CornerDetection::GetDistanceOfTwoPoint(double pt1_x, double pt1_y, double pt2_x, double pt2_y)
{
	return sqrt((double)((pt1_x - pt2_x) * (pt1_x - pt2_x) + (pt1_y - pt2_y) * (pt1_y - pt2_y)));
}

double ChessBoard_CornerDetection::GetAngleOfTwoVector(double pt1_x, double pt1_y, double pt2_x, double pt2_y, double center_x, double center_y)
{
	//定义图像上顺时针旋转为正
	/*
		此处定义图像二维坐标系
			  /z
			 /
			/________ x
			|
			|
			|y
	*/
	double theta = atan2(pt2_y - center_y, pt2_x - center_x) - atan2(pt1_y - center_y, pt1_x - center_x);
	theta = theta * 180.0 / CV_PI;
	if (theta <= -180) {
		theta += 360;
	}
	else if (theta > 180) {
		theta -= 360;
	}
	return theta;
}

double ChessBoard_CornerDetection::GetAngleOfTwoVector_0to360(double pt1_x, double pt1_y, double pt2_x, double pt2_y, double center_x, double center_y)
{
	//定义图像上顺时针旋转为正
	/*
		此处定义图像二维坐标系
			  /z
			 /
			/________ x
			|
			|
			|y
	*/
	double theta = atan2(pt2_y - center_y, pt2_x - center_x) - atan2(pt1_y - center_y, pt1_x - center_x);
	theta = theta * 180.0 / CV_PI;
	if (theta < 0) {
		theta = 360 - (-1) * theta;
	}
	return theta;
}

void ChessBoard_CornerDetection::RotateVector(double pt_before_x, double pt_before_y, double center_x, double center_y, double angle, double& pt_after_x, double& pt_after_y)
{
	//定义图像上顺时针旋转为正
	/*
		此处定义图像二维坐标系
			  /z
			 /
			/________ x
			|
			|
			|y
	*/
	double rad = angle / 180.0 * CV_PI;
	pt_after_x = (pt_before_x - center_x) * cos(rad) - (pt_before_y - center_y) * sin(rad) + center_x;
	pt_after_y = (pt_before_x - center_x) * sin(rad) + (pt_before_y - center_y) * cos(rad) + center_y;

	return;
}

double ChessBoard_CornerDetection::GetEdgeSlope(double pt1_x, double pt1_y, double pt2_x, double pt2_y, EdgeType& edgeType)
{
	double slope;

	if (edgeType == EDGE_HORIZONTAL) {
		slope = ((pt2_y - pt1_y) / (pt2_x - pt1_x));
	}
	else if (edgeType == EDGE_VERTICAL) {
		slope = ((pt2_x - pt1_x) / (pt2_y - pt1_y));
	}
	else {
		if (fabs(pt2_y - pt1_y) <= fabs(pt2_x - pt1_x)) {
			slope = ((pt2_y - pt1_y) / (pt2_x - pt1_x));
			edgeType = EDGE_HORIZONTAL;
		}
		else {
			slope = ((pt2_x - pt1_x) / (pt2_y - pt1_y));
			edgeType = EDGE_VERTICAL;
		}
	}

	return slope;
}

int ChessBoard_CornerDetection::SaveGrayImg(std::string folderPath, std::string imgName, cv::Mat grayImg, int width, int height)
{
	cv::imwrite(folderPath + "\\" + imgName + ".bmp", grayImg);

#ifdef SAVE_RAW
	unsigned char* tempBuf_Gray = new unsigned char[width * height];
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			tempBuf_Gray[y * width + x] = grayImg.at<uchar>(y, x);
		}
	}
	std::fstream fout;
	fout.open(folderPath + "\\" + imgName + ".raw", std::ios::out | std::ios::binary);
	fout.write((const char*)tempBuf_Gray, width * height);
	fout.close();

	delete[] tempBuf_Gray;
#endif

	return 0;
}

int ChessBoard_CornerDetection::SaveRGBImg(std::string folderPath, std::string imgName, cv::Mat rgbImg, int width, int height)
{
	cv::imwrite(folderPath + "\\" + imgName + ".bmp", rgbImg);

#ifdef SAVE_RAW
	cv::Mat tempBmp(height, width, CV_8UC1);
	cv::cvtColor(rgbImg, tempBmp, CV_RGB2GRAY);
	unsigned char* tempBuf_Gray = new unsigned char[width * height];
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			tempBuf_Gray[y * width + x] = tempBmp.at<uchar>(y, x);
		}
	}
	std::fstream fout;
	fout.open(folderPath + "\\" + imgName + ".raw", std::ios::out | std::ios::binary);
	fout.write((const char*)tempBuf_Gray, width * height);
	fout.close();

	delete[] tempBuf_Gray;
#endif

	return 0;
}

int ChessBoard_CornerDetection::DrawCountersOnImg(cv::Mat& contoursImg, std::vector<cv::Mat> contours, std::vector<bool> bContoursIsValid, unsigned char colorR, unsigned char colorG, unsigned char colorB)
{
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (bContoursIsValid[i]) {
			for (int j = 0; j < contours[i].rows; j++) {
				int x = contours[i].at<cv::Vec2i>(j, 0)[0];
				int y = contours[i].at<cv::Vec2i>(j, 0)[1];
				contoursImg.at<cv::Vec3b>(y, x)[0] = colorB;
				contoursImg.at<cv::Vec3b>(y, x)[1] = colorG;
				contoursImg.at<cv::Vec3b>(y, x)[2] = colorR;
			}
		}
	}

	return 0;
}

int ChessBoard_CornerDetection::ApproxPolyAndDrawOnImg(cv::Mat& contoursImg_polygon, std::vector<cv::Mat> contours, std::vector<bool> bContoursIsValid, int approxPolygonError, bool bClose, cv::Scalar color, int lineWidth)
{
	for (int i = 0; i < contours.size(); i++) {
		if (bContoursIsValid[i]) {
			cv::Mat contour_polygon;
			cv::approxPolyDP(contours[i], contour_polygon, approxPolygonError, bClose);

			for (size_t j = 0; j < contour_polygon.rows - 1; j++)
			{
				int curNum = j;
				int nextNum = j + 1;
				int cur_x = contour_polygon.at<cv::Vec2i>(curNum, 0)[0];
				int cur_y = contour_polygon.at<cv::Vec2i>(curNum, 0)[1];
				int next_x = contour_polygon.at<cv::Vec2i>(nextNum, 0)[0];
				int next_y = contour_polygon.at<cv::Vec2i>(nextNum, 0)[1];
				cv::Point tempPointCur(cur_x, cur_y);
				cv::Point tempPointNext(next_x, next_y);
				cv::line(contoursImg_polygon, tempPointCur, tempPointNext, color, lineWidth);
			}

			if (bClose) {
				int curNum = contour_polygon.rows - 1;
				int nextNum = 0;
				int cur_x = contour_polygon.at<cv::Vec2i>(curNum, 0)[0];
				int cur_y = contour_polygon.at<cv::Vec2i>(curNum, 0)[1];
				int next_x = contour_polygon.at<cv::Vec2i>(nextNum, 0)[0];
				int next_y = contour_polygon.at<cv::Vec2i>(nextNum, 0)[1];
				cv::Point tempPointCur(cur_x, cur_y);
				cv::Point tempPointNext(next_x, next_y);
				cv::line(contoursImg_polygon, tempPointCur, tempPointNext, color, lineWidth);
			}
		}
	}

	return 0;
}

int ChessBoard_CornerDetection::DrawEdgesOnImg(cv::Mat& edgesImg, std::vector<cv::Vec4i> edges, cv::Scalar color, int lineWidth)
{
	for (size_t i = 0; i < edges.size(); i++)
	{
		cv::line(edgesImg, cv::Point(edges[i][0], edges[i][1]), cv::Point(edges[i][2], edges[i][3]), color, lineWidth);
	}

	return 0;
}

bool ChessBoard_CornerDetection::IsParallelogram(cv::Mat contour, cv::Mat& contour_parallelogram, int roiWidth, int roiHeight, int approxPolygonError, double lengthDiff_Ratio_USL, double angleDiff_USL)
{
	// TODO: 判断是否为近似平行四边形

	if (!contour_parallelogram.empty()) {
		contour_parallelogram.release();
	}

	cv::Mat contour_polygon;

	cv::approxPolyDP(contour, contour_polygon, approxPolygonError, true);

	if (contour_polygon.rows != 4) {
		return false;
	}

	double aveLength = 0;
	std::vector<double> lengths;

	for (int i = 0; i < 4; i++) {
		int curNum = i;
		int beforeNum = (i - 1) < 0 ? (i + 4 - 1) : (i - 1);
		int nextNum = (i + 1) > 3 ? (i - 4 + 1) : (i + 1);
		int nextNum_2 = (nextNum + 1) > 3 ? (nextNum - 4 + 1) : (nextNum + 1);
		int cur_x = contour_polygon.at<cv::Vec2i>(curNum, 0)[0];
		int cur_y = contour_polygon.at<cv::Vec2i>(curNum, 0)[1];
		int before_x = contour_polygon.at<cv::Vec2i>(beforeNum, 0)[0];
		int before_y = contour_polygon.at<cv::Vec2i>(beforeNum, 0)[1];
		int next_x = contour_polygon.at<cv::Vec2i>(nextNum, 0)[0];
		int next_y = contour_polygon.at<cv::Vec2i>(nextNum, 0)[1];
		int next_x_2 = contour_polygon.at<cv::Vec2i>(nextNum_2, 0)[0];
		int next_y_2 = contour_polygon.at<cv::Vec2i>(nextNum_2, 0)[1];

		double curAngle = GetAngleOfTwoVector(before_x, before_y, next_x, next_y, cur_x, cur_y);
		double curAngle_2 = GetAngleOfTwoVector(cur_x, cur_y, next_x_2, next_y_2, next_x, next_y);
		double curLength = GetDistanceOfTwoPoint(before_x, before_y, cur_x, cur_y);
		double oppositeLength = GetDistanceOfTwoPoint(next_x, next_y, next_x_2, next_y_2);

		if (fabs((curLength - oppositeLength) / std::min(curLength, oppositeLength)) > lengthDiff_Ratio_USL) {
			return false;
		}

		double angleSum = curAngle + curAngle_2;

		if (curAngle >= 0 && curAngle_2 >= 0) {
			if (angleSum < 180 - angleDiff_USL || angleSum > 180 + angleDiff_USL) {
				return false;
			}
		}
		else if (curAngle <= 0 && curAngle_2 <= 0) {
			if (angleSum < -180 - angleDiff_USL || angleSum > -180 + angleDiff_USL) {
				return false;
			}
		}
		else {
			return false;
		}
	}

	contour_parallelogram = contour_polygon.clone();
	return true;
}

double ChessBoard_CornerDetection::CheckParallelogramsRotationAndRotateAxis(std::vector<cv::Mat>& contours_parallelogram)
{
	// TODO: 判断四边形在画面中的旋转角度，旋转坐标系，使得四边形四个角分布位置尽可能为 LT,RT,RB,LB
	/*
		  /\        ____
		 /  \      |    |
		/    \	-> |    |
		\    /     |    |
		 \  /      |____|
		  \/
	*/

	//记录四边形的边在-45~45度范围内的分布情况（坐标系最多旋转45度，deg超过45等价于反向旋转90-deg，考虑透视，H和V边都计算）
	//std::vector<int> edgeDeg_H_Count(91, 0);	//仅记录边数量
	//std::vector<int> edgeDeg_V_Count(91, 0);	//仅记录边数量
	std::vector<double> edgeDeg_H_LengthSum(91, 0);	//边越长，比重越大
	std::vector<double> edgeDeg_V_LengthSum(91, 0);	//边越长，比重越大
	for (int i = 0; i < contours_parallelogram.size(); i++) {
		for (int j = 0; j < contours_parallelogram[i].rows; j++) {
			int curCornerIndex = j;
			int nextCornerIndex = (j == contours_parallelogram[i].rows - 1) ? 0 : j + 1;

			double vecAxisX_x, vecAxisX_y, vecAxisY_x, vecAxisY_y, vecPoint_x, vecPoint_y, cen_x, cen_y;
			if (contours_parallelogram[i].at<cv::Vec2i>(curCornerIndex, 0)[0] < contours_parallelogram[i].at<cv::Vec2i>(nextCornerIndex, 0)[0]) {
				cen_x = contours_parallelogram[i].at<cv::Vec2i>(curCornerIndex, 0)[0];
				cen_y = contours_parallelogram[i].at<cv::Vec2i>(curCornerIndex, 0)[1];
				vecPoint_x = contours_parallelogram[i].at<cv::Vec2i>(nextCornerIndex, 0)[0];
				vecPoint_y = contours_parallelogram[i].at<cv::Vec2i>(nextCornerIndex, 0)[1];
			}
			else {
				cen_x = contours_parallelogram[i].at<cv::Vec2i>(nextCornerIndex, 0)[0];
				cen_y = contours_parallelogram[i].at<cv::Vec2i>(nextCornerIndex, 0)[1];
				vecPoint_x = contours_parallelogram[i].at<cv::Vec2i>(curCornerIndex, 0)[0];
				vecPoint_y = contours_parallelogram[i].at<cv::Vec2i>(curCornerIndex, 0)[1];
			}
			vecAxisX_x = cen_x + 1;
			vecAxisX_y = cen_y;
			int deg_H = GetAngleOfTwoVector(vecAxisX_x, vecAxisX_y, vecPoint_x, vecPoint_y, cen_x, cen_y) + 0.5;
			if (deg_H >= -45 && deg_H <= 45) {
				//edgeDeg_H_Count[deg + 45]++;
				edgeDeg_H_LengthSum[deg_H + 45] += GetDistanceOfTwoPoint(vecPoint_x, vecPoint_y, cen_x, cen_y);
			}

			if (contours_parallelogram[i].at<cv::Vec2i>(curCornerIndex, 0)[1] < contours_parallelogram[i].at<cv::Vec2i>(nextCornerIndex, 0)[1]) {
				cen_x = contours_parallelogram[i].at<cv::Vec2i>(curCornerIndex, 0)[0];
				cen_y = contours_parallelogram[i].at<cv::Vec2i>(curCornerIndex, 0)[1];
				vecPoint_x = contours_parallelogram[i].at<cv::Vec2i>(nextCornerIndex, 0)[0];
				vecPoint_y = contours_parallelogram[i].at<cv::Vec2i>(nextCornerIndex, 0)[1];
			}
			else {
				cen_x = contours_parallelogram[i].at<cv::Vec2i>(nextCornerIndex, 0)[0];
				cen_y = contours_parallelogram[i].at<cv::Vec2i>(nextCornerIndex, 0)[1];
				vecPoint_x = contours_parallelogram[i].at<cv::Vec2i>(curCornerIndex, 0)[0];
				vecPoint_y = contours_parallelogram[i].at<cv::Vec2i>(curCornerIndex, 0)[1];
			}
			vecAxisY_x = cen_x;
			vecAxisY_y = cen_y + 1;
			int deg_V = GetAngleOfTwoVector(vecAxisY_x, vecAxisY_y, vecPoint_x, vecPoint_y, cen_x, cen_y) + 0.5;
			if (deg_V >= -45 && deg_V <= 45) {
				//edgeDeg_V_Count[deg + 45]++;
				edgeDeg_V_LengthSum[deg_V + 45] += GetDistanceOfTwoPoint(vecPoint_x, vecPoint_y, cen_x, cen_y);
			}
		}
	}

	//计算主要角度分布
	//int maxCnt_H = 0;
	//int maxCnt_V = 0;
	double maxLengthSum_H = 0;
	double maxLengthSum_V = 0;
	int meanDeg_H = 0;
	int meanDeg_V = 0;
	for (int deg = -45; deg <= 45; deg++) {
		//if (blockDegCount_H[deg + 45] > maxCnt_H) {
		//	meanDeg_H = deg;
		//	maxCnt_H = edgeDeg_H_Count[deg + 45];
		//}
		if (edgeDeg_H_LengthSum[deg + 45] > maxLengthSum_H) {
			meanDeg_H = deg;
			maxLengthSum_H = edgeDeg_H_LengthSum[deg + 45];
		}

		//if (blockDegCount_V[deg + 45] > maxCnt_V) {
		//	meanDeg_V = deg;
		//	maxCnt_V = edgeDeg_V_Count[deg + 45];
		//}
		if (edgeDeg_V_LengthSum[deg + 45] > maxLengthSum_V) {
			meanDeg_V = deg;
			maxLengthSum_V = edgeDeg_V_LengthSum[deg + 45];
		}
	}

	//旋转坐标系（坐标反向旋转）
	double angle_axisRotate = maxLengthSum_H > maxLengthSum_V ? meanDeg_H : meanDeg_V;
	for (int i = 0; i < contours_parallelogram.size(); i++) {
		for (int j = 0; j < contours_parallelogram[i].rows; j++) {
			double x_before = contours_parallelogram[i].at<cv::Vec2i>(j, 0)[0];
			double y_before = contours_parallelogram[i].at<cv::Vec2i>(j, 0)[1];
			double x_new, y_new;
			RotateVector(x_before, y_before, 0, 0, -angle_axisRotate, x_new, y_new);
			contours_parallelogram[i].at<cv::Vec2i>(j, 0)[0] = x_new;
			contours_parallelogram[i].at<cv::Vec2i>(j, 0)[1] = y_new;
		}
	}

	return angle_axisRotate;
}

int ChessBoard_CornerDetection::DrawClosePolygonOnImg(cv::Mat& contoursImg_closePolygon, std::vector<cv::Mat> contours_closePolygon, cv::Scalar color, int lineWidth)
{
	for (size_t i = 0; i < contours_closePolygon.size(); i++)
	{
		for (int j = 0; j < contours_closePolygon[i].rows; j++) {
			int curNum = j;
			int nextNum = j + 1 >= contours_closePolygon[i].rows ? j + 1 - contours_closePolygon[i].rows : j + 1;
			int cur_x = contours_closePolygon[i].at<cv::Vec2i>(curNum, 0)[0];
			int cur_y = contours_closePolygon[i].at<cv::Vec2i>(curNum, 0)[1];
			int next_x = contours_closePolygon[i].at<cv::Vec2i>(nextNum, 0)[0];
			int next_y = contours_closePolygon[i].at<cv::Vec2i>(nextNum, 0)[1];
			cv::Point tempPointCur(cur_x, cur_y);
			cv::Point tempPointNext(next_x, next_y);
			cv::line(contoursImg_closePolygon, tempPointCur, tempPointNext, color, lineWidth);
		}
	}

	return 0;
}

ChessBoard_CornerDetection::MyChessBlock ChessBoard_CornerDetection::ParallelogramContourToChessBlock(cv::Mat contour_parallelogram, int blockIndex, double angle_axisRotate)
{
	// TODO: 生成 block 对象

	MyChessBlock chessBlock;

	if (contour_parallelogram.rows != 4) {
		return chessBlock;
	}

	double cen_X = 0, cen_Y = 0;
	for (int j = 0; j < 4; j++) {
		cen_X += contour_parallelogram.at<cv::Vec2i>(j, 0)[0];
		cen_Y += contour_parallelogram.at<cv::Vec2i>(j, 0)[1];
	}
	cen_X /= 4.0;
	cen_Y /= 4.0;

	chessBlock.centerPoint.x = cen_X;
	chessBlock.centerPoint.y = cen_Y;
	std::vector<std::pair<double, int>> cornerPos;
	for (int j = 0; j < 4; j++) {
		double corner_X = contour_parallelogram.at<cv::Vec2i>(j, 0)[0];
		double corner_Y = contour_parallelogram.at<cv::Vec2i>(j, 0)[1];
		double cenLeft_X = cen_X - 1;
		double cenLeft_Y = cen_Y;

		cornerPos.push_back(std::pair<double, int>(GetAngleOfTwoVector_0to360(cenLeft_X, cenLeft_Y, corner_X, corner_Y, cen_X, cen_Y), j));
	}
	std::sort(cornerPos.begin(), cornerPos.end());
	for (int j = 0; j < 4; j++) {
		int index = cornerPos[j].second;
		chessBlock.cornerPoint[j].x = contour_parallelogram.at<cv::Vec2i>(index, 0)[0];
		chessBlock.cornerPoint[j].y = contour_parallelogram.at<cv::Vec2i>(index, 0)[1];
	}

	EdgeType edgeType;
	edgeType = EDGE_HORIZONTAL;
	chessBlock.edgeSlope_H[0] = GetEdgeSlope(chessBlock.cornerPoint[0].x, chessBlock.cornerPoint[0].y, chessBlock.cornerPoint[1].x, chessBlock.cornerPoint[1].y, edgeType);
	chessBlock.edgeSlope_H[1] = GetEdgeSlope(chessBlock.cornerPoint[2].x, chessBlock.cornerPoint[2].y, chessBlock.cornerPoint[3].x, chessBlock.cornerPoint[3].y, edgeType);
	edgeType = EDGE_VERTICAL;
	chessBlock.edgeSlope_V[0] = GetEdgeSlope(chessBlock.cornerPoint[3].x, chessBlock.cornerPoint[3].y, chessBlock.cornerPoint[0].x, chessBlock.cornerPoint[0].y, edgeType);
	chessBlock.edgeSlope_V[1] = GetEdgeSlope(chessBlock.cornerPoint[1].x, chessBlock.cornerPoint[1].y, chessBlock.cornerPoint[2].x, chessBlock.cornerPoint[2].y, edgeType);

	chessBlock.edgeLength_H[0] = GetDistanceOfTwoPoint(chessBlock.cornerPoint[0].x, chessBlock.cornerPoint[0].y, chessBlock.cornerPoint[1].x, chessBlock.cornerPoint[1].y);
	chessBlock.edgeLength_H[1] = GetDistanceOfTwoPoint(chessBlock.cornerPoint[2].x, chessBlock.cornerPoint[2].y, chessBlock.cornerPoint[3].x, chessBlock.cornerPoint[3].y);
	chessBlock.edgeLength_V[0] = GetDistanceOfTwoPoint(chessBlock.cornerPoint[3].x, chessBlock.cornerPoint[3].y, chessBlock.cornerPoint[0].x, chessBlock.cornerPoint[0].y);
	chessBlock.edgeLength_V[1] = GetDistanceOfTwoPoint(chessBlock.cornerPoint[1].x, chessBlock.cornerPoint[1].y, chessBlock.cornerPoint[2].x, chessBlock.cornerPoint[2].y);

	chessBlock.bIsValid = true;
	chessBlock.blockIndex = blockIndex;

	chessBlock.angle_axisRotate = angle_axisRotate;

	return chessBlock;
}

bool ChessBoard_CornerDetection::IsBlackChessBlock(MyChessBlock chessblock, const cv::Mat& grayImg_threshold)
{
	// TODO: 判断是不是黑色 block

	cv::Point2d blockCenterPoint_beforeRotate;
	cv::Point2d blockCornerPoint_beforeRotate[4];
	if (chessblock.angle_axisRotate != 0.0) {
		RotateVector(chessblock.centerPoint.x, chessblock.centerPoint.y, 0, 0, chessblock.angle_axisRotate,
			blockCenterPoint_beforeRotate.x, blockCenterPoint_beforeRotate.y);
		for (int j = 0; j < 4; j++) {
			RotateVector(chessblock.cornerPoint[j].x, chessblock.cornerPoint[j].y, 0, 0, chessblock.angle_axisRotate,
				blockCornerPoint_beforeRotate[j].x, blockCornerPoint_beforeRotate[j].y);
		}
	}
	else {
		blockCenterPoint_beforeRotate = chessblock.centerPoint;
		for (int j = 0; j < 4; j++) {
			blockCornerPoint_beforeRotate[j] = chessblock.cornerPoint[j];
		}
	}

	int checkPoint_x[4] = { 0 };
	int checkPoint_y[4] = { 0 };

	int blackPointCnt = 0;
	for (int j = 0; j < 4; j++) {
		checkPoint_x[j] = (blockCornerPoint_beforeRotate[j].x - blockCenterPoint_beforeRotate.x) * 4.0 / 5.0 + blockCenterPoint_beforeRotate.x + 0.5;
		checkPoint_y[j] = (blockCornerPoint_beforeRotate[j].y - blockCenterPoint_beforeRotate.y) * 4.0 / 5.0 + blockCenterPoint_beforeRotate.y + 0.5;
		if (grayImg_threshold.at<uchar>(checkPoint_y[j], checkPoint_x[j]) == 0) {
			blackPointCnt++;
		}
	}

	if (blackPointCnt >= 3) {
		return true;
	}

	return false;
}

int ChessBoard_CornerDetection::ConnectChessBlocks(MyChessBlock* startBlock, std::vector<MyChessBlock>& chessBlocks, std::vector<int>& bTraversed,
	std::vector<const MyChessBlock*>& unitPionter, double distance_LSL, double distance_USL, double slopeDiff_USL, double lengthDiff_Ratio_USL,
	double& sumBlockDis, int& connectionCount)
{
	// TODO: 判断 block 是否相邻，创建 graph（申请内存）

	// block 太多时容易 stack overflow...

	//已遍历过的直接退出
	if (find(bTraversed.begin(), bTraversed.end(), startBlock->blockIndex) != bTraversed.end()) {
		return 0;
	}
	bTraversed.push_back(startBlock->blockIndex);

	double minDis_0 = FLT_MAX;
	double minDis_1 = FLT_MAX;
	double minDis_2 = FLT_MAX;
	double minDis_3 = FLT_MAX;
	int minDisIndex_0 = -1;
	int minDisIndex_1 = -1;
	int minDisIndex_2 = -1;
	int minDisIndex_3 = -1;
	int curBlockIndex = -1;

	bool bFound_0 = false;
	bool bFound_1 = false;
	bool bFound_2 = false;
	bool bFound_3 = false;
	bool bFound_Cur = false;

	MyChessBlock foundBlock_0, foundBlock_1, foundBlock_2, foundBlock_3;

	//判断相对位置用，左边的中心
	double pt_3clk_x = (startBlock->cornerPoint[0].x + startBlock->cornerPoint[3].x) / 2.0;
	double pt_3clk_y = (startBlock->cornerPoint[0].y + startBlock->cornerPoint[3].y) / 2.0;

	//寻找四个方向最近的block
	for (int i = 0; i < chessBlocks.size(); i++) {
		if (startBlock->blockIndex == chessBlocks[i].blockIndex) {
			curBlockIndex = i;
			bFound_Cur = true;
			//跳过自己，只记录index
		}
		else {
			double curDis = GetDistanceOfTwoPoint(startBlock->centerPoint.x, startBlock->centerPoint.y, chessBlocks[i].centerPoint.x, chessBlocks[i].centerPoint.y);
			double curBlockPos = GetAngleOfTwoVector_0to360(pt_3clk_x, pt_3clk_y, chessBlocks[i].centerPoint.x, chessBlocks[i].centerPoint.y, startBlock->centerPoint.x, startBlock->centerPoint.y);

			//找距离最近的，并判断相对位置
			if (curBlockPos >= 0 && curBlockPos < 90) {
				if ((!startBlock->bConnected[0]) && (!chessBlocks[i].bConnected[2])) {
					if (curDis < minDis_0)
					{
						//判断是否可以相连
						//1.判断找到的角点是否在当前block之外（合理性）
						//2.判断找到的角点与当前角点的角度关系（合理性）
						//3.判断角点相距的x/y距离：distance_LSL，distance_USL
						//4.判断角点两侧的边 斜率/长度 是否一致：slopeThreshold，lengthDiffThreshold_Ratio
						double centerToCurCorner = GetDistanceOfTwoPoint(startBlock->centerPoint.x, startBlock->centerPoint.y, startBlock->cornerPoint[0].x, startBlock->cornerPoint[0].y);
						double centerToFoundConner = GetDistanceOfTwoPoint(startBlock->centerPoint.x, startBlock->centerPoint.y, chessBlocks[i].cornerPoint[2].x, chessBlocks[i].cornerPoint[2].y);
						double foundCornerPos = GetAngleOfTwoVector_0to360(pt_3clk_x, pt_3clk_y, chessBlocks[i].cornerPoint[2].x, chessBlocks[i].cornerPoint[2].y, startBlock->centerPoint.x, startBlock->centerPoint.y);
						double cornerDis_X = fabs(startBlock->cornerPoint[0].x - chessBlocks[i].cornerPoint[2].x);
						double cornerDis_Y = fabs(startBlock->cornerPoint[0].y - chessBlocks[i].cornerPoint[2].y);
						double edgeSlopeDiff_H = fabs(startBlock->edgeSlope_H[0] - chessBlocks[i].edgeSlope_H[1]);
						double edgeSlopeDiff_V = fabs(startBlock->edgeSlope_V[0] - chessBlocks[i].edgeSlope_V[1]);
						double lengthDiff_H = fabs(startBlock->edgeLength_H[0] - chessBlocks[i].edgeLength_H[1]) / std::min(startBlock->edgeLength_H[0], chessBlocks[i].edgeLength_H[1]);
						double lengthDiff_V = fabs(startBlock->edgeLength_V[0] - chessBlocks[i].edgeLength_V[1]) / std::min(startBlock->edgeLength_V[0], chessBlocks[i].edgeLength_V[1]);

						if (centerToCurCorner < centerToFoundConner &&
							foundCornerPos >= 0 && foundCornerPos < 90 &&
							cornerDis_X >= distance_LSL && cornerDis_X <= distance_USL &&
							cornerDis_Y >= distance_LSL && cornerDis_Y <= distance_USL &&
							edgeSlopeDiff_H <= slopeDiff_USL && edgeSlopeDiff_V <= slopeDiff_USL &&
							lengthDiff_H <= lengthDiff_Ratio_USL && lengthDiff_V <= lengthDiff_Ratio_USL)
						{
							minDis_0 = curDis;
							minDisIndex_0 = i;
							foundBlock_0 = chessBlocks[i];
						}
					}
				}
			}
			else if (curBlockPos < 180) {
				if ((!startBlock->bConnected[1]) && (!chessBlocks[i].bConnected[3])) {
					if (curDis < minDis_1)
					{
						//判断是否可以相连
						//1.判断找到的角点是否在当前block之外（合理性）
						//2.判断找到的角点与当前角点的角度关系（合理性）
						//3.判断角点相距的x/y距离：distance_LSL，distance_USL
						//4.判断角点两侧的边 斜率/长度 是否一致：slopeThreshold，lengthDiffThreshold_Ratio
						double centerToCurCorner = GetDistanceOfTwoPoint(startBlock->centerPoint.x, startBlock->centerPoint.y, startBlock->cornerPoint[1].x, startBlock->cornerPoint[1].y);
						double centerToFoundConner = GetDistanceOfTwoPoint(startBlock->centerPoint.x, startBlock->centerPoint.y, chessBlocks[i].cornerPoint[3].x, chessBlocks[i].cornerPoint[3].y);
						double foundCornerPos = GetAngleOfTwoVector_0to360(pt_3clk_x, pt_3clk_y, chessBlocks[i].cornerPoint[3].x, chessBlocks[i].cornerPoint[3].y, startBlock->centerPoint.x, startBlock->centerPoint.y);
						double cornerDis_X = fabs(startBlock->cornerPoint[1].x - chessBlocks[i].cornerPoint[3].x);
						double cornerDis_Y = fabs(startBlock->cornerPoint[1].y - chessBlocks[i].cornerPoint[3].y);
						double edgeSlopeDiff_H = fabs(startBlock->edgeSlope_H[0] - chessBlocks[i].edgeSlope_H[1]);
						double edgeSlopeDiff_V = fabs(startBlock->edgeSlope_V[1] - chessBlocks[i].edgeSlope_V[0]);
						double lengthDiff_H = fabs(startBlock->edgeLength_H[0] - chessBlocks[i].edgeLength_H[1]) / std::min(startBlock->edgeLength_H[0], chessBlocks[i].edgeLength_H[1]);
						double lengthDiff_V = fabs(startBlock->edgeLength_V[1] - chessBlocks[i].edgeLength_V[0]) / std::min(startBlock->edgeLength_V[1], chessBlocks[i].edgeLength_V[0]);

						if (centerToCurCorner < centerToFoundConner &&
							foundCornerPos >= 90 && foundCornerPos < 180 &&
							cornerDis_X >= distance_LSL && cornerDis_X <= distance_USL &&
							cornerDis_Y >= distance_LSL && cornerDis_Y <= distance_USL &&
							edgeSlopeDiff_H <= slopeDiff_USL && edgeSlopeDiff_V <= slopeDiff_USL &&
							lengthDiff_H <= lengthDiff_Ratio_USL && lengthDiff_V <= lengthDiff_Ratio_USL)
						{
							minDis_1 = curDis;
							minDisIndex_1 = i;
							foundBlock_1 = chessBlocks[i];
						}
					}
				}
			}
			else if (curBlockPos < 270) {
				if ((!startBlock->bConnected[2]) && (!chessBlocks[i].bConnected[0])) {
					if (curDis < minDis_2)
					{
						//判断是否可以相连
						//1.判断找到的角点是否在当前block之外（合理性）
						//2.判断找到的角点与当前角点的角度关系（合理性）
						//3.判断角点相距的x/y距离：distance_LSL，distance_USL
						//4.判断角点两侧的边 斜率/长度 是否一致：slopeThreshold，lengthDiffThreshold_Ratio
						double centerToCurCorner = GetDistanceOfTwoPoint(startBlock->centerPoint.x, startBlock->centerPoint.y, startBlock->cornerPoint[2].x, startBlock->cornerPoint[2].y);
						double centerToFoundConner = GetDistanceOfTwoPoint(startBlock->centerPoint.x, startBlock->centerPoint.y, chessBlocks[i].cornerPoint[0].x, chessBlocks[i].cornerPoint[0].y);
						double foundCornerPos = GetAngleOfTwoVector_0to360(pt_3clk_x, pt_3clk_y, chessBlocks[i].cornerPoint[0].x, chessBlocks[i].cornerPoint[0].y, startBlock->centerPoint.x, startBlock->centerPoint.y);
						double cornerDis_X = fabs(startBlock->cornerPoint[2].x - chessBlocks[i].cornerPoint[0].x);
						double cornerDis_Y = fabs(startBlock->cornerPoint[2].y - chessBlocks[i].cornerPoint[0].y);
						double edgeSlopeDiff_H = fabs(startBlock->edgeSlope_H[1] - chessBlocks[i].edgeSlope_H[0]);
						double edgeSlopeDiff_V = fabs(startBlock->edgeSlope_V[1] - chessBlocks[i].edgeSlope_V[0]);
						double lengthDiff_H = fabs(startBlock->edgeLength_H[1] - chessBlocks[i].edgeLength_H[0]) / std::min(startBlock->edgeLength_H[1], chessBlocks[i].edgeLength_H[0]);
						double lengthDiff_V = fabs(startBlock->edgeLength_V[1] - chessBlocks[i].edgeLength_V[0]) / std::min(startBlock->edgeLength_V[1], chessBlocks[i].edgeLength_V[0]);

						if (centerToCurCorner < centerToFoundConner &&
							foundCornerPos >= 180 && foundCornerPos < 270 &&
							cornerDis_X >= distance_LSL && cornerDis_X <= distance_USL &&
							cornerDis_Y >= distance_LSL && cornerDis_Y <= distance_USL &&
							edgeSlopeDiff_H <= slopeDiff_USL && edgeSlopeDiff_V <= slopeDiff_USL &&
							lengthDiff_H <= lengthDiff_Ratio_USL && lengthDiff_V <= lengthDiff_Ratio_USL)
						{
							minDis_2 = curDis;
							minDisIndex_2 = i;
							foundBlock_2 = chessBlocks[i];
						}
					}
				}
			}
			else {
				if ((!startBlock->bConnected[3]) && (!chessBlocks[i].bConnected[1])) {
					if (curDis < minDis_3)
					{
						//判断是否可以相连
						//1.判断找到的角点是否在当前block之外（合理性）
						//2.判断找到的角点与当前角点的角度关系（合理性）
						//3.判断角点相距的x/y距离：distance_LSL，distance_USL
						//4.判断角点两侧的边 斜率/长度 是否一致：slopeThreshold，lengthDiffThreshold_Ratio
						double centerToCurCorner = GetDistanceOfTwoPoint(startBlock->centerPoint.x, startBlock->centerPoint.y, startBlock->cornerPoint[3].x, startBlock->cornerPoint[3].y);
						double centerToFoundConner = GetDistanceOfTwoPoint(startBlock->centerPoint.x, startBlock->centerPoint.y, chessBlocks[i].cornerPoint[1].x, chessBlocks[i].cornerPoint[1].y);
						double foundCornerPos = GetAngleOfTwoVector_0to360(pt_3clk_x, pt_3clk_y, chessBlocks[i].cornerPoint[1].x, chessBlocks[i].cornerPoint[1].y, startBlock->centerPoint.x, startBlock->centerPoint.y);
						double cornerDis_X = fabs(startBlock->cornerPoint[3].x - chessBlocks[i].cornerPoint[1].x);
						double cornerDis_Y = fabs(startBlock->cornerPoint[3].y - chessBlocks[i].cornerPoint[1].y);
						double edgeSlopeDiff_H = fabs(startBlock->edgeSlope_H[1] - chessBlocks[i].edgeSlope_H[0]);
						double edgeSlopeDiff_V = fabs(startBlock->edgeSlope_V[0] - chessBlocks[i].edgeSlope_V[1]);
						double lengthDiff_H = fabs(startBlock->edgeLength_H[1] - chessBlocks[i].edgeLength_H[0]) / std::min(startBlock->edgeLength_H[1], chessBlocks[i].edgeLength_H[0]);
						double lengthDiff_V = fabs(startBlock->edgeLength_V[0] - chessBlocks[i].edgeLength_V[1]) / std::min(startBlock->edgeLength_V[0], chessBlocks[i].edgeLength_V[1]);

						if (centerToCurCorner < centerToFoundConner &&
							foundCornerPos >= 270 && foundCornerPos < 360 &&
							cornerDis_X >= distance_LSL && cornerDis_X <= distance_USL &&
							cornerDis_Y >= distance_LSL && cornerDis_Y <= distance_USL &&
							edgeSlopeDiff_H <= slopeDiff_USL && edgeSlopeDiff_V <= slopeDiff_USL &&
							lengthDiff_H <= lengthDiff_Ratio_USL && lengthDiff_V <= lengthDiff_Ratio_USL)
						{
							minDis_3 = curDis;
							minDisIndex_3 = i;
							foundBlock_3 = chessBlocks[i];
						}
					}
				}
			}
		}
	}

	//连接已找到的block
	if (foundBlock_0.bIsValid) {
		//std::cout << cornerDis_X << ", " << cornerDis_Y << std::endl;
		if (unitPionter[minDisIndex_0] == NULL) {	//找到的block未存在于graph内的情况，第一次new
			startBlock->connection_out[0] = new MyChessBlock;
			*startBlock->connection_out[0] = foundBlock_0;
			unitPionter[minDisIndex_0] = startBlock->connection_out[0];
		}
		else {	//找到的block已经存在于graph内的情况，已经new过
			startBlock->connection_out[0] = (MyChessBlock*)unitPionter[minDisIndex_0];
		}

		startBlock->bConnected[0] = true;
		startBlock->connection_out[0]->bConnected[2] = true;
		chessBlocks[curBlockIndex].bConnected[0] = true;
		chessBlocks[minDisIndex_0].bConnected[2] = true;
		startBlock->connection_out[0]->inCount++;
		chessBlocks[minDisIndex_0].inCount++;
		bFound_0 = true;

		sumBlockDis += GetDistanceOfTwoPoint(startBlock->centerPoint.x, startBlock->centerPoint.y, foundBlock_0.centerPoint.x, foundBlock_0.centerPoint.y);
		connectionCount++;
	}
	if (foundBlock_1.bIsValid) {
		//std::cout << cornerDis_X << ", " << cornerDis_Y << std::endl;
		if (unitPionter[minDisIndex_1] == NULL) {	//找到的block未存在于graph内的情况，第一次new
			startBlock->connection_out[1] = new MyChessBlock;
			*startBlock->connection_out[1] = foundBlock_1;
			unitPionter[minDisIndex_1] = startBlock->connection_out[1];
		}
		else {	//找到的block已经存在于graph内的情况，已经new过
			startBlock->connection_out[1] = (MyChessBlock*)unitPionter[minDisIndex_1];
		}

		startBlock->bConnected[1] = true;
		startBlock->connection_out[1]->bConnected[3] = true;
		chessBlocks[curBlockIndex].bConnected[1] = true;
		chessBlocks[minDisIndex_1].bConnected[3] = true;
		startBlock->connection_out[1]->inCount++;
		chessBlocks[minDisIndex_1].inCount++;
		bFound_1 = true;

		sumBlockDis += GetDistanceOfTwoPoint(startBlock->centerPoint.x, startBlock->centerPoint.y, foundBlock_1.centerPoint.x, foundBlock_1.centerPoint.y);
		connectionCount++;
	}
	if (foundBlock_2.bIsValid) {
		//std::cout << cornerDis_X << ", " << cornerDis_Y << std::endl;
		if (unitPionter[minDisIndex_2] == NULL) {	//找到的block未存在于graph内的情况，第一次new
			startBlock->connection_out[2] = new MyChessBlock;
			*startBlock->connection_out[2] = foundBlock_2;
			unitPionter[minDisIndex_2] = startBlock->connection_out[2];
		}
		else {	//找到的block已经存在于graph内的情况，已经new过
			startBlock->connection_out[2] = (MyChessBlock*)unitPionter[minDisIndex_2];
		}

		startBlock->bConnected[2] = true;
		startBlock->connection_out[2]->bConnected[0] = true;
		chessBlocks[curBlockIndex].bConnected[2] = true;
		chessBlocks[minDisIndex_2].bConnected[0] = true;
		startBlock->connection_out[2]->inCount++;
		chessBlocks[minDisIndex_2].inCount++;
		bFound_2 = true;

		sumBlockDis += GetDistanceOfTwoPoint(startBlock->centerPoint.x, startBlock->centerPoint.y, foundBlock_2.centerPoint.x, foundBlock_2.centerPoint.y);
		connectionCount++;
	}
	if (foundBlock_3.bIsValid) {
		//std::cout << cornerDis_X << ", " << cornerDis_Y << std::endl;
		if (unitPionter[minDisIndex_3] == NULL) {	//找到的block未存在于graph内的情况，第一次new
			startBlock->connection_out[3] = new MyChessBlock;
			*startBlock->connection_out[3] = foundBlock_3;
			unitPionter[minDisIndex_3] = startBlock->connection_out[3];
		}
		else {	//找到的block已经存在于graph内的情况，已经new过
			startBlock->connection_out[3] = (MyChessBlock*)unitPionter[minDisIndex_3];
		}

		startBlock->bConnected[3] = true;
		startBlock->connection_out[3]->bConnected[1] = true;
		chessBlocks[curBlockIndex].bConnected[3] = true;
		chessBlocks[minDisIndex_3].bConnected[1] = true;
		startBlock->connection_out[3]->inCount++;
		chessBlocks[minDisIndex_3].inCount++;
		bFound_3 = true;

		sumBlockDis += GetDistanceOfTwoPoint(startBlock->centerPoint.x, startBlock->centerPoint.y, foundBlock_3.centerPoint.x, foundBlock_3.centerPoint.y);
		connectionCount++;
	}

	//已连接4个的block无需遍历，删除
	std::vector<int> blockIndex_forDelete;
	if (bFound_0) {
		blockIndex_forDelete.push_back(minDisIndex_0);
	}
	if (bFound_1) {
		blockIndex_forDelete.push_back(minDisIndex_1);
	}
	if (bFound_2) {
		blockIndex_forDelete.push_back(minDisIndex_2);
	}
	if (bFound_3) {
		blockIndex_forDelete.push_back(minDisIndex_3);
	}
	if (bFound_Cur) {
		blockIndex_forDelete.push_back(curBlockIndex);
	}
	std::sort(blockIndex_forDelete.begin(), blockIndex_forDelete.end());
	for (int i = blockIndex_forDelete.size() - 1; i >= 0; i--) {
		int cnt = 0;
		for (int j = 0; j < 4; j++) {
			if (chessBlocks[blockIndex_forDelete[i]].bConnected[j] == true) {
				cnt++;
			}
		}
		if (cnt == 4) {
			chessBlocks.erase(chessBlocks.begin() + blockIndex_forDelete[i]);
			unitPionter.erase(unitPionter.begin() + blockIndex_forDelete[i]);
		}
	}

	if (bFound_0) {
		ConnectChessBlocks(startBlock->connection_out[0], chessBlocks, bTraversed, unitPionter, distance_LSL, distance_USL, slopeDiff_USL, lengthDiff_Ratio_USL,
			sumBlockDis, connectionCount);
	}
	if (bFound_1) {
		ConnectChessBlocks(startBlock->connection_out[1], chessBlocks, bTraversed, unitPionter, distance_LSL, distance_USL, slopeDiff_USL, lengthDiff_Ratio_USL,
			sumBlockDis, connectionCount);
	}
	if (bFound_2) {
		ConnectChessBlocks(startBlock->connection_out[2], chessBlocks, bTraversed, unitPionter, distance_LSL, distance_USL, slopeDiff_USL, lengthDiff_Ratio_USL,
			sumBlockDis, connectionCount);
	}
	if (bFound_3) {
		ConnectChessBlocks(startBlock->connection_out[3], chessBlocks, bTraversed, unitPionter, distance_LSL, distance_USL, slopeDiff_USL, lengthDiff_Ratio_USL,
			sumBlockDis, connectionCount);
	}

	return 0;
}

void ChessBoard_CornerDetection::DeleteChessGraph(MyChessBlock* startBlock, std::vector<int>& bTraversed)
{
	// TODO: 释放 graph 内存

	//已遍历过的block，无需再进行深度递归
	if (find(bTraversed.begin(), bTraversed.end(), startBlock->blockIndex) != bTraversed.end()) {
		//判断该block踩过几次，inCount小于等于1直接删除，反之inCount--
		if (startBlock->inCount <= 1) {
			delete startBlock;
			//startBlock->bIsValid = false;
			//std::cout << startBlock->blockIndex << "\t" << startBlock->bIsValid << std::endl;
		}
		else {
			startBlock->inCount--;
		}
		return;
	}
	bTraversed.push_back(startBlock->blockIndex);

	for (int i = 0; i < 4; i++) {
		if (startBlock->connection_out[i]) {
			DeleteChessGraph(startBlock->connection_out[i], bTraversed);
			startBlock->connection_out[i] = NULL;
		}
	}

	//判断该block踩过几次，inCount小于等于1直接删除，反之inCount--
	if (startBlock->inCount <= 1) {
		delete startBlock;
		//startBlock->bIsValid = false;
		//std::cout << startBlock->blockIndex << "\t" << startBlock->bIsValid << std::endl;
	}
	else {
		startBlock->inCount--;
	}

	return;
}

int ChessBoard_CornerDetection::TraverseChessGraph(MyChessBlock* startBlock, std::vector<int>& bTraversed)
{
	// TODO: 遍历 block graph

	//已遍历过的直接退出
	if (find(bTraversed.begin(), bTraversed.end(), startBlock->blockIndex) != bTraversed.end()) {
		return 0;
	}
	bTraversed.push_back(startBlock->blockIndex);

	for (int i = 0; i < 4; i++) {
		if (startBlock->connection_out[i]) {
			TraverseChessGraph(startBlock->connection_out[i], bTraversed);
		}
	}

	return 0;
}

int ChessBoard_CornerDetection::SetChessGraphAxisRotationAngle(MyChessBlock* startBlock, double angle_axisRotate_new, std::vector<int>& bTraversed)
{
	// TODO: 变更 block graph 中 block 坐标的坐标系旋转，更新 block 内坐标

	//已遍历过的直接退出
	if (find(bTraversed.begin(), bTraversed.end(), startBlock->blockIndex) != bTraversed.end()) {
		return 0;
	}
	bTraversed.push_back(startBlock->blockIndex);

	cv::Point2d centerPoint_new;
	cv::Point2d cornerPoint_new[4];
	double angle = startBlock->angle_axisRotate - angle_axisRotate_new;
	RotateVector(startBlock->centerPoint.x, startBlock->centerPoint.y, 0, 0, angle, centerPoint_new.x, centerPoint_new.y);
	startBlock->centerPoint = centerPoint_new;
	for (int j = 0; j < 4; j++) {
		RotateVector(startBlock->cornerPoint[j].x, startBlock->cornerPoint[j].y, 0, 0, angle, cornerPoint_new[j].x, cornerPoint_new[j].y);
		startBlock->cornerPoint[j] = cornerPoint_new[j];
	}

	for (int i = 0; i < 4; i++) {
		if (startBlock->connection_out[i]) {
			SetChessGraphAxisRotationAngle(startBlock->connection_out[i], angle_axisRotate_new, bTraversed);
		}
	}

	return 0;
}

ChessBoard_CornerDetection::MyChessBlock* ChessBoard_CornerDetection::FindClosestBlockInChessGraph(MyChessBlock* startBlock, cv::Point2d point, std::vector<int>& bTraversed, double& minDis)
{
	// TODO: 在 graph 中找到离指定点最近的 block

	//已遍历过的直接退出
	if (find(bTraversed.begin(), bTraversed.end(), startBlock->blockIndex) != bTraversed.end()) {
		return 0;
	}
	bTraversed.push_back(startBlock->blockIndex);

	MyChessBlock* foundBlock[4] = { NULL };

	for (int i = 0; i < 4; i++) {
		if (startBlock->connection_out[i]) {
			foundBlock[i] = FindClosestBlockInChessGraph(startBlock->connection_out[i], point, bTraversed, minDis);
		}
	}

	int closestIndex = -1;
	double minDis_foundInNeighbor = FLT_MAX;

	for (int i = 0; i < 4; i++) {
		if (foundBlock[i]) {
			double curNeighborDis = GetDistanceOfTwoPoint(foundBlock[i]->centerPoint.x, foundBlock[i]->centerPoint.y, point.x, point.y);
			if (curNeighborDis < minDis_foundInNeighbor) {
				closestIndex = i;
				minDis_foundInNeighbor = curNeighborDis;
			}
		}
	}

	double curblockDis = GetDistanceOfTwoPoint(startBlock->centerPoint.x, startBlock->centerPoint.y, point.x, point.y);

	if (minDis_foundInNeighbor > minDis && curblockDis >= minDis) {
		return NULL;
	}

	if (closestIndex != -1) {
		if (curblockDis <= minDis_foundInNeighbor) {
			minDis = curblockDis;
			return startBlock;
		}
		else {
			minDis = minDis_foundInNeighbor;
			return foundBlock[closestIndex];
		}
	}

	return startBlock;
}

bool ChessBoard_CornerDetection::ChangeChessGraphStartBlock_WithBlockIndex(MyChessBlock* startBlock, int blockIndex, std::vector<int>& bTraversed,
	MyChessBlock** startBlock_new, int fromWhichCorner, MyChessBlock* beforeBlock)
{
	// TODO: 变更 graph 原点至指定 Index 的 block

	MyChessBlock* temp_new = startBlock;

	if (beforeBlock != NULL) {
		if (startBlock->blockIndex == blockIndex) {	//找到指定block后开始返回
			beforeBlock->connection_out[fromWhichCorner] = NULL;
			beforeBlock->inCount++;
			startBlock->connection_out[fromWhichCorner + 2 > 3 ? fromWhichCorner - 2 : fromWhichCorner + 2] = beforeBlock;
			startBlock->inCount--;
			for (int i = 0; i < 4; i++) {	//找到目标 block 后也需要进行深度递归，可能路线中存在绕一圈且仅有一条路线连到了自己的情况
				if (startBlock->connection_out[i]) {
					ChangeChessGraphStartBlock_WithBlockIndex(startBlock->connection_out[i], blockIndex, bTraversed, startBlock_new, i, startBlock);
				}
			}
			*startBlock_new = startBlock;
			return true;
		}
	}

	//已遍历过的直接退出
	if (find(bTraversed.begin(), bTraversed.end(), startBlock->blockIndex) != bTraversed.end()) {
		return false;
	}
	bTraversed.push_back(startBlock->blockIndex);

	bool bChanged = false;
	for (int i = 0; i < 4; i++) {
		if (startBlock->connection_out[i]) {
			if (beforeBlock == NULL) {
				if (ChangeChessGraphStartBlock_WithBlockIndex(startBlock->connection_out[i], blockIndex, bTraversed, &temp_new, i, startBlock)) {
					bChanged = true;
				}
			}
			else {
				if (ChangeChessGraphStartBlock_WithBlockIndex(startBlock->connection_out[i], blockIndex, bTraversed, startBlock_new, i, startBlock)) {
					bChanged = true;
				}
			}
		}
	}

	if (bChanged) {
		if (beforeBlock == NULL) {
			*startBlock_new = temp_new;
		}
		else {	//如果递归返回结果为路径需要变更，则变更本block与上一个block的关系
			if (beforeBlock->blockIndex != blockIndex) {	//如果上一个block为目标block则不改变路径，即 上面提到的找到目标block后深度遍历时，存在绕一圈且仅有一条路线连到了自己的情况
				beforeBlock->connection_out[fromWhichCorner] = NULL;
				beforeBlock->inCount++;
				startBlock->connection_out[fromWhichCorner + 2 > 3 ? fromWhichCorner - 2 : fromWhichCorner + 2] = beforeBlock;
				startBlock->inCount--;
			}
		}
	}

	return bChanged;
}

int ChessBoard_CornerDetection::GetChessGraphConnerPixelAndIndexes(MyChessBlock* startBlock, std::vector<int>& bTraversed, cv::Mat grayImg,
	std::vector<cv::Point2f>& cornerPoints, bool bGetSubPixel, int subPixelWinSize, std::vector<cv::Point2i>& cornerIndexes, int curIndex_X, int curIndex_Y)
{
	// TODO: 根据 block 关系获取角点坐标，可选择是否精确化角点坐标

	//已遍历过的直接退出
	if (find(bTraversed.begin(), bTraversed.end(), startBlock->blockIndex) != bTraversed.end()) {
		return 0;
	}
	bTraversed.push_back(startBlock->blockIndex);

	if (startBlock->connection_out[0]) {
		double newCorner_X;
		double newCorner_Y;

		if (bGetSubPixel) {
			newCorner_X = (startBlock->cornerPoint[0].x + startBlock->connection_out[0]->cornerPoint[2].x) / 2.0;
			newCorner_Y = (startBlock->cornerPoint[0].y + startBlock->connection_out[0]->cornerPoint[2].y) / 2.0;
			cv::Point2f newCorner(newCorner_X, newCorner_Y);
			std::vector<cv::Point2f> newCornerArr;
			newCornerArr.push_back(newCorner);
			cv::Size winSize = cv::Size(subPixelWinSize, subPixelWinSize);
			cv::Size zerozone = cv::Size(-1, -1);
			cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.001);
			cv::cornerSubPix(grayImg, newCornerArr, winSize, zerozone, criteria);
			newCorner_X = newCornerArr[0].x;
			newCorner_Y = newCornerArr[0].y;
			startBlock->cornerPoint[0].x = startBlock->connection_out[0]->cornerPoint[2].x = newCorner_X;
			startBlock->cornerPoint[0].y = startBlock->connection_out[0]->cornerPoint[2].y = newCorner_Y;
		}
		else {
			newCorner_X = startBlock->cornerPoint[0].x;
			newCorner_Y = startBlock->cornerPoint[0].y;
		}

		cornerPoints.push_back(cv::Point2f(newCorner_X, newCorner_Y));
		cornerIndexes.push_back(cv::Point2i(curIndex_X, curIndex_Y));
		GetChessGraphConnerPixelAndIndexes(startBlock->connection_out[0], bTraversed, grayImg, cornerPoints, bGetSubPixel, subPixelWinSize, cornerIndexes, curIndex_X - 1, curIndex_Y - 1);
	}
	if (startBlock->connection_out[1]) {
		double newCorner_X;
		double newCorner_Y;

		if (bGetSubPixel) {
			newCorner_X = (startBlock->cornerPoint[1].x + startBlock->connection_out[1]->cornerPoint[3].x) / 2.0;
			newCorner_Y = (startBlock->cornerPoint[1].y + startBlock->connection_out[1]->cornerPoint[3].y) / 2.0;
			cv::Point2f newCorner(newCorner_X, newCorner_Y);
			std::vector<cv::Point2f> newCornerArr;
			newCornerArr.push_back(newCorner);
			cv::Size winSize = cv::Size(subPixelWinSize, subPixelWinSize);
			cv::Size zerozone = cv::Size(-1, -1);
			cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.001);
			cv::cornerSubPix(grayImg, newCornerArr, winSize, zerozone, criteria);
			newCorner_X = newCornerArr[0].x;
			newCorner_Y = newCornerArr[0].y;
			startBlock->cornerPoint[1].x = startBlock->connection_out[1]->cornerPoint[3].x = newCorner_X;
			startBlock->cornerPoint[1].y = startBlock->connection_out[1]->cornerPoint[3].y = newCorner_Y;
		}
		else {
			newCorner_X = startBlock->cornerPoint[1].x;
			newCorner_Y = startBlock->cornerPoint[1].y;
		}

		cornerPoints.push_back(cv::Point2f(newCorner_X, newCorner_Y));
		cornerIndexes.push_back(cv::Point2i(curIndex_X + 1, curIndex_Y));
		GetChessGraphConnerPixelAndIndexes(startBlock->connection_out[1], bTraversed, grayImg, cornerPoints, bGetSubPixel, subPixelWinSize, cornerIndexes, curIndex_X + 1, curIndex_Y - 1);
	}
	if (startBlock->connection_out[2]) {
		double newCorner_X;
		double newCorner_Y;

		if (bGetSubPixel) {
			newCorner_X = (startBlock->cornerPoint[2].x + startBlock->connection_out[2]->cornerPoint[0].x) / 2.0;
			newCorner_Y = (startBlock->cornerPoint[2].y + startBlock->connection_out[2]->cornerPoint[0].y) / 2.0;
			cv::Point2f newCorner(newCorner_X, newCorner_Y);
			std::vector<cv::Point2f> newCornerArr;
			newCornerArr.push_back(newCorner);
			cv::Size winSize = cv::Size(subPixelWinSize, subPixelWinSize);
			cv::Size zerozone = cv::Size(-1, -1);
			cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.001);
			cv::cornerSubPix(grayImg, newCornerArr, winSize, zerozone, criteria);
			newCorner_X = newCornerArr[0].x;
			newCorner_Y = newCornerArr[0].y;
			startBlock->cornerPoint[2].x = startBlock->connection_out[2]->cornerPoint[0].x = newCorner_X;
			startBlock->cornerPoint[2].y = startBlock->connection_out[2]->cornerPoint[0].y = newCorner_Y;
		}
		else {
			newCorner_X = startBlock->cornerPoint[2].x;
			newCorner_Y = startBlock->cornerPoint[2].y;
		}

		cornerPoints.push_back(cv::Point2f(newCorner_X, newCorner_Y));
		cornerIndexes.push_back(cv::Point2i(curIndex_X + 1, curIndex_Y + 1));
		GetChessGraphConnerPixelAndIndexes(startBlock->connection_out[2], bTraversed, grayImg, cornerPoints, bGetSubPixel, subPixelWinSize, cornerIndexes, curIndex_X + 1, curIndex_Y + 1);
	}
	if (startBlock->connection_out[3]) {
		double newCorner_X;
		double newCorner_Y;

		if (bGetSubPixel) {
			newCorner_X = (startBlock->cornerPoint[3].x + startBlock->connection_out[3]->cornerPoint[1].x) / 2.0;
			newCorner_Y = (startBlock->cornerPoint[3].y + startBlock->connection_out[3]->cornerPoint[1].y) / 2.0;
			cv::Point2f newCorner(newCorner_X, newCorner_Y);
			std::vector<cv::Point2f> newCornerArr;
			newCornerArr.push_back(newCorner);
			cv::Size winSize = cv::Size(subPixelWinSize, subPixelWinSize);
			cv::Size zerozone = cv::Size(-1, -1);
			cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.001);
			cv::cornerSubPix(grayImg, newCornerArr, winSize, zerozone, criteria);
			newCorner_X = newCornerArr[0].x;
			newCorner_Y = newCornerArr[0].y;
			startBlock->cornerPoint[3].x = startBlock->connection_out[3]->cornerPoint[1].x = newCorner_X;
			startBlock->cornerPoint[3].y = startBlock->connection_out[3]->cornerPoint[1].y = newCorner_Y;
		}
		else {
			newCorner_X = startBlock->cornerPoint[3].x;
			newCorner_Y = startBlock->cornerPoint[3].y;
		}

		cornerPoints.push_back(cv::Point2f(newCorner_X, newCorner_Y));
		cornerIndexes.push_back(cv::Point2i(curIndex_X, curIndex_Y + 1));
		GetChessGraphConnerPixelAndIndexes(startBlock->connection_out[3], bTraversed, grayImg, cornerPoints, bGetSubPixel, subPixelWinSize, cornerIndexes, curIndex_X - 1, curIndex_Y + 1);
	}

	return 0;
}

int ChessBoard_CornerDetection::GetChessGraphBlockPositionAndIndexes(MyChessBlock* startBlock, std::vector<int>& bTraversed,
	std::vector<cv::Point2f>& blockPositions, std::vector<cv::Point2i>& blockIndexes, int curIndex_X, int curIndex_Y)
{
	// TODO: 根据 block 关系获取 block 位置坐标

	//已遍历过的直接退出
	if (find(bTraversed.begin(), bTraversed.end(), startBlock->blockIndex) != bTraversed.end()) {
		return 0;
	}
	bTraversed.push_back(startBlock->blockIndex);

	blockPositions.push_back(cv::Point2f(startBlock->centerPoint.x, startBlock->centerPoint.y));
	blockIndexes.push_back(cv::Point2i(curIndex_X, curIndex_Y));

	if (startBlock->connection_out[0]) {
		GetChessGraphBlockPositionAndIndexes(startBlock->connection_out[0], bTraversed, blockPositions, blockIndexes, curIndex_X - 1, curIndex_Y - 1);
	}
	if (startBlock->connection_out[1]) {
		GetChessGraphBlockPositionAndIndexes(startBlock->connection_out[1], bTraversed, blockPositions, blockIndexes, curIndex_X + 1, curIndex_Y - 1);
	}
	if (startBlock->connection_out[2]) {
		GetChessGraphBlockPositionAndIndexes(startBlock->connection_out[2], bTraversed, blockPositions, blockIndexes, curIndex_X + 1, curIndex_Y + 1);
	}
	if (startBlock->connection_out[3]) {
		GetChessGraphBlockPositionAndIndexes(startBlock->connection_out[3], bTraversed, blockPositions, blockIndexes, curIndex_X - 1, curIndex_Y + 1);
	}

	return 0;
}

int ChessBoard_CornerDetection::DrawChessBlocksOnImg(cv::Mat& chessBlocksImg, std::vector<MyChessBlock> chessBlocks, cv::Scalar color, int lineWidth)
{
	for (size_t i = 0; i < chessBlocks.size(); i++)
	{
		cv::Point2d centerPoint_beforeRotate;
		cv::Point2d cornerPoint_beforeRotate[4];
		RotateVector(chessBlocks[i].centerPoint.x, chessBlocks[i].centerPoint.y, 0, 0, chessBlocks[i].angle_axisRotate,
			centerPoint_beforeRotate.x, centerPoint_beforeRotate.y);
		for (int j = 0; j < 4; j++) {
			RotateVector(chessBlocks[i].cornerPoint[j].x, chessBlocks[i].cornerPoint[j].y, 0, 0, chessBlocks[i].angle_axisRotate,
				cornerPoint_beforeRotate[j].x, cornerPoint_beforeRotate[j].y);
		}

		for (int j = 0; j < 4; j++) {
			int curNum = j;
			int nextNum = j + 1 >= 4 ? j + 1 - 4 : j + 1;
			int cur_x = cornerPoint_beforeRotate[curNum].x;
			int cur_y = cornerPoint_beforeRotate[curNum].y;
			int next_x = cornerPoint_beforeRotate[nextNum].x;
			int next_y = cornerPoint_beforeRotate[nextNum].y;
			cv::Point tempPointCur(cur_x, cur_y);
			cv::Point tempPointNext(next_x, next_y);
			cv::line(chessBlocksImg, tempPointCur, tempPointNext, color, lineWidth);
		}

		//标block序号
		cv::Point2d textPoint((centerPoint_beforeRotate.x + cornerPoint_beforeRotate[0].x) / 2.0, (centerPoint_beforeRotate.y + cornerPoint_beforeRotate[0].y) / 2.0);
		cv::putText(chessBlocksImg, std::to_string(chessBlocks[i].blockIndex), textPoint, CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(100, 100, 100), 2);
	}

	return 0;
}

int ChessBoard_CornerDetection::DrawChessGraphOnImg(cv::Mat& chessGraphImg, MyChessBlock* startBlock, std::vector<int>& bTraversed, cv::Scalar lineColor, int lineWidth, cv::Scalar connetCornerColor)
{
	//已遍历过的直接退出
	if (find(bTraversed.begin(), bTraversed.end(), startBlock->blockIndex) != bTraversed.end()) {
		return 0;
	}
	bTraversed.push_back(startBlock->blockIndex);

	//标block序号
	//cv::Point2d textPoint(startBlock->centerPoint.x, (startBlock->centerPoint.y + (startBlock->cornerPoint[2].y + startBlock->cornerPoint[3].y) / 2.0) / 2.0);
	cv::Point2d textPoint((startBlock->centerPoint.x + startBlock->cornerPoint[0].x) / 2.0, (startBlock->centerPoint.y + startBlock->cornerPoint[0].y) / 2.0);
	cv::putText(chessGraphImg, std::to_string(startBlock->blockIndex), textPoint, CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

	for (int i = 0; i < 4; i++) {

		if (startBlock->connection_out[i]) {
			cv::line(chessGraphImg, startBlock->centerPoint, startBlock->connection_out[i]->centerPoint, lineColor, lineWidth);

			//画箭头
			double arrowPoint_x = (startBlock->connection_out[i]->centerPoint.x - startBlock->centerPoint.x) * 1.0 / 4.0 + startBlock->centerPoint.x;
			double arrowPoint_y = (startBlock->connection_out[i]->centerPoint.y - startBlock->centerPoint.y) * 1.0 / 4.0 + startBlock->centerPoint.y;
			double arrowHeadBase_x = (startBlock->connection_out[i]->centerPoint.x - startBlock->centerPoint.x) * 1.0 / 8.0 + startBlock->centerPoint.x;
			double arrowHeadBase_y = (startBlock->connection_out[i]->centerPoint.y - startBlock->centerPoint.y) * 1.0 / 8.0 + startBlock->centerPoint.y;
			double arrowHead1_x, arrowHead1_y, arrowHead2_x, arrowHead2_y;
			RotateVector(arrowHeadBase_x, arrowHeadBase_y, arrowPoint_x, arrowPoint_y, -30, arrowHead1_x, arrowHead1_y);
			RotateVector(arrowHeadBase_x, arrowHeadBase_y, arrowPoint_x, arrowPoint_y, 30, arrowHead2_x, arrowHead2_y);
			cv::line(chessGraphImg, cv::Point2d(arrowPoint_x, arrowPoint_y), cv::Point2d(arrowHead1_x, arrowHead1_y), lineColor, lineWidth);
			cv::line(chessGraphImg, cv::Point2d(arrowPoint_x, arrowPoint_y), cv::Point2d(arrowHead2_x, arrowHead2_y), lineColor, lineWidth);

			//画角点
			cv::line(chessGraphImg, startBlock->cornerPoint[i], startBlock->cornerPoint[i], connetCornerColor, 20);

			DrawChessGraphOnImg(chessGraphImg, startBlock->connection_out[i], bTraversed, lineColor, lineWidth, connetCornerColor);
		}
	}

	return 0;
}

void ChessBoard_CornerDetection::findContours2(const cv::Mat image, std::vector<cv::Mat>& contours, std::vector<cv::Vec4i>& hierarchy, int mode, int method, cv::Point offset)
{
	cv::Mat image2 = image.clone();
#if CV_VERSION_REVISION <= 6
	CvMat c_image = image2;
#else
	CvMat c_image;
	c_image = cvMat(image2.rows, image2.cols, image2.type(), image2.data);
	c_image.step = image2.step[0];
	c_image.type = (c_image.type & ~cv::Mat::CONTINUOUS_FLAG) | (image2.flags & cv::Mat::CONTINUOUS_FLAG);
#endif
	cv::MemStorage storage(cvCreateMemStorage());
	CvSeq* _ccontours = nullptr;

#if CV_VERSION_REVISION <= 6
	cvFindContours(&c_image, storage, &_ccontours, sizeof(CvContour), mode, method, CvPoint(offset));
#else
	cvFindContours(&c_image, storage, &_ccontours, sizeof(CvContour), mode, method, CvPoint{ offset.x, offset.y });
#endif
	if (!_ccontours)
	{
		contours.clear();
		return;
	}
	cv::Seq<CvSeq*> all_contours(cvTreeToNodeSeq(_ccontours, sizeof(CvSeq), storage));
	size_t total = all_contours.size();
	contours.resize(total);

	cv::SeqIterator<CvSeq*> it = all_contours.begin();
	for (size_t i = 0; i < total; i++, ++it)
	{
		CvSeq* c = *it;
		reinterpret_cast<CvContour*>(c)->color = static_cast<int>(i);
		int count = c->total;
		int* data = new int[static_cast<size_t>(count * 2)];
		cvCvtSeqToArray(c, data);
		for (int j = 0; j < count; j++)
			contours[i].push_back(cv::Point(data[j * 2], data[j * 2 + 1]));
		delete[] data;
	}

	hierarchy.resize(total);
	it = all_contours.begin();
	for (size_t i = 0; i < total; i++, ++it)
	{
		CvSeq* c = *it;
		int h_next = c->h_next ? reinterpret_cast<CvContour*>(c->h_next)->color : -1;
		int h_prev = c->h_prev ? reinterpret_cast<CvContour*>(c->h_prev)->color : -1;
		int v_next = c->v_next ? reinterpret_cast<CvContour*>(c->v_next)->color : -1;
		int v_prev = c->v_prev ? reinterpret_cast<CvContour*>(c->v_prev)->color : -1;
		hierarchy[i] = cv::Vec4i(h_next, h_prev, v_next, v_prev);
	}

	storage.release();
}

void ChessBoard_CornerDetection::findContours2(const cv::Mat image, std::vector<cv::Mat>& contours, int mode, int method, cv::Point offset)
{
	cv::Mat image2 = image.clone();
#if CV_VERSION_REVISION <= 6
	CvMat c_image = image2;
#else
	CvMat c_image;
	c_image = cvMat(image2.rows, image2.cols, image2.type(), image2.data);
	c_image.step = image2.step[0];
	c_image.type = (c_image.type & ~cv::Mat::CONTINUOUS_FLAG) | (image2.flags & cv::Mat::CONTINUOUS_FLAG);
#endif
	cv::MemStorage storage(cvCreateMemStorage());
	CvSeq* _ccontours = nullptr;

#if CV_VERSION_REVISION <= 6
	cvFindContours(&c_image, storage, &_ccontours, sizeof(CvContour), mode, method, CvPoint(offset));
#else
	cvFindContours(&c_image, storage, &_ccontours, sizeof(CvContour), mode, method, CvPoint{ offset.x, offset.y });
#endif
	if (!_ccontours)
	{
		contours.clear();
		return;
	}
	cv::Seq<CvSeq*> all_contours(cvTreeToNodeSeq(_ccontours, sizeof(CvSeq), storage));
	size_t total = all_contours.size();
	contours.resize(total);

	cv::SeqIterator<CvSeq*> it = all_contours.begin();
	for (size_t i = 0; i < total; i++, ++it)
	{
		CvSeq* c = *it;
		reinterpret_cast<CvContour*>(c)->color = static_cast<int>(i);
		int count = c->total;
		int* data = new int[static_cast<size_t>(count * 2)];
		cvCvtSeqToArray(c, data);
		for (int j = 0; j < count; j++)
			contours[i].push_back(cv::Point(data[j * 2], data[j * 2 + 1]));
		delete[] data;
	}

	storage.release();
}

void ChessBoard_CornerDetection::findContours2(const cv::Mat image, std::vector<std::vector<cv::Point>>& contours, int mode, int method, cv::Point offset) {
	cv::Mat image2 = image.clone();
#if CV_VERSION_REVISION <= 6
	CvMat c_image = image2;
#else
	CvMat c_image;
	c_image = cvMat(image2.rows, image2.cols, image2.type(), image2.data);
	c_image.step = image2.step[0];
	c_image.type = (c_image.type & ~cv::Mat::CONTINUOUS_FLAG) | (image2.flags & cv::Mat::CONTINUOUS_FLAG);
#endif
	cv::MemStorage storage(cvCreateMemStorage());
	CvSeq* _ccontours = nullptr;

#if CV_VERSION_REVISION <= 6
	cvFindContours(&c_image, storage, &_ccontours, sizeof(CvContour), mode, method, CvPoint(offset));
#else
	cvFindContours(&c_image, storage, &_ccontours, sizeof(CvContour), mode, method, CvPoint{ offset.x, offset.y });
#endif
	if (!_ccontours) {
		contours.clear();
		return;
	}
	cv::Seq<CvSeq*> all_contours(cvTreeToNodeSeq(_ccontours, sizeof(CvSeq), storage));
	size_t total = all_contours.size();
	contours.resize(total);

	cv::SeqIterator<CvSeq*> it = all_contours.begin();
	for (size_t i = 0; i < total; i++, ++it) {
		CvSeq* c = *it;
		reinterpret_cast<CvContour*>(c)->color = static_cast<int>(i);
		int count = c->total;
		int* data = new int[static_cast<size_t>(count * 2)];
		cvCvtSeqToArray(c, data);
		for (int j = 0; j < count; j++)
			contours[i].push_back(cv::Point(data[j * 2], data[j * 2 + 1]));
		delete[] data;
	}

	storage.release();
}

int ChessBoard_CornerDetection::ProcessGetChessSfrEdgeImages_4Dir(std::vector<cv::Mat>& edge_img_h_w2b, std::vector<cv::Mat>& edge_img_h_b2w,
	std::vector<cv::Mat>& edge_img_v_w2b, std::vector<cv::Mat>& edge_img_v_b2w,
	SfrEdgeDectOption sfr_edge_dect_option,
	std::vector<cv::Rect>* edge_pos_h_w2b, std::vector<cv::Rect>* edge_pos_h_b2w,
	std::vector<cv::Rect>* edge_pos_v_w2b, std::vector<cv::Rect>* edge_pos_v_b2w) {

	if (m_GrayImage.empty()) {
		if (m_OutputLog) {
			m_OutputLog("Empty Image");
		}
		return -1;
	}

	if (!edge_img_h_w2b.empty()) {
		edge_img_h_w2b.clear();
	}
	if (!edge_img_h_b2w.empty()) {
		edge_img_h_b2w.clear();
	}
	if (!edge_img_v_w2b.empty()) {
		edge_img_v_w2b.clear();
	}
	if (!edge_img_v_b2w.empty()) {
		edge_img_v_b2w.clear();
	}
	if (edge_pos_h_w2b){
		if (!edge_pos_h_w2b->empty()) {
			edge_pos_h_w2b->clear();
		}
	}
	if (edge_pos_h_b2w){
		if (!edge_pos_h_b2w->empty()) {
			edge_pos_h_b2w->clear();
		}
	}
	if (edge_pos_v_w2b){
		if (!edge_pos_v_w2b->empty()) {
			edge_pos_v_w2b->clear();
		}
	}
	if (edge_pos_v_b2w){
		if (!edge_pos_v_b2w->empty()) {
			edge_pos_v_b2w->clear();
		}
	}

	int ret = 0;
	int block_size = 0;

	for (int point_cnt = 0; point_cnt < sfr_edge_dect_option.dect_pos.size(); point_cnt++) {
		double cur_field = sfr_edge_dect_option.dect_pos[point_cnt].first;
		double cur_degree = sfr_edge_dect_option.dect_pos[point_cnt].second;
		if (cur_field < 0 || cur_field >= 1) {
			if (m_OutputLog) {
				m_OutputLog("Invalid Field Position: " + std::to_string(cur_field));
			}
			edge_img_h_w2b.push_back(cv::Mat());
			if (edge_pos_h_w2b)	edge_pos_h_w2b->push_back(cv::Rect(-1, -1, -1, -1));
			edge_img_h_b2w.push_back(cv::Mat());
			if (edge_pos_h_b2w)	edge_pos_h_b2w->push_back(cv::Rect(-1, -1, -1, -1));
			edge_img_v_w2b.push_back(cv::Mat());
			if (edge_pos_v_w2b)	edge_pos_v_w2b->push_back(cv::Rect(-1, -1, -1, -1));
			edge_img_v_b2w.push_back(cv::Mat());
			if (edge_pos_v_b2w)	edge_pos_v_b2w->push_back(cv::Rect(-1, -1, -1, -1));
			continue;
		}
		while (cur_degree < 0) {
			cur_degree += 360;
		}
		while (cur_degree >= 360) {
			cur_degree -= 360;
		}

		cv::Point point;
		double radius = (double)sqrt(m_ImgWidth * m_ImgWidth + m_ImgHeight * m_ImgHeight) / 2.0 * cur_field;
		point.x = m_ImgWidth / 2 + (int)(radius * cos(cur_degree / 180.0 * CV_PI) + 0.5);
		point.y = m_ImgHeight / 2 + (int)(radius * sin(cur_degree / 180.0 * CV_PI) + 0.5);
		point.x = std::max(point.x, 0);
		point.x = std::min(point.x, m_ImgWidth - 1);
		point.y = std::max(point.y, 0);
		point.y = std::min(point.y, m_ImgHeight - 1);

		if (block_size == 0)
			block_size = sqrt(m_ImgWidth * m_ImgWidth + m_ImgHeight * m_ImgHeight) / 2.0 / 10.0;

		cv::Rect point_box;

		int cen_offset_x = 0;
		int cen_offset_y = 0;
		point_box.x = point.x - 2 * block_size;
		point_box.y = point.y - 2 * block_size;
		point_box.width = 4 * block_size;
		point_box.height = 4 * block_size;
		if (point_box.x < 0) {
			int reduce_size = 0 - point_box.x;
			point_box.x += reduce_size;
			point_box.width -= reduce_size;
			cen_offset_x -= reduce_size / 2;
		}
		if (point_box.x + point_box.width - 1 > m_ImgWidth - 1) {
			int reduce_size = (point_box.x + point_box.width) - m_ImgWidth;
			point_box.width -= reduce_size;
			cen_offset_x += reduce_size / 2;
		}
		if (point_box.y < 0) {
			int reduce_size = 0 - point_box.y;
			point_box.y += reduce_size;
			point_box.height -= reduce_size;
			cen_offset_y -= reduce_size / 2;
		}
		if (point_box.y + point_box.height - 1 > m_ImgHeight - 1) {
			int reduce_size = (point_box.y + point_box.height) - m_ImgHeight;
			point_box.height -= reduce_size;
			cen_offset_y += reduce_size / 2;
		}

		cv::Mat point_img = m_GrayImage(point_box).clone();
		cv::Mat point_img_marked = m_MarkedImage(point_box);
		std::string str_field = std::to_string(cur_field);
		str_field = str_field.substr(0, str_field.find(".") + 2);
		std::string str_degree = std::to_string(cur_degree);
		str_degree = str_degree.substr(0, str_degree.find(".") + 2);
		std::string point_result_folder = m_ResultFolderPath + "\\Field" + str_field;

		point_result_folder = point_result_folder + "\\Degree" + str_degree;

		if (m_bSaveImg) {
			SaveGrayImg(point_result_folder, "0_RoiImg", point_img, point_img.cols, point_img.rows);
		}
		cv::Mat temp_img_up, temp_img_down, temp_img_left, temp_img_right;
		cv::Rect temp_pos_up, temp_pos_down, temp_pos_left, temp_pos_right;

		int ret_sub = 0;

		ret_sub = getChessEdgeImage_4Dir(point_img, point_img_marked, temp_img_up, temp_img_down, temp_img_left, temp_img_right,
			temp_pos_up, temp_pos_down, temp_pos_left, temp_pos_right, cen_offset_x, cen_offset_y,
			sfr_edge_dect_option.roi_length, sfr_edge_dect_option.roi_width, point_result_folder, &block_size);

		if (!(ret_sub & (int)EdgeDetectError::kHorWhiteToBlackFailed)) {
			edge_img_h_w2b.push_back(temp_img_up);
			temp_pos_up.x += point_box.x;
			temp_pos_up.y += point_box.y;
			if (edge_pos_h_w2b)	edge_pos_h_w2b->push_back(temp_pos_up);
		}
		else {
			if (m_OutputLog) {
				m_OutputLog("Field" + str_field + " Degree" + str_degree + " Edge H White To Black: Failed to get Edge Image");
			}
			edge_img_h_w2b.push_back(cv::Mat());
			if (edge_pos_h_w2b)	edge_pos_h_w2b->push_back(cv::Rect(-1, -1, -1, -1));
		}
		if (!(ret_sub & (int)EdgeDetectError::kHorBlackToWhiteFailed)) {
			edge_img_h_b2w.push_back(temp_img_down);
			temp_pos_down.x += point_box.x;
			temp_pos_down.y += point_box.y;
			if (edge_pos_h_b2w)	edge_pos_h_b2w->push_back(temp_pos_down);
		}
		else {
			if (m_OutputLog) {
				m_OutputLog("Field" + str_field + " Degree" + str_degree + " Edge H Black To White: Failed to get Edge Image");
			}
			edge_img_h_b2w.push_back(cv::Mat());
			if (edge_pos_h_b2w)	edge_pos_h_b2w->push_back(cv::Rect(-1, -1, -1, -1));
		}
		if (!(ret_sub & (int)EdgeDetectError::kVerWhiteToBlackFailed)) {
			edge_img_v_w2b.push_back(temp_img_left);
			temp_pos_left.x += point_box.x;
			temp_pos_left.y += point_box.y;
			if (edge_pos_v_w2b)	edge_pos_v_w2b->push_back(temp_pos_left);
		}
		else {
			if (m_OutputLog) {
				m_OutputLog("Field" + str_field + " Degree" + str_degree + " Edge V White To Black: Failed to get Edge Image");
			}
			edge_img_v_w2b.push_back(cv::Mat());
			if (edge_pos_v_w2b)	edge_pos_v_w2b->push_back(cv::Rect(-1, -1, -1, -1));
		}
		if (!(ret_sub & (int)EdgeDetectError::kVerBlackToWhiteFailed)) {
			edge_img_v_b2w.push_back(temp_img_right);
			temp_pos_right.x += point_box.x;
			temp_pos_right.y += point_box.y;
			if (edge_pos_v_b2w)	edge_pos_v_b2w->push_back(temp_pos_right);
		}
		else {
			if (m_OutputLog) {
				m_OutputLog("Field" + str_field + " Degree" + str_degree + " Edge V Black To White: Failed to get Edge Image");
			}
			edge_img_v_b2w.push_back(cv::Mat());
			if (edge_pos_v_b2w)	edge_pos_v_b2w->push_back(cv::Rect(-1, -1, -1, -1));
		}

		ret += ret_sub;
	}

	return ret;
}

int ChessBoard_CornerDetection::getChessEdgeImage_4Dir(cv::Mat gray_img, cv::Mat& gray_img_marked, cv::Mat& edge_img_h_w2b, cv::Mat& edge_img_h_b2w, cv::Mat& edge_img_v_w2b, cv::Mat& edge_img_v_b2w,
	cv::Rect& edge_pos_h_w2b, cv::Rect& edge_pos_h_b2w, cv::Rect& edge_pos_v_w2b, cv::Rect& edge_pos_v_b2w,
	int cen_offset_x, int cen_offset_y, int edge_roi_length, int edge_roi_width, std::string result_folder, int* cur_block_size) {

	// TODO: 识别直角边，返回靠近中心的有效 H/V 刀边（仅多边形近似最大误差循环搜索，节省些许时间）

	int ret = 0;

	int img_width = gray_img.cols;
	int img_height = gray_img.rows;

	std::vector<cv::Vec4i> edges_h, edges_v;
	std::vector<cv::Vec4i> edges_h_w2b, edges_h_b2w, edges_v_w2b, edges_v_b2w;
	double avg_slope_h, avg_slope_v;
	int approx_polygon_error;

	int point_cen_x = gray_img.cols / 2 + cen_offset_x;
	int point_cen_y = gray_img.rows / 2 + cen_offset_y;
	cv::Point point_cen(point_cen_x, point_cen_y);

	cv::Mat gray_img_threshold;
	cv::Mat gray_img_threshold_filtered;
	double median_edge_length = 0;
	int morph_iter;

	cv::Mat gray_img_hist;
	cv::equalizeHist(gray_img, gray_img_hist);

	for (int threshold_cnt = 0; threshold_cnt < 2; threshold_cnt++) {

		if (threshold_cnt == 1 && median_edge_length == 0) {
			break;
		}

		if (threshold_cnt == 0) {
			cv::threshold(gray_img_hist, gray_img_threshold, 0, 255, cv::THRESH_OTSU);	//选择 OTSU
		}
		else {
			int threshold_block_size = median_edge_length == 0 ? ((int)((double)img_width * 0.2) | 1) : ((int)(median_edge_length * 2/*4*/) | 1);
			int threshold_offset = 0;
			cv::adaptiveThreshold(gray_img_hist, gray_img_threshold, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, threshold_block_size, threshold_offset);
		}
		cv::morphologyEx(gray_img_threshold, gray_img_threshold_filtered, cv::MORPH_CLOSE, cv::Mat(), cv::Point(1, 1), 1);
		cv::morphologyEx(gray_img_threshold_filtered, gray_img_threshold_filtered, cv::MORPH_OPEN, cv::Mat(), cv::Point(1, 1), 1);

		if (m_bSaveImg) {
			SaveGrayImg(result_folder, "1_Img_Threshold", gray_img_threshold, gray_img.cols, gray_img.rows);
		}

		morph_iter = 5;
		int found_edges_num[2] = { 0 };
		for (int morph_cnt = 0; morph_cnt < 3; morph_cnt++) {

			cv::Mat element/* = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size))*/;
			cv::Mat gray_img_morph;

			if (morph_iter <= 0) {
				gray_img_morph = gray_img_threshold_filtered.clone();
			} else {
				if (morph_cnt == 0) {
					cv::dilate(gray_img_threshold_filtered, gray_img_morph, element, cv::Point(1, 1), morph_iter);	// 膨胀
				} else if (morph_cnt == 1) {
					cv::erode(gray_img_threshold_filtered, gray_img_morph, element, cv::Point(1, 1), morph_iter);	// 腐蚀
				} else {
					if (found_edges_num[0] >= found_edges_num[1]) {
						cv::dilate(gray_img_threshold_filtered, gray_img_morph, element, cv::Point(1, 1), morph_iter);	// 膨胀
					} else {
						cv::erode(gray_img_threshold_filtered, gray_img_morph, element, cv::Point(1, 1), morph_iter);	// 腐蚀
					}
				}
			}

			if (m_bSaveImg) {
				SaveGrayImg(result_folder, "2_Img_Morph", gray_img_morph, gray_img.cols, gray_img.rows);
			}

			//std::vector<std::vector<cv::Point>> contours(100000);
			//cv::findContours(gray_img_morph, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
			std::vector<std::vector<cv::Point>> contours;
			findContours2(gray_img_morph, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

			std::vector<bool> is_contour_valid(contours.size(), true);
#if 0
			for (size_t i = 0; i < contours.size(); i++) {
				cv::Rect rect = cv::boundingRect(contours[i]);
				if (rect.height == gray_img.rows - 2) {	// 此处排掉占据整个图像大小的边界
					is_contour_valid[i] = false;
				}
			}
#endif

			approx_polygon_error = median_edge_length == 0 ? 5/*10*//*20*/ : (int)(median_edge_length / 20.0/*10.0*//*5.0*/);
			std::vector<cv::Vec4i> right_angle_edges;
			double angle_diff_usl = 8.0/*5.0*/;
			for (int i = 0; i < contours.size(); i++) {
				if (is_contour_valid[i]) {
					findRightAngleEdges(contours[i], right_angle_edges, gray_img.cols, gray_img.rows, approx_polygon_error, angle_diff_usl);
				}
			}

			double length_diff_ratio_usl = 0.5;
			double length_usl = 30; //edge_roi_length * 0.8;
			filterEdgesByLength(right_angle_edges, length_diff_ratio_usl, length_usl);

			edges_h.clear();
			edges_v.clear();
			double slope_diff_usl = 0.1;
			filterEdgesBySlope(right_angle_edges, edges_h, edges_v, avg_slope_h, avg_slope_v, slope_diff_usl);

			if (threshold_cnt == 0) {
				std::vector<double> edges_length_h, edges_length_v;
				for (int i = 0; i < edges_h.size(); i++) {
					double cur_length = GetDistanceOfTwoPoint(edges_h[i][0], edges_h[i][1], edges_h[i][2], edges_h[i][3]);
					edges_length_h.push_back(cur_length);
				}
				for (int i = 0; i < edges_v.size(); i++) {
					double cur_length = GetDistanceOfTwoPoint(edges_v[i][0], edges_v[i][1], edges_v[i][2], edges_v[i][3]);
					edges_length_v.push_back(cur_length);
				}
				sort(edges_length_h.begin(), edges_length_h.end());
				sort(edges_length_v.begin(), edges_length_v.end());
				if (edges_length_h.size() != 0 && edges_length_v.size() != 0) {
					median_edge_length = (edges_length_h[edges_length_h.size() / 2] + edges_length_v[edges_length_v.size() / 2]) / 2.0;
				}
				else if (edges_length_h.size() != 0) {
					median_edge_length = edges_length_h[edges_length_h.size() / 2];
				}
				else if (edges_length_v.size() != 0) {
					median_edge_length = edges_length_v[edges_length_v.size() / 2];
				}
			}

			if (morph_cnt < 2) {
				found_edges_num[morph_cnt] = edges_h.size() + edges_v.size();
			}
		}
	}

	if (cur_block_size)
		*cur_block_size = median_edge_length + 0.5;

	cv::Mat gray_img_for_side_check = gray_img_threshold.clone();
	cv::morphologyEx(gray_img_for_side_check, gray_img_for_side_check, cv::MORPH_CLOSE, cv::Mat(), cv::Point(1, 1), 2);
	cv::morphologyEx(gray_img_for_side_check, gray_img_for_side_check, cv::MORPH_OPEN, cv::Mat(), cv::Point(1, 1), 2);
	int search_range;
	if (morph_iter <= 0) {
		search_range = edge_roi_width * 0.5 + approx_polygon_error * 2;
	}
	else {
		search_range = edge_roi_width * 0.5 + morph_iter * 2 + approx_polygon_error * 2;
	}
	double slope_diff_relocated_usl = 0.2;
	relocateEdgesAndCheckSides_4Dir(gray_img_for_side_check, edges_h, edges_v,
		edges_h_w2b, edges_h_b2w, edges_v_w2b, edges_v_b2w,
		avg_slope_h, avg_slope_v, search_range, slope_diff_relocated_usl, edge_roi_length, edge_roi_width);

	if (m_bSaveImg) {
		cv::Mat relocated_edges_img(gray_img.size(), CV_8UC3);
		cv::cvtColor(gray_img, relocated_edges_img, CV_GRAY2RGB);
		DrawLinesOnImg(relocated_edges_img, edges_h_w2b, cv::Scalar(0, 255, 255), 3);
		DrawLinesOnImg(relocated_edges_img, edges_h_b2w, cv::Scalar(255, 0, 0), 3);
		DrawLinesOnImg(relocated_edges_img, edges_v_w2b, cv::Scalar(0, 0, 255), 3);
		DrawLinesOnImg(relocated_edges_img, edges_v_b2w, cv::Scalar(0, 255, 0), 3);
		cv::line(relocated_edges_img, point_cen, point_cen, cv::Scalar(255, 0, 0), 25);
		SaveRGBImg(result_folder, "8_Img_RelocateEdges", relocated_edges_img, gray_img.cols, gray_img.rows);
	}

	cv::Vec4i closest_edge[4] = { cv::Vec4i(-1, -1, -1, -1), cv::Vec4i(-1, -1, -1, -1), cv::Vec4i(-1, -1, -1, -1), cv::Vec4i(-1, -1, -1, -1) };
	findClosestEdge(point_cen, edges_h_w2b, closest_edge[0]);
	findClosestEdge(point_cen, edges_h_b2w, closest_edge[1]);
	findClosestEdge(point_cen, edges_v_w2b, closest_edge[2]);
	findClosestEdge(point_cen, edges_v_b2w, closest_edge[3]);
	// 判断寻找的边缘是否异常，或边缘距离指定点是否过远（0.15 field 以上）
	for (int i = 0; i < 4; i++){
		if (closest_edge[i][0] < 0 || closest_edge[i][1] < 0 || closest_edge[i][2] < 0 || closest_edge[i][3] < 0 ||
			closest_edge[i][0] > gray_img.cols - 1 || closest_edge[i][1] > gray_img.rows - 1 || closest_edge[i][2] > gray_img.cols - 1 || closest_edge[i][3] > gray_img.rows - 1 ||
			GetDistanceOfTwoPoint(point_cen.x, point_cen.y, (closest_edge[i][0] + closest_edge[i][2]) / 2.0, (closest_edge[i][1] + closest_edge[i][3]) / 2.0) >
			sqrt(m_ImgWidth * m_ImgWidth + m_ImgHeight * m_ImgHeight) / 2.0 * 0.15) {
			switch (i){
			case 0:	ret |= (int)EdgeDetectError::kHorWhiteToBlackFailed; break;
			case 1:	ret |= (int)EdgeDetectError::kHorBlackToWhiteFailed; break;
			case 2:	ret |= (int)EdgeDetectError::kVerWhiteToBlackFailed; break;
			case 3:	ret |= (int)EdgeDetectError::kVerBlackToWhiteFailed; break;
			}
		}
	}

	cv::Rect rect;
	for (int i = 0; i < 2; i++){
		int ref;
		switch (i){
		case 0:	ref = (int)EdgeDetectError::kHorWhiteToBlackFailed; break;
		case 1:	ref = (int)EdgeDetectError::kHorBlackToWhiteFailed; break;
		}
		if (!(ret & ref)) {
			rect.x = (closest_edge[i][0] + closest_edge[i][2]) / 2 - edge_roi_length / 2;
			rect.y = (closest_edge[i][1] + closest_edge[i][3]) / 2 - edge_roi_width / 2;
			rect.width = edge_roi_length;
			rect.height = edge_roi_width;
			if (rect.x < std::min(closest_edge[i][0], closest_edge[i][2])) {
				rect.width -= std::min(closest_edge[i][0], closest_edge[i][2]) - rect.x;
				rect.x = std::min(closest_edge[i][0], closest_edge[i][2]);
			}
			if (rect.y < 0) {
				rect.height -= 0 - rect.y;
				rect.y = 0;
			}
			if (rect.x + rect.width - 1 > std::max(closest_edge[i][0], closest_edge[i][2])) {
				rect.width = std::max(closest_edge[i][0], closest_edge[i][2]) - rect.x + 1;
			}
			if (rect.y + rect.height - 1 > gray_img.rows - 1) {
				rect.height = gray_img.rows - rect.y;
			}
			switch (i){
			case 0:
				edge_img_h_w2b = (gray_img(rect)).clone();
				edge_pos_h_w2b = rect;
				break;
			case 1:
				edge_img_h_b2w = (gray_img(rect)).clone();
				edge_pos_h_b2w = rect;
				break;
			}
		}
	}
	for (int i = 2; i < 4; i++){
		int ref;
		switch (i){
		case 2:	ref = (int)EdgeDetectError::kVerWhiteToBlackFailed; break;
		case 3:	ref = (int)EdgeDetectError::kVerBlackToWhiteFailed; break;
		}
		if (!(ret & ref)) {
			rect.x = (closest_edge[i][0] + closest_edge[i][2]) / 2 - edge_roi_width / 2;
			rect.y = (closest_edge[i][1] + closest_edge[i][3]) / 2 - edge_roi_length / 2;
			rect.width = edge_roi_width;
			rect.height = edge_roi_length;
			if (rect.x < 0) {
				rect.width -= 0 - rect.x;
				rect.x = 0;
			}
			if (rect.y < std::min(closest_edge[i][1], closest_edge[i][3])) {
				rect.height -= std::min(closest_edge[i][1], closest_edge[i][3]) - rect.y;
				rect.y = std::min(closest_edge[i][1], closest_edge[i][3]);
			}
			if (rect.x + rect.width - 1 > gray_img.cols - 1) {
				rect.width = gray_img.cols - rect.x;
			}
			if (rect.y + rect.height - 1 > std::max(closest_edge[i][1], closest_edge[i][3])) {
				rect.height = std::max(closest_edge[i][1], closest_edge[i][3]) - rect.y + 1;
			}
			switch (i){
			case 2:
				edge_img_v_w2b = (gray_img(rect)).clone();
				edge_pos_v_w2b = rect;
				break;
			case 3:
				edge_img_v_b2w = (gray_img(rect)).clone();
				edge_pos_v_b2w = rect;
				break;
			}
		}
	}

	if (m_bSaveImg) {
		if (!(ret & (int)EdgeDetectError::kHorWhiteToBlackFailed)) {
			SaveGrayImg(result_folder, "11_Img_EdgeHorWhiteToBlack", edge_img_h_w2b, edge_img_h_w2b.cols, edge_img_h_w2b.rows);
		}
		if (!(ret & (int)EdgeDetectError::kHorBlackToWhiteFailed)) {
			SaveGrayImg(result_folder, "11_Img_EdgeHorBlackToWhite", edge_img_h_b2w, edge_img_h_b2w.cols, edge_img_h_b2w.rows);
		}
		if (!(ret & (int)EdgeDetectError::kVerWhiteToBlackFailed)) {
			SaveGrayImg(result_folder, "11_Img_EdgeVerWhiteToBlack", edge_img_v_w2b, edge_img_v_w2b.cols, edge_img_v_w2b.rows);
		}
		if (!(ret & (int)EdgeDetectError::kVerBlackToWhiteFailed)) {
			SaveGrayImg(result_folder, "11_Img_EdgeVerBlackToWhite", edge_img_v_b2w, edge_img_v_b2w.cols, edge_img_v_b2w.rows);
		}
	}

	if (!(ret & (int)EdgeDetectError::kHorWhiteToBlackFailed)) {
		cv::rectangle(gray_img_marked, edge_pos_h_w2b, cv::Scalar(0, 255, 255), 3);
	}
	if (!(ret & (int)EdgeDetectError::kHorBlackToWhiteFailed)) {
		cv::rectangle(gray_img_marked, edge_pos_h_b2w, cv::Scalar(255, 0, 0), 3);
	}
	if (!(ret & (int)EdgeDetectError::kVerWhiteToBlackFailed)) {
		cv::rectangle(gray_img_marked, edge_pos_v_w2b, cv::Scalar(0, 0, 255), 3);
	}
	if (!(ret & (int)EdgeDetectError::kVerBlackToWhiteFailed)) {
		cv::rectangle(gray_img_marked, edge_pos_v_b2w, cv::Scalar(0, 255, 0), 3);
	}
	cv::line(gray_img_marked, point_cen, point_cen, cv::Scalar(255, 0, 0), 25);

	return ret;
}

void ChessBoard_CornerDetection::findRightAngleEdges(std::vector<cv::Point> contour, std::vector<cv::Vec4i>& edges, int img_width, int img_height, int approx_polygon_error, double angle_diff_usl) {

	cv::Mat contour_polygon;

	cv::approxPolyDP(contour, contour_polygon, approx_polygon_error, false);

	cv::Vec4i temp;
	int point_num = contour_polygon.rows;
	if (point_num < 3)	return;
	std::vector<bool> edge_exsit(point_num - 1);

	for (int i = 0; i < point_num - 1; i++) {
		int cur_num = i;
		int before_num = i - 1;
		int next_num = i + 1;
		if (i == 0) {
			if (abs(contour_polygon.at<cv::Vec2i>(point_num - 1, 0)[0] - contour_polygon.at<cv::Vec2i>(0, 0)[0]) <= 1 &&
				abs(contour_polygon.at<cv::Vec2i>(point_num - 1, 0)[1] - contour_polygon.at<cv::Vec2i>(0, 0)[1]) <= 1) {
				before_num = point_num - 2;
			}
			else {
				continue;
			}
		}
		int cur_x = contour_polygon.at<cv::Vec2i>(cur_num, 0)[0];
		int cur_y = contour_polygon.at<cv::Vec2i>(cur_num, 0)[1];
		int before_x = contour_polygon.at<cv::Vec2i>(before_num, 0)[0];
		int before_y = contour_polygon.at<cv::Vec2i>(before_num, 0)[1];
		int next_x = contour_polygon.at<cv::Vec2i>(next_num, 0)[0];
		int next_y = contour_polygon.at<cv::Vec2i>(next_num, 0)[1];

		double cur_angle = GetAngleOfTwoVector(before_x, before_y, next_x, next_y, cur_x, cur_y);
		//double cur_length1 = MyImgUtil::GetDistanceOfTwoPoint(before_x, before_y, cur_x, cur_y);
		//double cur_length2 = MyImgUtil::GetDistanceOfTwoPoint(next_x, next_y, cur_x, cur_y);

		if (cur_angle > 0) {
			if (cur_angle < 90 - angle_diff_usl || cur_angle > 90 + angle_diff_usl) {
				edge_exsit[cur_num] = false;
				continue;
			}
		}
		else {
			if (cur_angle < -90 - angle_diff_usl || cur_angle > -90 + angle_diff_usl) {
				edge_exsit[cur_num] = false;
				continue;
			}
		}

		if (!edge_exsit[before_num]) {
			if ((cur_x < 10 && before_x < 10) ||
				(img_width - 1 - cur_x < 10 && img_width - 1 - before_x < 10) ||
				(cur_y < 10 && before_y < 10) ||
				(img_height - 1 - cur_y < 10 && img_height - 1 - before_y < 10)) {
				continue;
			}
			temp[0] = before_x;
			temp[1] = before_y;
			temp[2] = cur_x;
			temp[3] = cur_y;
			edges.push_back(temp);
			edge_exsit[before_num] = true;
		}

		if (!edge_exsit[cur_num]) {
			if ((cur_x < 10 && next_x < 10) ||
				(img_width - 1 - cur_x < 10 && img_width - 1 - next_x < 10) ||
				(cur_y < 10 && next_y < 10) ||
				(img_height - 1 - cur_y < 10 && img_height - 1 - next_y < 10)) {
				continue;
			}
			temp[0] = cur_x;
			temp[1] = cur_y;
			temp[2] = next_x;
			temp[3] = next_y;
			edges.push_back(temp);
			edge_exsit[cur_num] = true;
		}
	}

	return;
}

void ChessBoard_CornerDetection::filterEdgesBySlope(std::vector<cv::Vec4i>& edges, std::vector<cv::Vec4i>& edges_h, std::vector<cv::Vec4i>& edges_v,
	double& avg_slope_h, double& avg_slope_v, double slope_diff_usl) {

	double temp_avg_slope_h = 0;
	double temp_avg_slope_v = 0;
	int sum_cnt_h = 0;
	int sum_cnt_v = 0;

	for (int i = 0; i < edges.size(); i++) {
		int x1 = edges[i][0];
		int y1 = edges[i][1];
		int x2 = edges[i][2];
		int y2 = edges[i][3];

		EdgeType edge_type = EdgeType::EDGE_NOT_DEFINDED;
		double temp_slope = GetEdgeSlope(x1, y1, x2, y2, edge_type);

		if (edge_type == EdgeType::EDGE_HORIZONTAL) {
			temp_avg_slope_h += temp_slope;
			sum_cnt_h++;
		}
		else {
			temp_avg_slope_v += temp_slope;
			sum_cnt_v++;
		}
	}
	temp_avg_slope_h /= (double)sum_cnt_h;
	temp_avg_slope_v /= (double)sum_cnt_v;

	for (int i = 0; i < edges.size(); i++) {
		int x1 = edges[i][0];
		int y1 = edges[i][1];
		int x2 = edges[i][2];
		int y2 = edges[i][3];

		EdgeType edge_type = EdgeType::EDGE_NOT_DEFINDED;
		double temp_slope = GetEdgeSlope(x1, y1, x2, y2, edge_type);

		if (edge_type == EdgeType::EDGE_HORIZONTAL) {
			if (fabs(temp_slope - temp_avg_slope_h) > slope_diff_usl) {
				edges.erase(edges.begin() + i);
				i--;
				continue;
			}
		}
		else {
			if (fabs(temp_slope - temp_avg_slope_v) > slope_diff_usl) {
				edges.erase(edges.begin() + i);
				i--;
				continue;
			}
		}
	}

	temp_avg_slope_h = 0;
	temp_avg_slope_v = 0;
	sum_cnt_h = 0;
	sum_cnt_v = 0;
	for (int i = 0; i < edges.size(); i++) {
		int x1 = edges[i][0];
		int y1 = edges[i][1];
		int x2 = edges[i][2];
		int y2 = edges[i][3];

		EdgeType edge_type = EdgeType::EDGE_NOT_DEFINDED;
		double temp_slope = GetEdgeSlope(x1, y1, x2, y2, edge_type);

		if (edge_type == EdgeType::EDGE_HORIZONTAL) {
			temp_avg_slope_h += temp_slope;
			sum_cnt_h++;

			cv::Vec4i temp(edges[i][0], edges[i][1], edges[i][2], edges[i][3]);
			edges_h.push_back(temp);
		}
		else {
			temp_avg_slope_v += temp_slope;
			sum_cnt_v++;

			cv::Vec4i temp(edges[i][0], edges[i][1], edges[i][2], edges[i][3]);
			edges_v.push_back(temp);
		}
	}
	temp_avg_slope_h /= (double)sum_cnt_h;
	temp_avg_slope_v /= (double)sum_cnt_v;

	avg_slope_h = temp_avg_slope_h;
	avg_slope_v = temp_avg_slope_v;

	return;
}

void ChessBoard_CornerDetection::filterEdgesByLength(std::vector<cv::Vec4i>& edges, double length_diff_ratio_usl, double length_usl) {

	double temp_avg_edge_length = 0;
	int sum_cnt = 0;

	for (int i = 0; i < edges.size(); i++) {
		int x1 = edges[i][0];
		int y1 = edges[i][1];
		int x2 = edges[i][2];
		int y2 = edges[i][3];
		double temp_length = GetDistanceOfTwoPoint(x1, y1, x2, y2);

		if (temp_length < length_usl) {
			edges.erase(edges.begin() + i);
			i--;
			continue;
		}
		temp_avg_edge_length += temp_length;
		sum_cnt++;
	}
	temp_avg_edge_length /= (double)sum_cnt;

	for (int i = 0; i < edges.size(); i++) {
		int x1 = edges[i][0];
		int y1 = edges[i][1];
		int x2 = edges[i][2];
		int y2 = edges[i][3];
		double temp_length = GetDistanceOfTwoPoint(x1, y1, x2, y2);

		if ((temp_length < temp_avg_edge_length) && (fabs((temp_length - temp_avg_edge_length) / temp_avg_edge_length) > length_diff_ratio_usl)) {
			edges.erase(edges.begin() + i);
			i--;
			continue;
		}
	}

	return;
}

void ChessBoard_CornerDetection::relocateEdgesAndCheckSides_4Dir(const cv::Mat& gray_img_threshold, std::vector<cv::Vec4i> edges_h, std::vector<cv::Vec4i> edges_v,
	std::vector<cv::Vec4i>& edges_h_w2b, std::vector<cv::Vec4i>& edges_h_b2w, std::vector<cv::Vec4i>& edges_v_w2b, std::vector<cv::Vec4i>& edges_v_b2w,
	double avg_slope_h, double avg_slope_v, int search_range, double slope_diff_usl, double required_roi_length, double required_roi_width) {

	for (int i = 0; i < edges_h.size(); i++) {
		//double line_reduce_ratio = ((double)abs(edges_h[i][2] - edges_h[i][0]) - required_roi_length) / 2.0 / (double)abs(edges_h[i][2] - edges_h[i][0]);
		//if (line_reduce_ratio < 0.1) {
		//	line_reduce_ratio = 0.1;
		//}
		double line_reduce_ratio = 0.1;
		cv::Point base_pt[2];
		base_pt[0].x = (edges_h[i][2] - edges_h[i][0]) * line_reduce_ratio + edges_h[i][0];
		base_pt[0].y = (edges_h[i][3] - edges_h[i][1]) * line_reduce_ratio + edges_h[i][1];
		base_pt[1].x = (edges_h[i][0] - edges_h[i][2]) * line_reduce_ratio + edges_h[i][2];
		base_pt[1].y = (edges_h[i][1] - edges_h[i][3]) * line_reduce_ratio + edges_h[i][3];

		std::vector<cv::Point> found_pt[2];

		for (int pt_cnt = 0; pt_cnt < 2; pt_cnt++) {

			cv::Point up, down;
			bool no_more_expand_up = false;
			bool no_more_expand_down = false;
			bool is_first_change_up = true;
			bool is_first_change_down = true;

			for (int search_cnt = 0; search_cnt < search_range; search_cnt++) {
				up.y = base_pt[pt_cnt].y - search_cnt;
				down.y = base_pt[pt_cnt].y + search_cnt;
				if (up.y < 0) {
					no_more_expand_up = true;
					up.y = 0;
				}
				if (down.y > gray_img_threshold.rows - 1) {
					no_more_expand_down = true;
					down.y = gray_img_threshold.rows - 1;
				}

				int x_offset = (int)(-avg_slope_h * (double)search_cnt + 0.5f);
				up.x = base_pt[pt_cnt].x - x_offset;
				down.x = base_pt[pt_cnt].x + x_offset;
				up.x = std::max(up.x, 0);
				up.x = std::min(up.x, gray_img_threshold.cols - 1);
				down.x = std::max(down.x, 0);
				down.x = std::min(down.x, gray_img_threshold.cols - 1);

				if (!no_more_expand_up) {
					if (gray_img_threshold.at<uchar>(up.y, up.x) != gray_img_threshold.at<uchar>(base_pt[pt_cnt].y, base_pt[pt_cnt].x)) {
						if (is_first_change_up == true) {
							found_pt[pt_cnt].push_back(up);
							is_first_change_up = false;
						}
					}
				}
				else {
					is_first_change_up = true;
				}

				if (!no_more_expand_down) {
					if (gray_img_threshold.at<uchar>(down.y, down.x) != gray_img_threshold.at<uchar>(base_pt[pt_cnt].y, base_pt[pt_cnt].x)) {
						if (is_first_change_down == true) {
							found_pt[pt_cnt].push_back(down);
							is_first_change_down = false;
						}
					}
					else {
						is_first_change_down = true;
					}
				}

				if (no_more_expand_up && no_more_expand_down) {
					break;
				}
			}
		}

		if (found_pt[0].size() == 0 || found_pt[1].size() == 0) {
			//edges_h.erase(edges_h.begin() + i);
			//i--;
			continue;
		}

		cv::Point final_pt[2] = { cv::Point(-1, -1), cv::Point(-1, -1) };
		double slope_final;
		double cur_slope = (double)(edges_h[i][3] - edges_h[i][1]) / (double)(edges_h[i][2] - edges_h[i][0]);
		double min_slope_diff = FLT_MAX;
		for (int num1 = 0; num1 < found_pt[0].size(); num1++) {
			for (int num2 = 0; num2 < found_pt[1].size(); num2++) {
				double temp_slope = (double)(found_pt[1][num2].y - found_pt[0][num1].y) / (double)(found_pt[1][num2].x - found_pt[0][num1].x);
				double temp_diff = fabs(temp_slope - cur_slope);
				if (min_slope_diff > temp_diff) {
					min_slope_diff = temp_diff;
					slope_final = temp_slope;
					final_pt[0].x = found_pt[0][num1].x;
					final_pt[0].y = found_pt[0][num1].y;
					final_pt[1].x = found_pt[1][num2].x;
					final_pt[1].y = found_pt[1][num2].y;
				}
			}
		}
		if (min_slope_diff > slope_diff_usl || final_pt[0].x == -1 || final_pt[0].y == -1 || final_pt[1].x == -1 || final_pt[1].y == -1) {
			//edges_h.erase(edges_h.begin() + i);
			//i--;
			continue;
		}

		edges_h[i][0] = final_pt[0].x;
		edges_h[i][1] = final_pt[0].y;
		edges_h[i][2] = final_pt[1].x;
		edges_h[i][3] = final_pt[1].y;

		bool is_w2b = true;
		if (checkEdgeSides_AndBrightnessDir(gray_img_threshold, edges_h[i], EdgeType::EDGE_HORIZONTAL, required_roi_length, required_roi_width, is_w2b) == false) {
			//edges_h.erase(edges_h.begin() + i);
			//i--;
			continue;
		}

		if (is_w2b){
			edges_h_w2b.push_back(edges_h[i]);
		}
		else {
			edges_h_b2w.push_back(edges_h[i]);
		}
	}

	for (int i = 0; i < edges_v.size(); i++) {
		//double line_reduce_ratio = ((double)abs(edges_v[i][3] - edges_v[i][1]) - required_roi_length) / 2.0 / (double)abs(edges_v[i][3] - edges_v[i][1]);
		//if (line_reduce_ratio < 0.1) {
		//	line_reduce_ratio = 0.1;
		//}
		double line_reduce_ratio = 0.1;
		cv::Point base_pt[2];
		base_pt[0].x = (edges_v[i][2] - edges_v[i][0]) * line_reduce_ratio + edges_v[i][0];
		base_pt[0].y = (edges_v[i][3] - edges_v[i][1]) * line_reduce_ratio + edges_v[i][1];
		base_pt[1].x = (edges_v[i][0] - edges_v[i][2]) * line_reduce_ratio + edges_v[i][2];
		base_pt[1].y = (edges_v[i][1] - edges_v[i][3]) * line_reduce_ratio + edges_v[i][3];

		std::vector<cv::Point> found_pt[2];

		for (int pt_cnt = 0; pt_cnt < 2; pt_cnt++) {

			cv::Point left, right;
			bool no_more_expand_left = false;
			bool no_more_expand_right = false;
			bool is_first_change_left = true;
			bool is_first_change_right = true;

			for (int search_cnt = 0; search_cnt < search_range; search_cnt++) {
				left.x = base_pt[pt_cnt].x - search_cnt;
				right.x = base_pt[pt_cnt].x + search_cnt;
				if (left.x < 0) {
					no_more_expand_left = true;
					left.x = 0;
				}
				if (right.x > gray_img_threshold.cols - 1) {
					no_more_expand_right = true;
					right.x = gray_img_threshold.cols - 1;
				}

				int y_offset = (int)(-avg_slope_v * (double)search_cnt + 0.5f);
				left.y = base_pt[pt_cnt].y - y_offset;
				right.y = base_pt[pt_cnt].y + y_offset;
				left.y = std::max(left.y, 0);
				left.y = std::min(left.y, gray_img_threshold.rows - 1);
				right.y = std::max(right.y, 0);
				right.y = std::min(right.y, gray_img_threshold.rows - 1);

				if (!no_more_expand_left) {
					if (gray_img_threshold.at<uchar>(left.y, left.x) != gray_img_threshold.at<uchar>(base_pt[pt_cnt].y, base_pt[pt_cnt].x)) {
						if (is_first_change_left == true) {
							found_pt[pt_cnt].push_back(left);
							is_first_change_left = false;
						}
					}
				}
				else {
					is_first_change_left = true;
				}

				if (!no_more_expand_right) {
					if (gray_img_threshold.at<uchar>(right.y, right.x) != gray_img_threshold.at<uchar>(base_pt[pt_cnt].y, base_pt[pt_cnt].x)) {
						if (is_first_change_right == true) {
							found_pt[pt_cnt].push_back(right);
							is_first_change_right = false;
						}
					}
					else {
						is_first_change_right = true;
					}
				}

				if (no_more_expand_left && no_more_expand_right) {
					break;
				}
			}
		}

		if (found_pt[0].size() == 0 || found_pt[1].size() == 0) {
			//edges_v.erase(edges_v.begin() + i);
			//i--;
			continue;
		}

		cv::Point final_pt[2] = { cv::Point(-1, -1), cv::Point(-1, -1) };
		double slope_final;
		double cur_slope = (double)(edges_v[i][2] - edges_v[i][0]) / (double)(edges_v[i][3] - edges_v[i][1]);
		double min_slope_diff = FLT_MAX;
		for (int num1 = 0; num1 < found_pt[0].size(); num1++) {
			for (int num2 = 0; num2 < found_pt[1].size(); num2++) {
				double temp_slope = (double)(found_pt[1][num2].x - found_pt[0][num1].x) / (double)(found_pt[1][num2].y - found_pt[0][num1].y);
				double temp_diff = fabs(temp_slope - cur_slope);
				if (min_slope_diff > temp_diff) {
					min_slope_diff = temp_diff;
					slope_final = temp_slope;
					final_pt[0].x = found_pt[0][num1].x;
					final_pt[0].y = found_pt[0][num1].y;
					final_pt[1].x = found_pt[1][num2].x;
					final_pt[1].y = found_pt[1][num2].y;
				}
			}
		}
		if (min_slope_diff > slope_diff_usl || final_pt[0].x == -1 || final_pt[0].y == -1 || final_pt[1].x == -1 || final_pt[1].y == -1) {
			//edges_v.erase(edges_v.begin() + i);
			//i--;
			continue;
		}

		edges_v[i][0] = final_pt[0].x;
		edges_v[i][1] = final_pt[0].y;
		edges_v[i][2] = final_pt[1].x;
		edges_v[i][3] = final_pt[1].y;

		bool is_w2b = true;
		if (checkEdgeSides_AndBrightnessDir(gray_img_threshold, edges_v[i], EdgeType::EDGE_VERTICAL, required_roi_length, required_roi_width, is_w2b) == false) {
			//edges_v.erase(edges_v.begin() + i);
			//i--;
			continue;
		}

		if (is_w2b){
			edges_v_w2b.push_back(edges_v[i]);
		}
		else {
			edges_v_b2w.push_back(edges_v[i]);
		}
	}

	return;
}

bool ChessBoard_CornerDetection::checkEdgeSides_AndBrightnessDir(const cv::Mat& gray_img_threshold, cv::Vec4i& edge, EdgeType edge_type, int required_roi_length, int required_roi_width,
	bool& is_w2b) {

	if (edge_type == EdgeType::EDGE_NOT_DEFINDED || gray_img_threshold.empty() ||
		edge[0] < 0 || edge[0] > gray_img_threshold.cols - 1 ||
		edge[1] < 0 || edge[1] > gray_img_threshold.rows - 1 ||
		edge[2] < 0 || edge[2] > gray_img_threshold.cols - 1 ||
		edge[3] < 0 || edge[3] > gray_img_threshold.rows - 1)
		return false;

	const int gap_size = 5;

	cv::Vec4i edge_origin = edge;
	cv::Mat half_diff;
	int check_roi_length = 0;
	int check_roi_width = 0;

	double overlap_ratio = FLT_MAX;
	bool is_ok = false;

	for (int cnt = 0; cnt <= 3; cnt++) {

		half_diff.release();

		switch (cnt) {
		case 1:
			edge = cv::Vec4i(edge_origin[0], edge_origin[1], (double)(edge_origin[0] + edge_origin[2]) / 2.0 + 0.5, (double)(edge_origin[1] + edge_origin[3]) / 2.0 + 0.5);
			break;
		case 2:
			edge = cv::Vec4i((double)(edge_origin[0] + edge_origin[2]) / 2.0 + 0.5, (double)(edge_origin[1] + edge_origin[3]) / 2.0 + 0.5, edge_origin[2], edge_origin[3]);
			break;
		case 3:
			edge = cv::Vec4i(
				edge_origin[0] + (double)(edge_origin[2] - edge_origin[0]) / 4.0 + 0.5,
				edge_origin[1] + (double)(edge_origin[3] - edge_origin[1]) / 4.0 + 0.5,
				edge_origin[0] + (double)(edge_origin[2] - edge_origin[0]) / 4.0 * 3.0 + 0.5,
				edge_origin[1] + (double)(edge_origin[3] - edge_origin[1]) / 4.0 * 3.0 + 0.5);
			break;
		default:
			edge = edge_origin;
			break;
		}

		if (edge_type == EdgeType::EDGE_HORIZONTAL) {

			check_roi_length = std::min(required_roi_length, abs(edge[0] - edge[2]));
			double line_reduce_ratio = ((double)abs(edge[2] - edge[0]) - check_roi_length) / 2.0 / (double)abs(edge[2] - edge[0]);
			line_reduce_ratio = std::min(std::max(0.0, line_reduce_ratio), 0.9);
			check_roi_width = std::max((int)((double)required_roi_width / 2.0 - (double)(abs(edge[1] - edge[3]) * (1.0 - 2 * line_reduce_ratio) + gap_size * 2) / 2.0 + 1.0 + 0.5), 1);

			int x_start = (int)((double)(edge[0] + edge[2]) / 2.0 + 0.5) - (int)((double)check_roi_length / 2.0 + 0.5) + 1;
			int x_end = (int)((double)(edge[0] + edge[2]) / 2.0 + 0.5) + (int)((double)check_roi_length / 2.0 + 0.5) - 1;
			//int y_up_half_end = std::min(edge[1], edge[3]) - gap_size;
			int y_up_half_end = (std::max(edge[1], edge[3]) - std::min(edge[1], edge[3])) * line_reduce_ratio + std::min(edge[1], edge[3]) - gap_size;
			int y_up_half_start = y_up_half_end - check_roi_width + 1;
			//int y_down_half_start = std::max(edge[1], edge[3]) + gap_size;
			int y_down_half_start = (std::min(edge[1], edge[3]) - std::max(edge[1], edge[3])) * line_reduce_ratio + std::max(edge[1], edge[3]) + gap_size;
			int y_down_half_end = y_down_half_start + check_roi_width - 1;

			if (x_start < 0 || x_start > gray_img_threshold.cols - 1 ||
				x_end < 0 || x_end > gray_img_threshold.cols - 1 ||
				y_up_half_end < 0 || y_up_half_end > gray_img_threshold.rows - 1 ||
				y_up_half_start < 0 || y_up_half_start > gray_img_threshold.rows - 1 ||
				y_down_half_start < 0 || y_down_half_start > gray_img_threshold.rows - 1 ||
				y_down_half_end < 0 || y_down_half_end > gray_img_threshold.rows - 1 ||
				x_end < x_start ||
				y_up_half_end < y_up_half_start ||
				y_down_half_end < y_down_half_start) {
				continue;
			}

			cv::Mat up_half = gray_img_threshold(cv::Rect(x_start, y_up_half_start, x_end - x_start + 1, y_up_half_end - y_up_half_start + 1)).clone();
			cv::Mat down_half = gray_img_threshold(cv::Rect(x_start, y_down_half_start, x_end - x_start + 1, y_down_half_end - y_down_half_start + 1)).clone();

			cv::flip(down_half, down_half, 1);
			cv::flip(down_half, down_half, 0);
			cv::bitwise_xor(up_half, down_half, half_diff);
			cv::bitwise_not(half_diff, half_diff);

			if (cv::mean(up_half)[0] > 2 * cv::mean(down_half)[0]){
				is_w2b = true;
			}
			if (cv::mean(down_half)[0] > 2 * cv::mean(up_half)[0]){
				is_w2b = false;
			}

		}
		else if (edge_type == EdgeType::EDGE_VERTICAL) {

			check_roi_length = std::min(required_roi_length, abs(edge[1] - edge[3]));
			double line_reduce_ratio = ((double)abs(edge[3] - edge[1]) - check_roi_length) / 2.0 / (double)abs(edge[3] - edge[1]);
			line_reduce_ratio = std::min(std::max(0.0, line_reduce_ratio), 0.9);
			check_roi_width = std::max((int)((double)required_roi_width / 2.0 - (double)(abs(edge[0] - edge[2]) * (1.0 - 2 * line_reduce_ratio) + gap_size * 2) / 2.0 + 1.0 + 0.5), 1);

			int y_start = (int)((double)(edge[1] + edge[3]) / 2.0 + 0.5) - (int)((double)check_roi_length / 2.0 + 0.5) + 1;
			int y_end = (int)((double)(edge[1] + edge[3]) / 2.0 + 0.5) + (int)((double)check_roi_length / 2.0 + 0.5) - 1;
			//int x_left_half_end = std::min(edge[0], edge[2]) - gap_size;
			int x_left_half_end = (std::max(edge[0], edge[2]) - std::min(edge[0], edge[2])) * line_reduce_ratio + std::min(edge[0], edge[2]) - gap_size;
			int x_left_half_start = x_left_half_end - check_roi_width + 1;
			//int x_right_half_start = std::max(edge[0], edge[2]) + gap_size;
			int x_right_half_start = (std::min(edge[0], edge[2]) - std::max(edge[0], edge[2])) * line_reduce_ratio + std::max(edge[0], edge[2]) + gap_size;
			int x_right_half_end = x_right_half_start + check_roi_width - 1;

			if (y_start < 0 || y_start > gray_img_threshold.rows - 1 ||
				y_end < 0 || y_end > gray_img_threshold.rows - 1 ||
				x_left_half_end < 0 || x_left_half_end > gray_img_threshold.cols - 1 ||
				x_left_half_start < 0 || x_left_half_start > gray_img_threshold.cols - 1 ||
				x_right_half_start < 0 || x_right_half_start > gray_img_threshold.cols - 1 ||
				x_right_half_end < 0 || x_right_half_end > gray_img_threshold.cols - 1 ||
				y_end < y_start ||
				x_left_half_end < x_left_half_start ||
				x_right_half_end < x_right_half_start) {
				continue;
			}

			cv::Mat left_half = gray_img_threshold(cv::Rect(x_left_half_start, y_start, x_left_half_end - x_left_half_start + 1, y_end - y_start + 1)).clone();
			cv::Mat right_half = gray_img_threshold(cv::Rect(x_right_half_start, y_start, x_right_half_end - x_right_half_start + 1, y_end - y_start + 1)).clone();

			cv::flip(right_half, right_half, 1);
			cv::flip(right_half, right_half, 0);
			cv::bitwise_xor(left_half, right_half, half_diff);
			cv::bitwise_not(half_diff, half_diff);

			if (cv::mean(left_half)[0] > 2 * cv::mean(right_half)[0]){
				is_w2b = true;
			}
			if (cv::mean(right_half)[0] > 2 * cv::mean(left_half)[0]){
				is_w2b = false;
			}

		}

		if (half_diff.empty()) {
			continue;
		}

		overlap_ratio = (double)cv::countNonZero(half_diff) / (double)(required_roi_length * required_roi_width/*check_roi_length * check_roi_width*/);

		if (!isnan(overlap_ratio) && overlap_ratio <= EDGE_SIDE_CHK_COLOR_OVERLAP_RATIO_THRES) {
			is_ok = true;
			break;
		}
	}

	if (is_ok == false) {
		edge = edge_origin;
	}

	return is_ok;
}

void ChessBoard_CornerDetection::findClosestEdge(cv::Point point, std::vector<cv::Vec4i> edges, cv::Vec4i& closest_edge) {

	cv::Vec4i temp_edge = cv::Vec4i(-1, -1, -1, -1);
	double min_distance;

	int pt_x = point.x;
	int pt_y = point.y;

	min_distance = FLT_MAX;
	for (int i = 0; i < edges.size(); i++) {
		double cur_edge_cen_x = (double)(edges[i][0] + edges[i][2]) / 2.0;
		double cur_edge_cen_y = (double)(edges[i][1] + edges[i][3]) / 2.0;
		double cur_distance = GetDistanceOfTwoPoint(pt_x, pt_y, cur_edge_cen_x, cur_edge_cen_y);

		if (cur_distance < min_distance) {
			min_distance = cur_distance;
			temp_edge = edges[i];
		}
	}

	closest_edge = temp_edge;

	return;
}

void ChessBoard_CornerDetection::DrawLinesOnImg(cv::Mat& img, std::vector<cv::Vec4i> lines, cv::Scalar color, int line_width) {

	for (size_t i = 0; i < lines.size(); i++) {
		cv::line(img, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), color, line_width);
	}

	return;
}