#include"ChessBoard_CamCalibration.h"

ChessBoard_CamCalibration::ChessBoard_CamCalibration()
{
	m_OutputLog = NULL;
}

ChessBoard_CamCalibration::~ChessBoard_CamCalibration()
{

}

void ChessBoard_CamCalibration::Init(pOutputLog OutputLogFuntion)
{
	m_OutputLog = OutputLogFuntion;

	return;
}

int ChessBoard_CamCalibration::ProcessSingleCamCal(std::vector<std::vector<cv::Point3f>> objectPoints, std::vector<std::vector<cv::Point2f>> imagePoints, int imgWidth, int imgHeight,
	cv::Mat& matrix_M, bool bFixMatrixM, cv::Mat& matrix_K, cv::Mat& matrix_P, std::vector<cv::Mat>& vec_R, std::vector<cv::Mat>& vec_T, std::vector<double>& error, double& avgError)
{
	if (imgWidth <= 0 || imgHeight <= 0) {
		m_OutputLog("Invalid Image Size!");
		return -1;
	}

	int numImg = objectPoints.size();
	if (imagePoints.size() != numImg) {
		if (m_OutputLog) {
			m_OutputLog("Group size mismatch: Object points & Image points");
		}
		return -1;
	}

	//1. 求解每张图对应的单应性矩阵 H
	std::vector<cv::Mat> matrix_H;
	for (int imgCnt = 0; imgCnt < numImg; imgCnt++)
	{
		//先转化为2维坐标
		int numPoint = objectPoints[imgCnt].size();
		std::vector<cv::Point2f> objectPoints_2d(numPoint);
		for (int ptCnt = 0; ptCnt < numPoint; ptCnt++) {
			objectPoints_2d[ptCnt].x = objectPoints[imgCnt][ptCnt].x;
			objectPoints_2d[ptCnt].y = objectPoints[imgCnt][ptCnt].y;
		}
		cv::Mat tempMatrix_H;
		if (GetMatrixH(objectPoints_2d, imagePoints[imgCnt], tempMatrix_H)) {
			return -1;
		}

		matrix_H.push_back(tempMatrix_H);
	}

	//2. 获取初始内参
	bool bFix_u0v0Gamma = false;
#ifdef	FIX_U0_V0_GAMMA
	bFix_u0v0Gamma = true;
#endif
	if (matrix_M.rows != 3 || matrix_M.cols != 3 || !bFixMatrixM) {
		if (GetInitialIntrinsicParams(matrix_H, imgWidth, imgHeight, matrix_M, bFix_u0v0Gamma)) {
			return -1;
		}
	}

	//3. 获取初始外参
	GetInitialExtrinsicParams(matrix_H, matrix_M, vec_R, vec_T);

	//4. 初始化畸变参数
#if DISTORTION_ACCURACY == 1
	if (matrix_K.rows != 1 || matrix_K.cols != 6) {
		matrix_K.release();
		matrix_K = (cv::Mat_<double>(1, 6) << 0, 0, 0, 0, 0, 0);
	}
#else
	if (matrix_K.rows != 1 || matrix_K.cols != 3) {
		matrix_K.release();
		matrix_K = (cv::Mat_<double>(1, 3) << 0, 0, 0);
	}
#endif
	if (matrix_K.rows != 1 || matrix_K.cols != 2) {
		matrix_P.release();
		matrix_P = (cv::Mat_<double>(1, 2) << 0, 0);
	}

	//5. LM迭代优化参数
	avgError = 0;
	if (OptimizeSingleCameraParams_LevMarq(objectPoints, imagePoints, matrix_M, bFixMatrixM, matrix_K, matrix_P, vec_R, vec_T, error, avgError)) {
		return -1;
	}

	return 0;
}

int ChessBoard_CamCalibration::ProcessGetHomography(std::vector<cv::Point2f> points_src, std::vector<cv::Point2f> points_dest, cv::Mat& matrix_H)
{
	if (GetMatrixH(points_src, points_dest, matrix_H)) {
		return -1;
	}

	return 0;
}

int ChessBoard_CamCalibration::RotationMatrixToEulerAngles(cv::Mat rotationMatrix, double& pitch, double& yaw, double& roll)
{
	//- 旋转和平移相机 可等价化为 在相机主点处建立坐标系，在该坐标系下 对被摄画面施加旋转向量和平移向量
	/*
		在相机光心处建立的三维坐标系如下：
		- 1.被摄画面相对于相机向右，Tx+
		- 2.被摄画面相对于相机向下，Ty+
		- 3.被摄画面相对于相机远离，Tz+
		- 4.被摄画面下侧远离上侧接近，Rx+
		- 5.被摄画面左侧远离右侧接近，Ry+
		- 6.被摄画面顺时针旋转，Rz+
			  /z
			 /
			/________ x
			|
			|
			|y
	*/
	//- 若考虑为 被摄画面不动 并 旋转和平移相机，则为 坐标系本身的旋转和平移，坐标系内物体坐标变化（即旋转向量与平移向量）与坐标系本身的旋转平移方向相反

	if (rotationMatrix.rows != 3 || rotationMatrix.cols != 3) {
		if (m_OutputLog) {
			m_OutputLog("Rotation matrix size must be 3*3");
		}
		return -1;
	}
	pitch = atan2(rotationMatrix.at<double>(2, 1), rotationMatrix.at<double>(2, 2)) / CV_PI * 180;	//绕x轴，俯仰
	yaw = atan2(-rotationMatrix.at<double>(2, 0), sqrt(pow(rotationMatrix.at<double>(2, 1), 2) + pow(rotationMatrix.at<double>(2, 2), 2))) / CV_PI * 180;	//绕y轴，偏航
	roll = -atan2(rotationMatrix.at<double>(1, 0), rotationMatrix.at<double>(0, 0)) / CV_PI * 180;	//绕z轴，滚转

	return 0;
}

int ChessBoard_CamCalibration::GetMatrixH(std::vector<cv::Point2f> src_Points, std::vector<cv::Point2f> dest_Points, cv::Mat& matrix_H)
{
	//TODO: 根据两组平面坐标，求解单应性矩阵 H

	cv::Mat tempMatrix_H;
	tempMatrix_H.create(3, 3, CV_64FC1);

	int numPoint = src_Points.size();

	if (dest_Points.size() != numPoint) {
		if (m_OutputLog) {
			m_OutputLog("SrcPoints & DestPoints number mismatch");
		}
		return -1;
	}

	if (numPoint == 0) {
		if (m_OutputLog) {
			m_OutputLog("No SrcPoints & DestPoints");
		}
		return -1;
	}

	std::vector<cv::Point2d> src_homogeneous(numPoint);
	std::vector<cv::Point2d> dest_homogeneous(numPoint);

	//先转化为2维齐次坐标
	for (int ptCnt = 0; ptCnt < numPoint; ptCnt++) {
		src_homogeneous[ptCnt].x = src_Points[ptCnt].x;
		src_homogeneous[ptCnt].y = src_Points[ptCnt].y;

		dest_homogeneous[ptCnt].x = dest_Points[ptCnt].x;
		dest_homogeneous[ptCnt].y = dest_Points[ptCnt].y;
	}

#if 1
	//先对 src 点群归一化
	double cx = 0;	//平均值，用于平移，点群原点变为 0,0
	double cy = 0;
	for (int ptCnt = 0; ptCnt < numPoint; ptCnt++) {
		cx += src_homogeneous[ptCnt].x;
		cy += src_homogeneous[ptCnt].y;
	}
	cx /= (double)numPoint;
	cy /= (double)numPoint;
	for (int ptCnt = 0; ptCnt < numPoint; ptCnt++) {
		src_homogeneous[ptCnt].x -= cx;
		src_homogeneous[ptCnt].y -= cy;
	}
	double sx = 0;	//标准差
	double sy = 0;
	for (int ptCnt = 0; ptCnt < numPoint; ptCnt++) {
		sx += pow(src_homogeneous[ptCnt].x, 2);
		sy += pow(src_homogeneous[ptCnt].y, 2);
	}
	sx = sqrt(sx / (double)numPoint);
	sy = sqrt(sy / (double)numPoint);
	double scaleX = sqrt(2) / sx;//用于缩放，点群标准差变为 1（单方向缩放为根号 2 ）
	double scaleY = sqrt(2) / sy;
	for (int ptCnt = 0; ptCnt < numPoint; ptCnt++) {
		src_homogeneous[ptCnt].x *= scaleX;
		src_homogeneous[ptCnt].y *= scaleY;
	}
	//用于将 H 逆变换回来的矩阵
	cv::Mat matrixT_src = (cv::Mat_<double>(3, 3) <<
		scaleX, 0, -scaleX * cx,
		0, scaleY, -scaleY * cy,
		0, 0, 1);

	//再对 dis 点群归一化
	cx = 0;	//平均值，用于平移，点群原点变为 0,0
	cy = 0;
	for (int ptCnt = 0; ptCnt < numPoint; ptCnt++) {
		cx += dest_homogeneous[ptCnt].x;
		cy += dest_homogeneous[ptCnt].y;
	}
	cx /= (double)numPoint;
	cy /= (double)numPoint;
	for (int ptCnt = 0; ptCnt < numPoint; ptCnt++) {
		dest_homogeneous[ptCnt].x -= cx;
		dest_homogeneous[ptCnt].y -= cy;
	}
	sx = 0;	//标准差
	sy = 0;
	for (int ptCnt = 0; ptCnt < numPoint; ptCnt++) {
		sx += pow(dest_homogeneous[ptCnt].x, 2);
		sy += pow(dest_homogeneous[ptCnt].y, 2);
	}
	sx = sqrt(sx / (double)numPoint);
	sy = sqrt(sy / (double)numPoint);
	scaleX = sqrt(2) / sx;//用于缩放，点群标准差变为 1（单方向缩放为根号 2 ）
	scaleY = sqrt(2) / sy;
	for (int ptCnt = 0; ptCnt < numPoint; ptCnt++) {
		dest_homogeneous[ptCnt].x *= scaleX;
		dest_homogeneous[ptCnt].y *= scaleY;
	}
	//用于将 H 逆变换回来的矩阵
	cv::Mat matrixT_dis = (cv::Mat_<double>(3, 3) <<
		scaleX, 0, -scaleX * cx,
		0, scaleY, -scaleY * cy,
		0, 0, 1);
	cv::Mat matrixT_dis_invert;
	cv::invert(matrixT_dis, matrixT_dis_invert);
#endif

	//建立A矩阵并赋值
	cv::Mat matrix_A;
	matrix_A.create(2 * numPoint, 9, CV_64FC1);
	for (int ptCnt = 0; ptCnt < numPoint; ptCnt++) {
		matrix_A.at<double>(2 * ptCnt, 0) = -src_homogeneous[ptCnt].x;
		matrix_A.at<double>(2 * ptCnt, 1) = -src_homogeneous[ptCnt].y;
		matrix_A.at<double>(2 * ptCnt, 2) = -1;
		matrix_A.at<double>(2 * ptCnt, 3) = 0;
		matrix_A.at<double>(2 * ptCnt, 4) = 0;
		matrix_A.at<double>(2 * ptCnt, 5) = 0;
		matrix_A.at<double>(2 * ptCnt, 6) = src_homogeneous[ptCnt].x * dest_homogeneous[ptCnt].x;
		matrix_A.at<double>(2 * ptCnt, 7) = src_homogeneous[ptCnt].y * dest_homogeneous[ptCnt].x;
		matrix_A.at<double>(2 * ptCnt, 8) = 1 * dest_homogeneous[ptCnt].x;

		matrix_A.at<double>(2 * ptCnt + 1, 0) = 0;
		matrix_A.at<double>(2 * ptCnt + 1, 1) = 0;
		matrix_A.at<double>(2 * ptCnt + 1, 2) = 0;
		matrix_A.at<double>(2 * ptCnt + 1, 3) = -src_homogeneous[ptCnt].x;
		matrix_A.at<double>(2 * ptCnt + 1, 4) = -src_homogeneous[ptCnt].y;
		matrix_A.at<double>(2 * ptCnt + 1, 5) = -1;
		matrix_A.at<double>(2 * ptCnt + 1, 6) = src_homogeneous[ptCnt].x * dest_homogeneous[ptCnt].y;
		matrix_A.at<double>(2 * ptCnt + 1, 7) = src_homogeneous[ptCnt].y * dest_homogeneous[ptCnt].y;
		matrix_A.at<double>(2 * ptCnt + 1, 8) = 1 * dest_homogeneous[ptCnt].y;
	}

	//SVD分解，求解未知数
	cv::Mat LtL = matrix_A.t() * matrix_A;
	cv::completeSymm(LtL);
	cv::Mat Vt;
	cv::Mat W;
	cv::eigen(LtL, W, Vt);

	//求解 h 向量
	if (Vt.rows < 9) {
		if (m_OutputLog) {
			m_OutputLog("Calc Homography SVD error");
		}
		return -1;
	}
	cv::Mat h = Vt.row(8).t(); // 假设 Vt 的最后一行对应的奇异值最小

	int rowCnt = 0;
	for (int y = 0; y < tempMatrix_H.rows; y++) {
		for (int x = 0; x < tempMatrix_H.cols; x++) {
			tempMatrix_H.at<double>(y, x) = h.at<double>(rowCnt, 0);
			rowCnt++;
		}
	}

	tempMatrix_H = matrixT_dis_invert * tempMatrix_H * matrixT_src;

	//H矩阵归一化
	for (int y = 0; y < tempMatrix_H.rows; y++) {
		for (int x = 0; x < tempMatrix_H.cols; x++) {
			tempMatrix_H.at<double>(y, x) /= tempMatrix_H.at<double>(2, 2);
			if (!isnormal(tempMatrix_H.at<double>(y, x))) {
				if (m_OutputLog) {
					m_OutputLog("Invalid Homography");
				}
				return -1;
			}
		}
	}

	matrix_H = tempMatrix_H.clone();

	return 0;
}

int ChessBoard_CamCalibration::GetInitialIntrinsicParams(std::vector<cv::Mat> matrix_H, int imgWidth, int imgHeight, cv::Mat& matrix_M_init, bool bFix_u0v0Gamma)
{
	//TODO: 根据单应性矩阵求解初始内参

	int numImg = matrix_H.size();

	if (!matrix_M_init.empty()) {
		matrix_M_init.release();
	}

	//1. 建立v矩阵和b矩阵
	cv::Mat matrix_v;

	double fx, fy, u0, v0, gamma, ramda;	//ramda 为尺度因子

	if (bFix_u0v0Gamma)	//固定 u0/v0 为图像中心，传感器坐标轴倾斜尺度 gamma 等于 0（简化计算）
	{
		/*
		此时 B 矩阵简化：
		1. gamma 等于 0，即 B12 = B22 = 0
		2. u0/v0 已知，即 B13 = B31 = -u0*B11, B23 = B32 = -v0*B22
		3. 即
		| ramda/(fx^2),		0,					-ramda*u0/(fx^2),								|		| B11,		0,		-u0B11,	|
		B = ramda * (M^-1)^T * M^-1 =	| 0,				ramda/(fy^2),		-ramda*v0/(fy^2),								| =		| 0,		B22,	-v0B22, |
		| -ramda*u0/(fx^2),	-ramda*v0/(fy^2),	ramda*(u0^2)/(fx^2)+ramda*(v0^2)/(fy^2)+ramda,	|		| -u0B11,	-v0B22,	B33,	|
		4. 化简 hi^T * B * hj，即
		| hi1*hj1-u0*(hi3*hj1+hi1*hj3)	|		| B11	|
		hi^T * B * hj = vij^T * b =	| hi2*hj2-v0*(hi3*hj2+hi2*hj3)	| ^ T *	| B22	|
		| hi3*hj3						|		| B33	|
		5. 根据 n 张图的下式，求解 b（一张图->两个方程，该 case 三个未知数至少需要两张图，且需存在与x0y面非共面的图）
		| v12^T			|
		| v11^T - v22^T | * b = 0
		*/
		matrix_v.create(2 * numImg, 3, CV_64FC1);

		u0 = (double)(imgWidth - 1) / 2.0;
		v0 = (double)(imgHeight - 1) / 2.0;

		for (int imgCnt = 0; imgCnt < numImg; imgCnt++) {
			matrix_v.at<double>(2 * imgCnt, 0) = matrix_H[imgCnt].at<double>(0, 0) * matrix_H[imgCnt].at<double>(0, 1) - u0 * (matrix_H[imgCnt].at<double>(2, 0) * matrix_H[imgCnt].at<double>(0, 1) + matrix_H[imgCnt].at<double>(0, 0) * matrix_H[imgCnt].at<double>(2, 1));
			matrix_v.at<double>(2 * imgCnt, 1) = matrix_H[imgCnt].at<double>(1, 0) * matrix_H[imgCnt].at<double>(1, 1) - v0 * (matrix_H[imgCnt].at<double>(2, 0) * matrix_H[imgCnt].at<double>(1, 1) + matrix_H[imgCnt].at<double>(1, 0) * matrix_H[imgCnt].at<double>(2, 1));
			matrix_v.at<double>(2 * imgCnt, 2) = matrix_H[imgCnt].at<double>(2, 0) * matrix_H[imgCnt].at<double>(2, 1);

			matrix_v.at<double>(2 * imgCnt + 1, 0) = matrix_H[imgCnt].at<double>(0, 0) * matrix_H[imgCnt].at<double>(0, 0) - u0 * (matrix_H[imgCnt].at<double>(2, 0) * matrix_H[imgCnt].at<double>(0, 0) + matrix_H[imgCnt].at<double>(0, 0) * matrix_H[imgCnt].at<double>(2, 0));
			matrix_v.at<double>(2 * imgCnt + 1, 1) = matrix_H[imgCnt].at<double>(1, 0) * matrix_H[imgCnt].at<double>(1, 0) - v0 * (matrix_H[imgCnt].at<double>(2, 0) * matrix_H[imgCnt].at<double>(1, 0) + matrix_H[imgCnt].at<double>(1, 0) * matrix_H[imgCnt].at<double>(2, 0));
			matrix_v.at<double>(2 * imgCnt + 1, 2) = matrix_H[imgCnt].at<double>(2, 0) * matrix_H[imgCnt].at<double>(2, 0);

			matrix_v.at<double>(2 * imgCnt + 1, 0) -= matrix_H[imgCnt].at<double>(0, 1) * matrix_H[imgCnt].at<double>(0, 1) - u0 * (matrix_H[imgCnt].at<double>(2, 1) * matrix_H[imgCnt].at<double>(0, 1) + matrix_H[imgCnt].at<double>(0, 1) * matrix_H[imgCnt].at<double>(2, 1));
			matrix_v.at<double>(2 * imgCnt + 1, 1) -= matrix_H[imgCnt].at<double>(1, 1) * matrix_H[imgCnt].at<double>(1, 1) - v0 * (matrix_H[imgCnt].at<double>(2, 1) * matrix_H[imgCnt].at<double>(1, 1) + matrix_H[imgCnt].at<double>(1, 1) * matrix_H[imgCnt].at<double>(2, 1));
			matrix_v.at<double>(2 * imgCnt + 1, 2) -= matrix_H[imgCnt].at<double>(2, 1) * matrix_H[imgCnt].at<double>(2, 1);
		}

		//SVD分解，求解未知数
		cv::Mat LtL = matrix_v.t() * matrix_v;
		cv::completeSymm(LtL);
		cv::Mat Vt;
		cv::Mat W;
		cv::eigen(LtL, W, Vt);

		//求解 b 向量
		if (Vt.rows < 3) {
			if (m_OutputLog) {
				m_OutputLog("Calc Intrinsic Params SVD error");
			}
			return -1;
		}
		cv::Mat matrix_b = Vt.row(2).t(); // 假设 Vt 的最后一行对应的奇异值最小

		double B11 = matrix_b.at<double>(0, 0);
		double B22 = matrix_b.at<double>(1, 0);
		double B33 = matrix_b.at<double>(2, 0);

		//2. 获取初始内参
		ramda = B33 - u0 * u0 * B11 - v0 * v0 * B22;
		fx = sqrt(ramda / B11);
		fy = sqrt(ramda / B22);
		gamma = 0;

	}
	else	//所有值非固定，传感器坐标轴倾斜尺度 gamma 不等于 0
	{
		/*
		一张图->两个方程，该 case 六个未知数至少需要三张图，且需存在与x0y面非共面的图
		| v12^T			|
		| v11^T - v22^T | * b = 0
		*/
		matrix_v.create(2 * numImg, 6, CV_64FC1);

		for (int imgCnt = 0; imgCnt < numImg; imgCnt++) {
			matrix_v.at<double>(2 * imgCnt, 0) = matrix_H[imgCnt].at<double>(0, 0) * matrix_H[imgCnt].at<double>(0, 1);
			matrix_v.at<double>(2 * imgCnt, 1) = matrix_H[imgCnt].at<double>(0, 0) * matrix_H[imgCnt].at<double>(1, 1) + matrix_H[imgCnt].at<double>(1, 0) * matrix_H[imgCnt].at<double>(0, 1);
			matrix_v.at<double>(2 * imgCnt, 2) = matrix_H[imgCnt].at<double>(1, 0) * matrix_H[imgCnt].at<double>(1, 1);
			matrix_v.at<double>(2 * imgCnt, 3) = matrix_H[imgCnt].at<double>(2, 0) * matrix_H[imgCnt].at<double>(0, 1) + matrix_H[imgCnt].at<double>(0, 0) * matrix_H[imgCnt].at<double>(2, 1);
			matrix_v.at<double>(2 * imgCnt, 4) = matrix_H[imgCnt].at<double>(2, 0) * matrix_H[imgCnt].at<double>(1, 1) + matrix_H[imgCnt].at<double>(1, 0) * matrix_H[imgCnt].at<double>(2, 1);
			matrix_v.at<double>(2 * imgCnt, 5) = matrix_H[imgCnt].at<double>(2, 0) * matrix_H[imgCnt].at<double>(2, 1);

			matrix_v.at<double>(2 * imgCnt + 1, 0) = matrix_H[imgCnt].at<double>(0, 0) * matrix_H[imgCnt].at<double>(0, 0);
			matrix_v.at<double>(2 * imgCnt + 1, 1) = matrix_H[imgCnt].at<double>(0, 0) * matrix_H[imgCnt].at<double>(1, 0) + matrix_H[imgCnt].at<double>(1, 0) * matrix_H[imgCnt].at<double>(0, 0);
			matrix_v.at<double>(2 * imgCnt + 1, 2) = matrix_H[imgCnt].at<double>(1, 0) * matrix_H[imgCnt].at<double>(1, 0);
			matrix_v.at<double>(2 * imgCnt + 1, 3) = matrix_H[imgCnt].at<double>(2, 0) * matrix_H[imgCnt].at<double>(0, 0) + matrix_H[imgCnt].at<double>(0, 0) * matrix_H[imgCnt].at<double>(2, 0);
			matrix_v.at<double>(2 * imgCnt + 1, 4) = matrix_H[imgCnt].at<double>(2, 0) * matrix_H[imgCnt].at<double>(1, 0) + matrix_H[imgCnt].at<double>(1, 0) * matrix_H[imgCnt].at<double>(2, 0);
			matrix_v.at<double>(2 * imgCnt + 1, 5) = matrix_H[imgCnt].at<double>(2, 0) * matrix_H[imgCnt].at<double>(2, 0);

			matrix_v.at<double>(2 * imgCnt + 1, 0) -= matrix_H[imgCnt].at<double>(0, 1) * matrix_H[imgCnt].at<double>(0, 1);
			matrix_v.at<double>(2 * imgCnt + 1, 1) -= matrix_H[imgCnt].at<double>(0, 1) * matrix_H[imgCnt].at<double>(1, 1) + matrix_H[imgCnt].at<double>(1, 1) * matrix_H[imgCnt].at<double>(0, 1);
			matrix_v.at<double>(2 * imgCnt + 1, 2) -= matrix_H[imgCnt].at<double>(1, 1) * matrix_H[imgCnt].at<double>(1, 1);
			matrix_v.at<double>(2 * imgCnt + 1, 3) -= matrix_H[imgCnt].at<double>(2, 1) * matrix_H[imgCnt].at<double>(0, 1) + matrix_H[imgCnt].at<double>(0, 1) * matrix_H[imgCnt].at<double>(2, 1);
			matrix_v.at<double>(2 * imgCnt + 1, 4) -= matrix_H[imgCnt].at<double>(2, 1) * matrix_H[imgCnt].at<double>(1, 1) + matrix_H[imgCnt].at<double>(1, 1) * matrix_H[imgCnt].at<double>(2, 1);
			matrix_v.at<double>(2 * imgCnt + 1, 5) -= matrix_H[imgCnt].at<double>(2, 1) * matrix_H[imgCnt].at<double>(2, 1);
		}

		//SVD分解，求解未知数
		cv::Mat LtL = matrix_v.t() * matrix_v;
		cv::completeSymm(LtL);
		cv::Mat Vt;
		cv::Mat W;
		cv::eigen(LtL, W, Vt);

		//求解 b 向量
		if (Vt.rows < 6) {
			if (m_OutputLog) {
				m_OutputLog("Calc Intrinsic Params SVD error");
			}
			return -1;
		}
		cv::Mat matrix_b = Vt.row(5).t(); // 假设 Vt 的最后一行对应的奇异值最小

		double B11 = matrix_b.at<double>(0, 0);
		double B12 = matrix_b.at<double>(1, 0);
		double B22 = matrix_b.at<double>(2, 0);
		double B13 = matrix_b.at<double>(3, 0);
		double B23 = matrix_b.at<double>(4, 0);
		double B33 = matrix_b.at<double>(5, 0);

		//2. 获取初始内参
		v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12);
		ramda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
		fx = sqrt(ramda / B11);
		fy = sqrt(ramda * B11 / (B11 * B22 - B12 * B12));
		gamma = -1 * B12 * fx * fx * fy / ramda;
		u0 = gamma * v0 / fy - B13 * fx * fx / ramda;
	}

	if (!isnormal(fx)) {
		if (m_OutputLog) {
			m_OutputLog("Invalid Intrinsic Param: fx");
		}
		return -1;
	}
	if (!isnormal(fy)) {
		if (m_OutputLog) {
			m_OutputLog("Invalid Intrinsic Param: fx");
		}
		return -1;
	}
	if (!isnormal(u0) && u0 != 0) {
		if (m_OutputLog) {
			m_OutputLog("Invalid Intrinsic Param: u0");
		}
		return -1;
	}
	if (!isnormal(v0) && v0 != 0) {
		if (m_OutputLog) {
			m_OutputLog("Invalid Intrinsic Param: v0");
		}
		return -1;
	}
	if (!isnormal(gamma) && gamma != 0) {
		if (m_OutputLog) {
			m_OutputLog("Invalid Intrinsic Param: gamma");
		}
		return -1;
	}

	matrix_M_init = (cv::Mat_<double>(3, 3) <<
		fx, gamma, u0,
		0, fy, v0,
		0, 0, 1);

	return 0;
}

int ChessBoard_CamCalibration::GetInitialExtrinsicParams(std::vector<cv::Mat> matrix_H, cv::Mat matrix_M, std::vector<cv::Mat>& vec_R_init, std::vector<cv::Mat>& vec_T_init)
{
	//TODO: 根据单应性矩阵和内参求解初始外参

	int numImg = matrix_H.size();

	cv::Mat matrix_M_invert;
	cv::invert(matrix_M, matrix_M_invert);

	vec_R_init.clear();
	vec_T_init.clear();

	for (int imgCnt = 0; imgCnt < numImg; imgCnt++) {
		cv::Mat vec_h1 = (cv::Mat_<double>(3, 1) <<
			matrix_H[imgCnt].at<double>(0, 0), matrix_H[imgCnt].at<double>(1, 0), matrix_H[imgCnt].at<double>(2, 0));
		cv::Mat vec_h2 = (cv::Mat_<double>(3, 1) <<
			matrix_H[imgCnt].at<double>(0, 1), matrix_H[imgCnt].at<double>(1, 1), matrix_H[imgCnt].at<double>(2, 1));
		cv::Mat vec_h3 = (cv::Mat_<double>(3, 1) <<
			matrix_H[imgCnt].at<double>(0, 2), matrix_H[imgCnt].at<double>(1, 2), matrix_H[imgCnt].at<double>(2, 2));

		//ramda2 为尺度因子，旋转矩阵性质：子向量 ||ri|| = 1
		double ramda2 = (1 / cv::norm(matrix_M_invert * vec_h1) + 1 / cv::norm(matrix_M_invert * vec_h2)) / 2;

		cv::Mat vec_r1 = ramda2 * matrix_M_invert * vec_h1;
		cv::Mat vec_r2 = ramda2 * matrix_M_invert * vec_h2;
		cv::Mat vec_r3 = vec_r1.cross(vec_r2);
		cv::Mat vec_t = ramda2 * matrix_M_invert * vec_h3;

		for (int i = 0; i < 3; i++) {
			if (!isnormal(vec_r1.at<double>(i, 0)) && vec_r1.at<double>(i, 0) != 0) {
				if (m_OutputLog) {
					m_OutputLog("Invalid Extrinsic Param: vec_r1");
				}
				return -1;
			}
			if (!isnormal(vec_r2.at<double>(i, 0)) && vec_r2.at<double>(i, 0) != 0) {
				if (m_OutputLog) {
					m_OutputLog("Invalid Extrinsic Param: vec_r2");
				}
				return -1;
			}
			if (!isnormal(vec_r3.at<double>(i, 0)) && vec_r3.at<double>(i, 0) != 0) {
				if (m_OutputLog) {
					m_OutputLog("Invalid Extrinsic Param: vec_r3");
				}
				return -1;
			}
			if (!isnormal(vec_t.at<double>(i, 0)) && vec_t.at<double>(i, 0) != 0) {
				if (m_OutputLog) {
					m_OutputLog("Invalid Extrinsic Param: vec_t");
				}
				return -1;
			}
		}

		cv::Mat tempMatrix_R = (cv::Mat_<double>(3, 3) <<
			vec_r1.at<double>(0, 0), vec_r2.at<double>(0, 0), vec_r3.at<double>(0, 0),
			vec_r1.at<double>(1, 0), vec_r2.at<double>(1, 0), vec_r3.at<double>(1, 0),
			vec_r1.at<double>(2, 0), vec_r2.at<double>(2, 0), vec_r3.at<double>(2, 0));

		cv::Mat tempVec_R;
		cv::Rodrigues(tempMatrix_R, tempVec_R);

		vec_R_init.push_back(tempVec_R);
		vec_T_init.push_back(vec_t);
	}

	return 0;
}

int ChessBoard_CamCalibration::OptimizeSingleCameraParams_LevMarq(std::vector<std::vector<cv::Point3f>> objectPoints, std::vector<std::vector<cv::Point2f>> imagePoints,
	cv::Mat& matrix_M, bool bFixMatrixM, cv::Mat& matrix_K, cv::Mat& matrix_P, std::vector<cv::Mat>& vec_R, std::vector<cv::Mat>& vec_T, std::vector<double>& error, double& avgError)
{
	//TODO: LM 算法优化相机参数

	int numImg = objectPoints.size();

	if (imagePoints.size() != numImg ||
		vec_R.size() != numImg ||
		vec_T.size() != numImg) {
		if (m_OutputLog) {
			m_OutputLog("Group size mismatch: Object points & Image points & Rotation vector & Translation vector");
		}
		return -1;
	}

	if (matrix_M.rows != 3 || matrix_M.cols != 3) {
		if (m_OutputLog) {
			m_OutputLog("Intrinsic matrix size must be 3*3");
		}
		return -1;
	}

#if DISTORTION_ACCURACY == 1
	if (matrix_K.rows != 1 || matrix_K.cols != 6) {
		if (m_OutputLog) {
			m_OutputLog("Radial Distortion matrix size must be 1*6");
		}
		return -1;
	}
#else
	if (matrix_K.rows != 1 || matrix_K.cols != 3) {
		if (m_OutputLog) {
			m_OutputLog("Radial Distortion matrix size must be 1*3");
		}
		return -1;
	}
#endif

	if (matrix_P.rows != 1 || matrix_P.cols != 2) {
		if (m_OutputLog) {
			m_OutputLog("Tangential Distortion matrix size must be 1*3");
		}
		return -1;
	}

	for (int imgCnt = 0; imgCnt < numImg; imgCnt++) {
		if (vec_R[imgCnt].rows != 3 || vec_R[imgCnt].cols != 1) {
			if (m_OutputLog) {
				m_OutputLog("Rotation vector size must be 3*1");
			}
			return -1;
		}
		if (vec_T[imgCnt].rows != 3 || vec_T[imgCnt].cols != 1) {
			if (m_OutputLog) {
				m_OutputLog("Translation vector size must be 3*3");
			}
			return -1;
		}
	}

	error.clear();
	error.resize(numImg);


	//1. 创建 LM 算法
	CvLevMarq solver;

	int paramNum_M, paramNum_K, paramNum_P;

	paramNum_M = 4;
#if DISTORTION_ACCURACY == 1
	paramNum_K = 6;
#else
	paramNum_K = 3;
#endif
	paramNum_P = 2;

	int paramNum_TotalIntrinsic = paramNum_M + paramNum_K + paramNum_P;

	//4个内参，6 or 3个径向畸变，2个切向畸变，(3个旋转向量元素，3个平移向量元素)*N张图
	solver.init(paramNum_TotalIntrinsic + numImg * 6, 0, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, DBL_EPSILON));

	double* param_init = solver.param->data.db;
	uchar* mask = solver.mask->data.ptr;

	param_init[0] = matrix_M.at<double>(0, 0);
	param_init[1] = matrix_M.at<double>(1, 1);
	param_init[2] = matrix_M.at<double>(0, 2);
	param_init[3] = matrix_M.at<double>(1, 2);

	if (bFixMatrixM) {
		mask[0] = mask[1] = 0;
		mask[2] = mask[3] = 0;
	}

	param_init[paramNum_M + 0] = matrix_K.at<double>(0, 0);
	param_init[paramNum_M + 1] = matrix_K.at<double>(0, 1);
	param_init[paramNum_M + 2] = matrix_K.at<double>(0, 2);
#if DISTORTION_ACCURACY == 1
	param_init[paramNum_M + 3] = matrix_K.at<double>(0, 3);
	param_init[paramNum_M + 4] = matrix_K.at<double>(0, 4);
	param_init[paramNum_M + 5] = matrix_K.at<double>(0, 5);
#endif

	param_init[paramNum_M + paramNum_K + 0] = matrix_P.at<double>(0, 0);
	param_init[paramNum_M + paramNum_K + 1] = matrix_P.at<double>(0, 0);

	for (int imgCnt = 0; imgCnt < numImg; imgCnt++) {
		param_init[paramNum_TotalIntrinsic + imgCnt * 6] = vec_R[imgCnt].at<double>(0, 0);
		param_init[paramNum_TotalIntrinsic + imgCnt * 6 + 1] = vec_R[imgCnt].at<double>(1, 0);
		param_init[paramNum_TotalIntrinsic + imgCnt * 6 + 2] = vec_R[imgCnt].at<double>(2, 0);

		param_init[paramNum_TotalIntrinsic + imgCnt * 6 + 3] = vec_T[imgCnt].at<double>(0, 0);
		param_init[paramNum_TotalIntrinsic + imgCnt * 6 + 4] = vec_T[imgCnt].at<double>(1, 0);
		param_init[paramNum_TotalIntrinsic + imgCnt * 6 + 5] = vec_T[imgCnt].at<double>(2, 0);
	}

	//2. LM 迭代
	int solverCnt = 0;
	double reprojErr;

	while (1)
	{
		const CvMat* _param = 0;
		CvMat* _JtJ = 0, *_JtErr = 0;
		double* _errNorm = 0;

		bool proceed = solver.updateAlt(_param, _JtJ, _JtErr, _errNorm);
		double* param = solver.param->data.db;

		if (!proceed)
			break;

		reprojErr = 0;

		for (int imgCnt = 0; imgCnt < numImg; imgCnt++)
		{
			int numPoint = objectPoints[imgCnt].size();

			cv::Mat cur_matrix_M = (cv::Mat_<double>(3, 3) <<
				param[0], 0, param[2],
				0, param[1], param[3],
				0, 0, 1);
#if DISTORTION_ACCURACY == 1
			cv::Mat cur_matrix_K = (cv::Mat_<double>(1, 6) << param[paramNum_M + 0], param[paramNum_M + 1], param[paramNum_M + 2],
				param[paramNum_M + 3], param[paramNum_M + 4], param[paramNum_M + 5]);
#else
			cv::Mat cur_matrix_K = (cv::Mat_<double>(1, 3) << param[paramNum_M + 0], param[paramNum_M + 1], param[paramNum_M + 2]);
#endif
			cv::Mat cur_matrix_P = (cv::Mat_<double>(1, 2) << param[paramNum_M + paramNum_K + 0], param[paramNum_M + paramNum_K + 1]);
			cv::Mat cur_vec_R = (cv::Mat_<double>(3, 1) <<
				param[paramNum_TotalIntrinsic + imgCnt * 6],
				param[paramNum_TotalIntrinsic + imgCnt * 6 + 1],
				param[paramNum_TotalIntrinsic + imgCnt * 6 + 2]);
			cv::Mat cur_vec_T = (cv::Mat_<double>(3, 1) <<
				param[paramNum_TotalIntrinsic + imgCnt * 6 + 3],
				param[paramNum_TotalIntrinsic + imgCnt * 6 + 4],
				param[paramNum_TotalIntrinsic + imgCnt * 6 + 5]);

			cv::Mat matrix_Ji, matrix_Je, vec_err;
			cv::Mat dpdf, dpdc, dpdk, dpdp, dpdr, dpdt;

			matrix_Ji.create(numPoint * 2, paramNum_TotalIntrinsic, CV_64FC1);
			matrix_Je.create(numPoint * 2, 6, CV_64FC1);
			vec_err.create(numPoint * 2, 1, CV_64FC1);

			dpdf = matrix_Ji(cv::Range::all(), cv::Range(0, 2));
			dpdc = matrix_Ji(cv::Range::all(), cv::Range(2, 4));
			dpdk = matrix_Ji(cv::Range::all(), cv::Range(4, paramNum_M + paramNum_K));
			dpdp = matrix_Ji(cv::Range::all(), cv::Range(paramNum_M + paramNum_K, paramNum_TotalIntrinsic));
			dpdr = matrix_Je(cv::Range::all(), cv::Range(0, 3));
			dpdt = matrix_Je(cv::Range::all(), cv::Range(3, 6));

			std::vector<cv::Point2f> projectionPoints;

			//CvMat部分
			CvMat _part;

			if (_JtJ || _JtErr) {
				ProjectPoints(objectPoints[imgCnt], cur_matrix_M, cur_matrix_K, cur_matrix_P, cur_vec_R, cur_vec_T, projectionPoints,
					&dpdf, &dpdc, &dpdk, &dpdp, &dpdr, &dpdt);
			}
			else {
				ProjectPoints(objectPoints[imgCnt], cur_matrix_M, cur_matrix_K, cur_matrix_P, cur_vec_R, cur_vec_T, projectionPoints);
			}

			for (int ptCnt = 0; ptCnt < numPoint; ptCnt++) {
				vec_err.at<double>(2 * ptCnt, 0) = projectionPoints[ptCnt].x - imagePoints[imgCnt][ptCnt].x;
				vec_err.at<double>(2 * ptCnt + 1, 0) = projectionPoints[ptCnt].y - imagePoints[imgCnt][ptCnt].y;
			}

			if (_JtJ || _JtErr)
			{
				cvGetSubRect(_JtJ, &_part, cvRect(0, 0, paramNum_TotalIntrinsic, paramNum_TotalIntrinsic));
				cv::Mat JiT_Ji_before = cv::cvarrToMat(&_part, true);
				cv::Mat matrix_JiT = matrix_Ji.t();
				cv::Mat JiT_Ji = matrix_JiT * matrix_Ji + JiT_Ji_before;
				cv::Ptr<CvMat> p_JiT_Ji;
				p_JiT_Ji = cvCloneMat(&_part);
				double* JiT_Ji_p = p_JiT_Ji->data.db;
				for (int y = 0; y < JiT_Ji.rows; y++) {
					for (int x = 0; x < JiT_Ji.cols; x++) {
						*JiT_Ji_p = JiT_Ji.at<double>(y, x);
						JiT_Ji_p++;
					}
				}
				cvConvert(p_JiT_Ji, &_part);

				cvGetSubRect(_JtJ, &_part, cvRect(paramNum_TotalIntrinsic + imgCnt * 6, paramNum_TotalIntrinsic + imgCnt * 6, 6, 6));
				cv::Mat matrix_JeT = matrix_Je.t();
				cv::Mat JeT_Je = matrix_JeT * matrix_Je;
				cv::Ptr<CvMat> p_JeT_Je;
				p_JeT_Je = cvCloneMat(&_part);
				double* JeT_Je_p = p_JeT_Je->data.db;
				for (int y = 0; y < JeT_Je.rows; y++) {
					for (int x = 0; x < JeT_Je.cols; x++) {
						*JeT_Je_p = JeT_Je.at<double>(y, x);
						JeT_Je_p++;
					}
				}
				cvConvert(p_JeT_Je, &_part);

				cvGetSubRect(_JtJ, &_part, cvRect(paramNum_TotalIntrinsic + imgCnt * 6, 0, 6, paramNum_TotalIntrinsic));
				cv::Mat JiT_Je = matrix_JiT * matrix_Je;
				cv::Ptr<CvMat> p_JiT_Je;
				p_JiT_Je = cvCloneMat(&_part);
				double* JiT_Je_p = p_JiT_Je->data.db;
				for (int y = 0; y < JiT_Je.rows; y++) {
					for (int x = 0; x < JiT_Je.cols; x++) {
						*JiT_Je_p = JiT_Je.at<double>(y, x);
						JiT_Je_p++;
					}
				}
				cvConvert(p_JiT_Je, &_part);

				cvGetRows(_JtErr, &_part, 0, paramNum_TotalIntrinsic);
				cv::Mat JiT_err_before = cv::cvarrToMat(&_part, true);
				cv::Mat JiT_err = matrix_JiT * vec_err + JiT_err_before;
				cv::Ptr<CvMat> p_JiT_err;
				p_JiT_err = cvCloneMat(&_part);
				double* JiT_err_p = p_JiT_err->data.db;
				for (int y = 0; y < JiT_err.rows; y++) {
					for (int x = 0; x < JiT_err.cols; x++) {
						*JiT_err_p = JiT_err.at<double>(y, x);
						JiT_err_p++;
					}
				}
				cvConvert(p_JiT_err, &_part);

				cvGetRows(_JtErr, &_part, paramNum_TotalIntrinsic + imgCnt * 6, paramNum_TotalIntrinsic + (imgCnt + 1) * 6);
				cv::Mat JeT_err_before = cv::cvarrToMat(&_part, true);
				cv::Mat JeT_err = matrix_JeT * vec_err + JeT_err_before;
				cv::Ptr<CvMat> p_JeT_err;
				p_JeT_err = cvCloneMat(&_part);
				double* JeT_err_p = p_JeT_err->data.db;
				for (int y = 0; y < JeT_err.rows; y++) {
					for (int x = 0; x < JeT_err.cols; x++) {
						*JeT_err_p = JeT_err.at<double>(y, x);
						JeT_err_p++;
					}
				}
				cvConvert(p_JeT_err, &_part);
			}

			double errNorm = cv::norm(vec_err, cv::NORM_L2);
			reprojErr += errNorm * errNorm;

			error[imgCnt] = errNorm / sqrt((double)numPoint);
		}
		if (_errNorm)
			*_errNorm = reprojErr;

		solverCnt++;
	}

	double* param_final = solver.param->data.db;

	matrix_M.release();
	matrix_M = (cv::Mat_<double>(3, 3) <<
		param_final[0], 0, param_final[2],
		0, param_final[1], param_final[3],
		0, 0, 1);

	matrix_K.release();
#if DISTORTION_ACCURACY == 1
	matrix_K = (cv::Mat_<double>(1, 6) << param_final[paramNum_M + 0], param_final[paramNum_M + 1], param_final[paramNum_M + 2],
		param_final[paramNum_M + 3], param_final[paramNum_M + 4], param_final[paramNum_M + 5]);
#else
	matrix_K = (cv::Mat_<double>(1, 3) << param_final[paramNum_M + 0], param_final[paramNum_M + 1], param_final[paramNum_M + 2]);
#endif

	matrix_P.release();
	matrix_P = (cv::Mat_<double>(1, 2) << param_final[paramNum_M + paramNum_K + 0], param_final[paramNum_M + paramNum_K + 1]);

	for (int imgCnt = 0; imgCnt < numImg; imgCnt++)
	{
		vec_R[imgCnt].release();
		vec_R[imgCnt] = (cv::Mat_<double>(3, 1) <<
			param_final[paramNum_TotalIntrinsic + imgCnt * 6],
			param_final[paramNum_TotalIntrinsic + imgCnt * 6 + 1],
			param_final[paramNum_TotalIntrinsic + imgCnt * 6 + 2]);

		vec_T[imgCnt].release();
		vec_T[imgCnt] = (cv::Mat_<double>(3, 1) <<
			param_final[paramNum_TotalIntrinsic + imgCnt * 6 + 3],
			param_final[paramNum_TotalIntrinsic + imgCnt * 6 + 4],
			param_final[paramNum_TotalIntrinsic + imgCnt * 6 + 5]);

		avgError += error[imgCnt];
	}

	avgError /= numImg;

	return 0;
}

int ChessBoard_CamCalibration::ProjectPoints(std::vector<cv::Point3f> objectPoints, cv::Mat matrix_M, cv::Mat matrix_K, cv::Mat matrix_P, cv::Mat vec_R, cv::Mat vec_T, std::vector<cv::Point2f>& projectionPoints,
	cv::Mat* dpdf, cv::Mat* dpdc, cv::Mat* dpdk, cv::Mat* dpdp, cv::Mat* dpdr, cv::Mat* dpdt)
{
	int numPoint = objectPoints.size();

	if (!projectionPoints.empty()) {
		projectionPoints.clear();
	}

	if (matrix_M.rows != 3 || matrix_M.cols != 3) {
		if (m_OutputLog) {
			m_OutputLog("Intrinsic matrix size must be 3*3");
		}
		return -1;
	}

#if DISTORTION_ACCURACY == 1
	if (matrix_K.rows != 1 || matrix_K.cols != 6) {
		if (m_OutputLog) {
			m_OutputLog("Radial Distortion matrix size must be 1*6");
		}
		return -1;
	}
#else
	if (matrix_K.rows != 1 || matrix_K.cols != 3) {
		if (m_OutputLog) {
			m_OutputLog("Radial Distortion matrix size must be 1*3");
		}
		return -1;
	}
#endif

	if (matrix_P.rows != 1 || matrix_P.cols != 2) {
		if (m_OutputLog) {
			m_OutputLog("Tangential Distortion matrix size must be 1*3");
		}
		return -1;
	}

	if (vec_R.rows != 3 || vec_R.cols != 1) {
		if (m_OutputLog) {
			m_OutputLog("Rotation vector size must be 3*1");
		}
		return -1;
	}

	if (vec_T.rows != 3 || vec_T.cols != 1) {
		if (m_OutputLog) {
			m_OutputLog("Translation vector size must be 3*3");
		}
		return -1;
	}

	if (dpdf) {
		if (dpdf->empty()) {
			dpdf->create(numPoint * 2, 2, CV_64FC1);
		}
		else {
			if ((dpdf->rows != numPoint * 2) || (dpdf->cols != 2) || (dpdf->type() != CV_64FC1)) {
				if (m_OutputLog) {
					m_OutputLog("dpdf matrix size must be 2N*2");
				}
				return -1;
			}
		}
	}
	if (dpdc) {
		if (dpdc->empty()) {
			dpdc->create(numPoint * 2, 2, CV_64FC1);
		}
		else {
			if ((dpdc->rows != numPoint * 2) || (dpdc->cols != 2) || (dpdc->type() != CV_64FC1)) {
				if (m_OutputLog) {
					m_OutputLog("dpdc matrix size must be 2N*2");
				}
				return -1;
			}
		}
	}
	if (dpdk) {
		if (dpdk->empty()) {
#if DISTORTION_ACCURACY == 1
			dpdk->create(numPoint * 2, 6, CV_64FC1);
#else
			dpdk->create(numPoint * 2, 3, CV_64FC1);
#endif
		}
		else {
#if DISTORTION_ACCURACY == 1
			if ((dpdk->rows != numPoint * 2) || (dpdk->cols != 6) || (dpdk->type() != CV_64FC1)) {
				if (m_OutputLog) {
					m_OutputLog("dpdk matrix size must be 2N*6");
				}
				return -1;
			}
#else
			if ((dpdk->rows != numPoint * 2) || (dpdk->cols != 3) || (dpdk->type() != CV_64FC1)) {
				if (m_OutputLog) {
					m_OutputLog("dpdk matrix size must be 2N*3");
				}
				return -1;
			}
#endif
		}
	}
	if (dpdp) {
		if (dpdp->empty()) {
			dpdp->create(numPoint * 2, 2, CV_64FC1);
		}
		else {
			if ((dpdp->rows != numPoint * 2) || (dpdp->cols != 2) || (dpdp->type() != CV_64FC1)) {
				if (m_OutputLog) {
					m_OutputLog("dpdp matrix size must be 2N*2");
				}
				return -1;
			}
		}
	}
	if (dpdr) {
		if (dpdr->empty()) {
			dpdr->create(numPoint * 2, 3, CV_64FC1);
		}
		else {
			if ((dpdr->rows != numPoint * 2) || (dpdr->cols != 3) || (dpdr->type() != CV_64FC1)) {
				if (m_OutputLog) {
					m_OutputLog("dpdr matrix size must be 2N*3");
				}
				return -1;
			}
		}
	}
	if (dpdt) {
		if (dpdt->empty()) {
			dpdt->create(numPoint * 2, 3, CV_64FC1);
		}
		else {
			if ((dpdt->rows != numPoint * 2) || (dpdt->cols != 3) || (dpdt->type() != CV_64FC1)) {
				if (m_OutputLog) {
					m_OutputLog("dpdt matrix size must be 2N*3");
				}
				return -1;
			}
		}
	}

	cv::Mat matrix_R;
	cv::Mat jacobian_R;
	jacobian_R.create(3, 9, CV_64FC1);
	cv::Rodrigues(vec_R, matrix_R, jacobian_R);
	/*
	旋转向量 转换为 旋转矩阵 时，雅可比矩阵为 3 * 9
	| dR11dr1, dR12dr1, dR13dr1, dR21dr1, dR22dr1, dR23dr1, dR31dr1, dR32dr1, dR33dr1,  |
	| dR11dr2, dR12dr2, dR13dr2, dR21dr2, dR22dr2, dR23dr2, dR31dr2, dR32dr2, dR33dr2,  |
	| dR11dr3, dR12dr3, dR13dr3, dR21dr3, dR22dr3, dR23dr3, dR31dr3, dR32dr3, dR33dr3,  |
	*/

	double fx = matrix_M.at<double>(0, 0);
	double fy = matrix_M.at<double>(1, 1);
	double u0 = matrix_M.at<double>(0, 2);
	double v0 = matrix_M.at<double>(1, 2);

	double k1 = matrix_K.at<double>(0, 0);
	double k2 = matrix_K.at<double>(0, 1);
	double k3 = matrix_K.at<double>(0, 2);
#if DISTORTION_ACCURACY == 1
	double k4 = matrix_K.at<double>(0, 3);
	double k5 = matrix_K.at<double>(0, 4);
	double k6 = matrix_K.at<double>(0, 5);
#endif

	double p1 = matrix_P.at<double>(0, 0);
	double p2 = matrix_P.at<double>(0, 1);

	double R11 = matrix_R.at<double>(0, 0);
	double R12 = matrix_R.at<double>(0, 1);
	double R13 = matrix_R.at<double>(0, 2);
	double R21 = matrix_R.at<double>(1, 0);
	double R22 = matrix_R.at<double>(1, 1);
	double R23 = matrix_R.at<double>(1, 2);
	double R31 = matrix_R.at<double>(2, 0);
	double R32 = matrix_R.at<double>(2, 1);
	double R33 = matrix_R.at<double>(2, 2);

	double t1 = vec_T.at<double>(0, 0);
	double t2 = vec_T.at<double>(1, 0);
	double t3 = vec_T.at<double>(2, 0);

	for (int ptCnt = 0; ptCnt < numPoint; ptCnt++) {
		double x_w = objectPoints[ptCnt].x;
		double y_w = objectPoints[ptCnt].y;
		double z_w = objectPoints[ptCnt].z;

		//刚体旋转平移后
		double x_rt = R11 * x_w + R12 * y_w + R13 * z_w + t1;
		double y_rt = R21 * x_w + R22 * y_w + R23 * z_w + t2;
		double z_rt = R31 * x_w + R32 * y_w + R33 * z_w + t3;

		//投影，归一化
		double x_p = x_rt / z_rt;
		double y_p = y_rt / z_rt;
		double z_p = z_rt / z_rt;

		//镜头畸变
		double r2 = x_p * x_p + y_p * y_p;
		double Kn = 1 + k1 * r2 + k2 * pow(r2, 2) + k3 * pow(r2, 3);
#if DISTORTION_ACCURACY == 1
		double Kd = 1 + k4 * r2 + k5 * pow(r2, 2) + k6 * pow(r2, 3);
#else
		double Kd = 1;
#endif
		double K = Kn / Kd;
		double x_d = x_p * K + 2 * p1 * x_p * y_p + p2 * (r2 + 2 * pow(x_p, 2));
		double y_d = y_p * K + p1 * (r2 + 2 * pow(y_p, 2)) + 2 * p2 * x_p * y_p;

		//坐标尺度变换及平移
		double x0 = fx * x_d + u0;
		double y0 = fy * y_d + v0;

		cv::Point2f tempPoint;
		tempPoint.x = x0;
		tempPoint.y = y0;
		projectionPoints.push_back(tempPoint);

		//按求导链式法则
		if (dpdf) {
			dpdf->at<double>(2 * ptCnt, 0) = x_d;
			dpdf->at<double>(2 * ptCnt, 1) = 0;
			dpdf->at<double>(2 * ptCnt + 1, 0) = 0;
			dpdf->at<double>(2 * ptCnt + 1, 1) = y_d;
		}
		if (dpdc) {
			dpdc->at<double>(2 * ptCnt, 0) = 1;
			dpdc->at<double>(2 * ptCnt, 1) = 0;
			dpdc->at<double>(2 * ptCnt + 1, 0) = 0;
			dpdc->at<double>(2 * ptCnt + 1, 1) = 1;
		}
		if (dpdk) {
			double dx0dk1 = fx * x_p * r2 / Kd;
			double dx0dk2 = fx * x_p * pow(r2, 2) / Kd;
			double dx0dk3 = fx * x_p * pow(r2, 3) / Kd;
			double dy0dk1 = fy * y_p * r2 / Kd;
			double dy0dk2 = fy * y_p * pow(r2, 2) / Kd;
			double dy0dk3 = fy * y_p * pow(r2, 3) / Kd;
#if DISTORTION_ACCURACY == 1
			double dx0dk4 = fx * x_p * r2 * (-1) * Kn / (Kd * Kd);
			double dx0dk5 = fx * x_p * pow(r2, 2) * (-1) * Kn / (Kd * Kd);
			double dx0dk6 = fx * x_p * pow(r2, 3) * (-1) * Kn / (Kd * Kd);
			double dy0dk4 = fy * y_p * r2 * (-1) * Kn / (Kd * Kd);
			double dy0dk5 = fy * y_p * pow(r2, 2) * (-1) * Kn / (Kd * Kd);
			double dy0dk6 = fy * y_p * pow(r2, 3) * (-1) * Kn / (Kd * Kd);
#endif

			dpdk->at<double>(2 * ptCnt, 0) = dx0dk1;
			dpdk->at<double>(2 * ptCnt, 1) = dx0dk2;
			dpdk->at<double>(2 * ptCnt, 2) = dx0dk3;
			dpdk->at<double>(2 * ptCnt + 1, 0) = dy0dk1;
			dpdk->at<double>(2 * ptCnt + 1, 1) = dy0dk2;
			dpdk->at<double>(2 * ptCnt + 1, 2) = dy0dk3;
#if DISTORTION_ACCURACY == 1
			dpdk->at<double>(2 * ptCnt, 3) = dx0dk4;
			dpdk->at<double>(2 * ptCnt, 4) = dx0dk5;
			dpdk->at<double>(2 * ptCnt, 5) = dx0dk6;
			dpdk->at<double>(2 * ptCnt + 1, 3) = dy0dk4;
			dpdk->at<double>(2 * ptCnt + 1, 4) = dy0dk5;
			dpdk->at<double>(2 * ptCnt + 1, 5) = dy0dk6;
#endif
		}
		if (dpdp) {
			double dx0dp1 = fx * 2 * x_p * y_p;
			double dx0dp2 = fx * (r2 + 2 * pow(x_p, 2));
			double dy0dp1 = fy * (r2 + 2 * pow(y_p, 2));
			double dy0dp2 = fy * 2 * x_p * y_p;

			dpdp->at<double>(2 * ptCnt, 0) = dx0dp1;
			dpdp->at<double>(2 * ptCnt, 1) = dx0dp2;
			dpdp->at<double>(2 * ptCnt + 1, 0) = dy0dp1;
			dpdp->at<double>(2 * ptCnt + 1, 1) = dy0dp2;
		}
		if (dpdr) {
			for (int i = 0; i < 3; i++) {
				double dR11dr = jacobian_R.at<double>(i, 0);
				double dR12dr = jacobian_R.at<double>(i, 1);
				double dR13dr = jacobian_R.at<double>(i, 2);
				double dR21dr = jacobian_R.at<double>(i, 3);
				double dR22dr = jacobian_R.at<double>(i, 4);
				double dR23dr = jacobian_R.at<double>(i, 5);
				double dR31dr = jacobian_R.at<double>(i, 6);
				double dR32dr = jacobian_R.at<double>(i, 7);
				double dR33dr = jacobian_R.at<double>(i, 8);

				double dxrt_dr = x_w * dR11dr + y_w * dR12dr + z_w * dR13dr;
				double dyrt_dr = x_w * dR21dr + y_w * dR22dr + z_w * dR23dr;
				double dzrt_dr = x_w * dR31dr + y_w * dR32dr + z_w * dR33dr;

				double dxp_dr = dxrt_dr / z_rt - dzrt_dr * x_rt / pow(z_rt, 2);
				double dyp_dr = dyrt_dr / z_rt - dzrt_dr * y_rt / pow(z_rt, 2);

				double dr2dr = 2 * x_p * dxp_dr + 2 * y_p * dyp_dr;

				double dKn_dr2 = k1 + 2 * k2 * r2 + 3 * k3 * pow(r2, 2);
				double dK_dKn = 1 / Kd;
#if DISTORTION_ACCURACY == 1
				double dKd_dr2 = k4 + 2 * k5 * r2 + 3 * k6 * pow(r2, 2);
				double dK_dKd = -1 * Kn / (Kd * Kd);
				double dKdr2 = dK_dKn * dKn_dr2 + dK_dKd * dKd_dr2;
#else
				double dKdr2 = dK_dKn * dKn_dr2;
#endif
				double dKdr = dKdr2 * dr2dr;

				double dxd_dr = K * dxp_dr + x_p * dKdr + 2 * p1 * (y_p * dxp_dr + x_p * dyp_dr) + p2 * (dr2dr + 4 * x_p * dxp_dr);
				double dyd_dr = K * dyp_dr + y_p * dKdr + p1 * (dr2dr + 4 * y_p * dyp_dr) + 2 * p2 * (y_p * dxp_dr + x_p * dyp_dr);

				double dx0dr = fx * dxd_dr;
				double dy0dr = fy * dyd_dr;

				dpdr->at<double>(2 * ptCnt, i) = dx0dr;
				dpdr->at<double>(2 * ptCnt + 1, i) = dy0dr;
			}
		}
		if (dpdt) {
			for (int i = 0; i < 3; i++) {
				double dxrt_dt, dyrt_dt, dzrt_dt;

				switch (i) {
				case 0:
					dxrt_dt = 1;
					dyrt_dt = 0;
					dzrt_dt = 0;
					break;
				case 1:
					dxrt_dt = 0;
					dyrt_dt = 1;
					dzrt_dt = 0;
					break;
				case 2:
					dxrt_dt = 0;
					dyrt_dt = 0;
					dzrt_dt = 1;
					break;
				}

				double dxp_dt = dxrt_dt / z_rt - dzrt_dt * x_rt / pow(z_rt, 2);
				double dyp_dt = dyrt_dt / z_rt - dzrt_dt * y_rt / pow(z_rt, 2);

				double dr2dt = 2 * x_p * dxp_dt + 2 * y_p * dyp_dt;

				double dKn_dr2 = k1 + 2 * k2 * r2 + 3 * k3 * pow(r2, 2);
				double dK_dKn = 1 / Kd;
#if DISTORTION_ACCURACY == 1
				double dKd_dr2 = k4 + 2 * k5 * r2 + 3 * k6 * pow(r2, 2);
				double dK_dKd = -1 * Kn / (Kd * Kd);
				double dKdr2 = dK_dKn * dKn_dr2 + dK_dKd * dKd_dr2;
#else
				double dKdr2 = dK_dKn * dKn_dr2;
#endif
				double dKdt = dKdr2 * dr2dt;

				double dxd_dt = K * dxp_dt + x_p * dKdt + 2 * p1 * (y_p * dxp_dt + x_p * dyp_dt) + p2 * (dr2dt + 4 * x_p * dxp_dt);
				double dyd_dt = K * dyp_dt + y_p * dKdt + p1 * (dr2dt + 4 * y_p * dyp_dt) + 2 * p2 * (y_p * dxp_dt + x_p * dyp_dt);

				double dx0dt = fx * dxd_dt;
				double dy0dt = fy * dyd_dt;

				dpdt->at<double>(2 * ptCnt, i) = dx0dt;
				dpdt->at<double>(2 * ptCnt + 1, i) = dy0dt;
			}
		}
	}

	return 0;
}