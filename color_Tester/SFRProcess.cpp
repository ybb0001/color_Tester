/*******************************************/
/*                                         */
/* - title : SFRProcess.c                  */
/* - description : SFR Calculation         */
/* - author : Jae-Jin Kim                  */
/*            (R&D Center, Samsung Electro-*/
/*			   Mechanics)                  */
/* - date : 2006-01-17                     */
/*                                         */
/*******************************************/

#include "SFRProcess.h"
#include <math.h>
#include <float.h>
#include <afx.h>
#define SWAP(a,b)	tempr=(a);(a)=(b);b=tempr
#define M_PI		3.14159265358979323846
#define	J_SFR_ZERO	0.000010

CSfr::CSfr()
{
	Init(192);
}

CSfr::~CSfr()
{
	Dealloc();
}

sImageInfo* CSfr::GetSFRDataHandle()
{
	return &m_sfrData;
}

void CSfr::Init(int size)
{
	if(m_sfrData.rectHeight != size || m_sfrData.rectWidth != size) {
		m_sfrData.rectHeight = size;
		m_sfrData.rectWidth	= size;
		m_sfrData.SFRLength	= m_sfrData.rectWidth*2;
	}

	for(int i = 0; i < RES_MAX_ROI; i++){
		m_sfrData.SFRGraph[i]	= (double *)calloc(m_sfrData.SFRLength , sizeof(double));
	}
	m_sfrData.MeasureFreq	= m_sfrData.MeasureFreq2 = 0;

	m_sfrData.ImageData		= (double**)calloc(m_sfrData.rectHeight, sizeof(double));
	for(int i = 0; i < m_sfrData.rectHeight; i++){
		m_sfrData.ImageData[i]	= (double*)calloc(m_sfrData.rectWidth, sizeof(double));
	}

	m_sfrData.DerivImage	= (double**)calloc(m_sfrData.rectHeight, sizeof(double));
	for(int i = 0; i < m_sfrData.rectHeight; i++){
		m_sfrData.DerivImage[i]	= (double*)calloc(m_sfrData.rectWidth, sizeof(double));
	}

	for(int i = 0; i < 256; i++){
		m_sfrData.oecfdata[i]	= i;
	}

	m_sfrData.depth = 0;
	m_sfrData.del = 0;
}

void CSfr::Dealloc()
{
	// Delete SFR
	for(int i = 0; i < RES_MAX_ROI; i++){
		free(m_sfrData.SFRGraph[i]);
	}
	for(int i = 0; i < m_sfrData.rectHeight; i++){
		free(m_sfrData.ImageData[i]);
	}
	free(m_sfrData.ImageData);
	for(int i = 0; i < m_sfrData.rectHeight; i++){
		free(m_sfrData.DerivImage[i]);
	}
	free(m_sfrData.DerivImage);
}

double CSfr::sfr(int position,double lpfreq,int minlevel,int maxlevel,bool norm)
{
	int cstatus=0;
	int smax = 255;
	int ncol = 1;
	double threshold = 0.005;
	double *win1;
	double *loc;
	double *fitme;
	double *fitme2;
	double *midloc,*slout;
	double *mtf;
	double *freq;
	double *win;
	double *esf;
	double *place;
	double *win2;
	double *c;
	double *temppoint;
	double *c_temp;
	double *cr,*ci;
	double *point;
	double result=0;
	int i;
	int templevel;
	double slope;
	double nbin;
	double dcvalue=0;
	double fillength;
	double nfreq;
	double nn2out;
	double freqlim;
	int color;
	double nn2;
	double nn;
	int ncolout;
	double temp;
	double test;
	double mid;
	//char stemp[100];

	//Normalization Option
	if(norm == true){
		slope = 255.0/(double)(maxlevel-minlevel);
		if(minlevel > maxlevel){
			templevel	 = minlevel;
			minlevel	 = maxlevel;
			maxlevel	 = templevel;
		}
		for(i = 0; i < minlevel; i++){
			m_sfrData.oecfdata[i]	 = 0;
		}
		for(i = minlevel; i < maxlevel; i++){
			m_sfrData.oecfdata[i]	 = (int)((double)(i-minlevel)*slope);
		}
		for(i = maxlevel;i < 256; i++){
			m_sfrData.oecfdata[i] = 255;
		}
		getoecf();
	}
	
	/* SFR 2005.11.03 */
	/*
	if (cstatus != 1) { 	
		//::MessageBox(NULL, "** WARNING: Clipping Occured!!", "SFR Warning", MB_OK);
		//return 0;
	}
	*/
	
	int rflag = 0;
	// rotate horizontal edge so it is vertical
	rflag	 = rotatev();

	double fil1[2]	 = {0.5, -0.5};
	double fil2[3]	 = {0.5, 0, -0.5};

	//Need 'positive' edge for good centroid calculation
	double tleft	 = 0;
	double tright	 = 0;
	
	for(i = 0; i < m_sfrData.rectHeight; i++){
		for(int j = 0; j < 5; j++){
			tleft	+= m_sfrData.ImageData[i][j];
			tright	+= m_sfrData.ImageData[i][m_sfrData.rectWidth-1-j];
		}
	}
	if(tleft > tright){
		//AfxMessageBox("tleft > tright");
		fil1[0]	 = -0.5;
		fil1[1]	 =  0.5;
		fil2[0]	 = -0.5;
		fil2[1]	 =  0;
		fil2[2]	 =  0.5;
	}
	else{
		//AfxMessageBox("tleft < tright");
	}
	
	test	 = 0;
	// Test for low contrast edge;
	test	 = fabs((tleft-tright)/(tleft+tright));

	if(test < 0.3){
		/* SFR 2005.11.03 */
		//AfxMessageBox("** WARNIG: Edge contrast is less that 20%, this can lead to high error in the SFR measurement.");
		return -1;//J_SFR_ZERO;
	}

	// smoothing window for first part of edge location estimation - 
	// to used on each line of ROI

	mid	 = ((double)m_sfrData.rectWidth+1)/2;

	win1 = (double*)calloc(m_sfrData.rectWidth, sizeof(double));
	ahamming(win1,m_sfrData.rectWidth,mid);

	deriv1(fil1,2);

	//loc = new double[m_sfrData.rectHeight];
	loc	= (double*)calloc(m_sfrData.rectHeight, sizeof(double));

	// compute centroid for derivative array for each line in ROI. NOTE WINDOW array 'win'
	for(i = 0; i < m_sfrData.rectHeight; i++){
		loc[i] = centroid_image(win1,i) - 0.5;   //% -0.5 shift for FIR phase
	}

	//fitme = new double[2];
	fitme = (double*)calloc(2, sizeof(double));
	fitme = findedge(loc,fitme); 
	if(fabs(fitme[1]) < 0.008 || fabs(fitme[1]) > 0.20 || _isnan(fitme[1])){
		free(win1);		//delete [] win1;
		free(loc);		//delete [] loc;
		free(fitme);	//delete [] fitme;
		return -2;//J_SFR_ZERO;
	}
	
	//place = new double[m_sfrData.rectHeight];
	place	= (double*)calloc(m_sfrData.rectHeight, sizeof(double));

	//win2 = new double[(int)m_sfrData.rectWidth];
	win2	= (double*)calloc(m_sfrData.rectWidth, sizeof(double));
	for(i = 0; i < m_sfrData.rectHeight; i++){
		place[i] = fitme[0] + fitme[1]*i;
		ahamming(win2,m_sfrData.rectWidth,place[i]);
		loc[i]	 = centroid_image(win2,i) - 0.5;
	}

	//fitme2 = new double[2];
	fitme2 = (double*)calloc(2, sizeof(double));
	fitme2 = findedge(loc,fitme2);

	if(fabs(fitme2[1]) < 0.008 || fabs(fitme2[1]) > 0.20 || _isnan(fitme2[1])){
		free(win1);
		free(loc);
		free(fitme);
		free(place);
		free(win2);
		free(fitme2);
		return -3;//J_SFR_ZERO;
	}

	// output edge location listing
	ncolout = ncol;
	if(ncol == 4){
		ncolout = ncol - 1;
	}	

	midloc	= (double*)calloc(ncolout, sizeof(double));
	slout	= (double*)calloc(ncolout, sizeof(double));
	for(i=0;i<ncolout;i++){
		slout[i] = -1/fitme2[1];	// slope is as normally defined in image coods.
		if(rflag ==1){				// positive flag it ROI was rotated
			slout[i] = -fitme2[1];
		}
		//evaluate equation(s) at the middle line as edge location
		midloc[i] = fitme2[0] + fitme2[1]*((m_sfrData.rectHeight-1)/2);
	}

	nbin	= 4;
	nn		= floor((double)m_sfrData.rectWidth * nbin);

	//mtf = new double[(int)nn];
	mtf		= (double*)calloc((int)nn, sizeof(double));

	nn2		=  nn/2 + 1;

	//freq = new double[(int)nn];
	freq	 = (double*)calloc((int)nn, sizeof(double));
	for(i = 0; i < nn; i++){
		freq[i]	 = nbin*((double)i-1)/(m_sfrData.del*nn);
	}

	freqlim	 = 1;
	// % limits plotted sfr to 0- 1 cy/pxel freqlim = 2 for all data

	nn2out	 = (double)((int)(((nn2*freqlim/2)*2+1)/2));
	
	//???
	nfreq	 = nn/(2*m_sfrData.del*nn);    //% half-sampling frequency
		
	//win = new double[(int)nbin*m_sfrData.rectWidth];
	win		= (double*)calloc((int)nbin*m_sfrData.rectWidth, sizeof(double));
	ahamming(win, nbin*m_sfrData.rectWidth, (nbin*m_sfrData.rectWidth+1)/2);      //% centered Hamming window

	//% ************** Large SFR loop for each color record
	esf		= (double*)calloc((int)nn, sizeof(double));
	point	= (double*)calloc(m_sfrData.rectWidth*(int)nbin, sizeof(double));

	for(color = 0; color < ncol; color++){
		//% project and bin data in 4x sampled array
		point = project(point, loc, fitme2[1], nbin);
	}

	for(i = 0; i < nn; i++){
		esf[i]	 = point[i];
	}
	// compute first derivative via FIR (1x3) filter fil
	//	c = deriv1(point, 1, nn, fil2);

	c		 = (double*)calloc((int)nn , sizeof(double));
	temppoint= (double*)calloc((int)nn+3-1 , sizeof(double));
		
	for(i = 0; i < nn; i++){
		temppoint[i]	 = point[i];
	}

	fillength	 = 3;
	for(i = 2; i < nn+fillength-1; i++){
		temp	 = fil2[0]*temppoint[i] + fil2[1]*temppoint[i-1] + fil2[2]*temppoint[i-2];
		if(i < nn){
			c[i]	 = temp;
		}
	}

	for(i = 0;i < nn; i++){
		if(i < (int)fillength-1){
			c[i]	 = c[(int)fillength-1];
		} 
	}
	mid		 = centroid(c,(int)nn);
	c_temp	 = cent(c, (int)((mid*2+1)/2),(int)nn);              // shift array so it is centered
	for(i = 0; i < nn; i++){
		c[i]	 = c_temp[i];
	}
	// apply window (symmetric Hamming)
	for(i = 0; i < nn; i++){
		c[i]	 = win[i]*c[i];
	}

	int status;
	//2의 승수로 표시되는지 확인하여 FFT와 DFR 중 택일 
	if((int)(log10(nn)/log10(2.0)) == (log10(nn)/log10(2.0))){
		cr	 = (double *)calloc((int)nn*2+1 , sizeof(double));
		ci	 = (double *)calloc((int)nn*2+1 , sizeof(double));

		cr[0]	 = 0;
		for(i = 0; i < nn; i++){
			cr[i+1]	 = c[i];
		}

		realft(cr,(int)nn,1);
		dcvalue	 = 0;
		dcvalue	 = sqrt(cr[1]*cr[1]+cr[2]*cr[2]);
		for(i = 0; i < nn; i++){
			c[i]	 = sqrt(cr[i*2+1]*cr[i*2+1]+cr[i*2+2]*cr[i*2+2])/dcvalue;
		}
	}
	else{
		cr	 = (double *)calloc((int)nn , sizeof(double));
		ci	 = (double *)calloc((int)nn , sizeof(double));

		for(i = 0; i < nn; i++){
			cr[i]	 = c[i];
		}

		status	 = DFT(1,(int)nn,cr,ci);
		dcvalue	 = 0;
		dcvalue	 = sqrt(cr[0]*cr[0] + ci[0]*ci[0]);
		for(i = 0; i < nn; i++){
			c[i] = sqrt(cr[i]*cr[i] + ci[i]*ci[i])/dcvalue;
		}
	}

	m_sfrData.SFRLength	 = (int32)nn/2;

	//측정 위치별 결과 저장
	if(!(position > RES_MAX_ROI)){
		for (i=0;i<m_sfrData.SFRLength;i++) {
			m_sfrData.SFRGraph[position][i] = c[i];
		}
		result = m_sfrData.SFRGraph[position][(int)lpfreq]-(m_sfrData.SFRGraph[position][(int)lpfreq]-m_sfrData.SFRGraph[position][(int)lpfreq+1])*(lpfreq-(int)lpfreq);
	}
	//free memory allocations
	free(win1);
	free(loc);
	free(fitme);
	free(place);
	free(win2);
	free(fitme2);
	free(c_temp);
	free(midloc);
	free(slout);
	free(mtf);
	free(freq);
	free(win);
	free(esf);
	free(point);
	free(c);
	free(temppoint);
	free(cr);
	free(ci);
	
	//측정값이 2가 넘거나 음수일 경우 0을 리턴
	if(result < 2 && result > 0 ){
		return result;
	}
	else{
		return -4;//J_SFR_ZERO;
	}

}

/************************************************************************/
/* [n, status] = clipping(a, low, high, thresh1) Checks for data clipping*/
/* Function checks for clipping of data array							*/
/*  a       = array														*/
/*  low     = low clip value											*/
/*  high    = high clip value											*/
/*  thresh1 = threshhold fraction [0-1] used for warning,				*/
/*            if thresh1 = 0, all clipping is reported					*/		
/************************************************************************/
int CSfr::clipping(int low,int high,double thresh1, uint16 nlin,uint16 npix)
{
	int status = 1;
	int n = nlin*npix;
	int ncol = 1; //나중에 RGB 칼라 값을 받을때는 3으로 변경

	int nhigh = 0;
	int nlow =  0;
	int i,j,k;
		

	for (k = 1; k <= ncol;k++)
	{
	   for (j = 0;j< npix;j++)
	   {
		  for (i = 0; i < nlin;i++)
		  {
			if (m_sfrData.ImageData[i][j] <= low)  {
				nlow = nlow + 1;
			}
			if (m_sfrData.ImageData[i][j] >= high) {
				nhigh = nhigh + 1;
			}
		  }
	   }
	}

	nhigh = (int)((double)nhigh/(double)n);
	nlow = (int)((double)nlow/(double)n);

	for (k = 1; k <= ncol;k++) {
		if (nlow > thresh1) {
			//disp([' *** Warning: low clipping in record ', num2str(k)]);
			status = 0;
		}				
		if (nhigh > thresh1) {
			//disp([' *** Warning: high clipping in record ', num2str(k)]);
			status = 0;
		}
	}	

	if (status != 1) {
	 //warndlg('Data clipping errors detected','ClipCheck');
	}
	return status;

}




void CSfr::getoecf()
{
	int status = 0;
	
	/* 그레이 스케일일 경우 ncol = 1
	if size(stuff)==[1 2]
	   ncol = 1;
	else
	   ncol = stuff(3);
	*/

	int ncol = 1;

	int lutxpos=0;

	if (ncol==1) {
		for (int i=0;i<m_sfrData.rectHeight;i++) {
		  for (int j=0;j<m_sfrData.rectWidth;j++) {
			  if (m_sfrData.depth == 16) {
				//AfxMessageBox("oecf!!");
				lutxpos = (uint16)(m_sfrData.ImageData[i][j] / 256);
				m_sfrData.ImageData[i][j] = (uint16)m_sfrData.oecfdata[lutxpos] * 256;
			  } else {
				//AfxMessageBox("oecf2!!");
				lutxpos = (uint16)m_sfrData.ImageData[i][j];
				m_sfrData.ImageData[i][j] = m_sfrData.oecfdata[lutxpos];
			  }			
		  }
		}
	} else {
		//RGB 이미지일 경우 차후에 코딩 예정 
		/*
		   for i=1: nlin;
			  for j = 1: npix;
				 for k=1:ncol;
					array(i,j,k) = oedat( array(i,j,k)+1, k);
				 end;
			  end;
		   end;
		*/
	}



}

/************************************************************************/
/* Rotate array															*/	
/*																		*/
/* Rotate array so that long dimensions is vertical (line) drection		*/
/* flag = 0 no roation, = 1 rotation was performed						*/
/*																		*/
/************************************************************************/
int CSfr::rotatev() {

	int xpoint[10],ypoint[10];
	int xperiod= 0, yperiod = 0;
	int i =0;
	double xsum = 0, ysum = 0;
	double **TempImage;
	xperiod = m_sfrData.rectWidth/10;

	for (i=0;i<9;i++) {
		xpoint[i] = i*xperiod;
	}
	xpoint[9] = m_sfrData.rectWidth-1;
	
	yperiod = m_sfrData.rectHeight/10;

	for (i=0;i<9;i++) {
		ypoint[i] = i*yperiod;
	}
	ypoint[9] = m_sfrData.rectHeight-1;
	
	if (m_sfrData.rectHeight > 5) {
		for (i=0;i<5;i++) {		
			xsum += abs((uint16)m_sfrData.ImageData[5][xpoint[i]]-(uint16)m_sfrData.ImageData[5][xpoint[9-i]]);
		}
	} else {
		//unable to measure vertical/horizontal direction - SEMCO
		return 0;
	}

	if (m_sfrData.rectWidth > 5) {
		for (i=0;i<5;i++) {		
			ysum += abs((uint16)m_sfrData.ImageData[ypoint[i]][5]-(uint16)m_sfrData.ImageData[ypoint[9-i]][5]);
		}
	} else {
		//unable to measure vertical/horizontal direction - SEMCO
		return 0;
	}

	//Directin Check
	if (xsum < ysum) {
		TempImage = new double*[m_sfrData.rectWidth]; //SEMCO
		for(i = 0; i < m_sfrData.rectWidth; i++)	//SEMCO
			TempImage[i] = new double[m_sfrData.rectHeight]; //SEMCO
		
		for (i=0;i<m_sfrData.rectHeight;i++) {
		  for (int j=0;j<m_sfrData.rectWidth;j++) {
			  TempImage[j][m_sfrData.rectHeight-1-i] = m_sfrData.ImageData[i][j];
		  }
		}
		for (i=0;i<m_sfrData.rectHeight;i++) {
		  for (int j=0;j<m_sfrData.rectWidth;j++) {
			  m_sfrData.ImageData[i][j] = TempImage[i][j];
		  }
		}
		int32 TempHeight = m_sfrData.rectWidth;
		int32 TempWidth = m_sfrData.rectHeight;
		m_sfrData.rectHeight = TempHeight;
		m_sfrData.rectWidth = TempWidth;

		for (int i=0;i<m_sfrData.rectWidth;i++) {
			//free(TempImage[i]);
			delete [] TempImage[i];
		}
		delete [] TempImage;
		//free(TempImage);
		return 1;

	} else {
		return 0;
	}
}

/************************************************************************/
/*   Generates asymmetrical Hamming window								*/
/*   array. If mid = (n+1)/2 then the usual symmetrical Hamming array	*/
/*   is returned														*/
/*   n = length of array												*/
/*   mid = midpoint (maximum) of window function						*/
/*   win = window array (nx1)											*/
/************************************************************************/
void CSfr::ahamming(double* win,double n,double mid)
{
	
	double wid1 = mid-1;
	double wid2 = (double)n-mid;


	double wid=0;

	if (wid1>wid2) {
		wid = wid1;
	} else {
		wid = wid2;
	}

	
	double arg = 0;
	double pi = 3.1416;
	for (int i = 0;i<n;i++) {
		arg = (double)i-mid;
		win[i] = 0.54 + 0.46*cos( pi*(arg+1)/wid );
	}

}


/************************************************************************/
/*  First derivative of array											*/
/*  Computes first derivative via FIR (1xn) filter						*/
/*  Edge effects are suppressed and vector size is preserved			*/
/*  Filter is applied in the npix direction only						*/
/*   m_sfrData.ImageData   = (nlin, npix) data array					*/
/*   fil = array of filter coefficients, eg [[-0.5 0.5]					*/
/*   DerivImage   = output (nlin, npix) data array						*/
/************************************************************************/
void CSfr::deriv1(double *fil,int fillength) {
	
	int i,j;
	double **TempImage;

	TempImage = new double*[m_sfrData.rectHeight]; //SEMCO
	for(i = 0; i < m_sfrData.rectHeight; i++)	//SEMCO
		TempImage[i] = new double[m_sfrData.rectWidth+fillength]; //SEMCO

	
	for (i=0;i < m_sfrData.rectHeight;i++) {
		for (j=0;j<m_sfrData.rectWidth;j++) {
			TempImage[i][j] = m_sfrData.ImageData[i][j];
		}
	}
	
	double temp;
	for (i=0;i < m_sfrData.rectHeight;i++) {
		for (j=fillength-1;j<m_sfrData.rectWidth+fillength;j++) {
			temp = 0;
			temp = fil[0]*TempImage[i][j] + fil[1]*TempImage[i][j-1];
			if (j<m_sfrData.rectWidth) {
				m_sfrData.DerivImage[i][j] = temp;
			}
		}
	}

	for (i=0;i < m_sfrData.rectHeight;i++) {
		for (j=0;j<m_sfrData.rectWidth;j++) {
			if (j<fillength-1) {
				m_sfrData.DerivImage[i][j] = m_sfrData.DerivImage[i][fillength];
			} 
			else {
			}
		}
	}
	for(i=0;i<m_sfrData.rectHeight;i++){
		//free(TempImage[i]);
		delete [] TempImage[i];
	}
	//free(TempImage);
	delete [] TempImage;
}

/************************************************************************/
/*  Finds centroid of image												*/
/*  Returns centroid location of a vector								*/
/*   loc = centroid in units of array index								*/
/************************************************************************/
double CSfr::centroid_image(double *win,int ypos) {

	double loc = 0;
	double sum = 0;
	double *TempImage;

	TempImage = new double[m_sfrData.rectWidth];

	for (int j=1;j<m_sfrData.rectWidth+1;j++) {
		TempImage[j-1] = m_sfrData.DerivImage[ypos][j-1]*win[j-1];
		sum += TempImage[j-1];
		loc = loc+ (double)(j)*TempImage[j-1];
	}
	loc = loc/sum;

	//free(TempImage);
	delete [] TempImage;
	return loc;
}

/************************************************************************/
/*  Finds centroid of vector											*/
/*  Returns centroid location of a vector								*/
/*   x   = vector														*/
/*   loc = centroid in units of array index								*/
/************************************************************************/
double CSfr::centroid(double* x,int length) {

	double loc = 0;
	double sum = 0;

	for (int j=1;j<length+1;j++) {		
		sum = sum + x[j-1];
		loc = loc+ (double)(j)*x[j-1];
	}
	loc = loc/sum;


	return loc-1;
}

/************************************************************************/
/* Fits linear equation to data											*/
/* Fit linear equation to data, written to process edge location array	*/
/*   cent = array of (centroid) values									*/
/*   nlin = length of cent												*/
/*   slope and int are from the least-square fit						*/
/*    x = bestint + bestslope*cent(x)									*/
/*  Note that this is the inverse of the usual cent(x) = int + slope*x	*/
/*  form																*/
/************************************************************************/
double* CSfr::findedge(double *loc,double *fitme) {

	int midpoint,lowpoint,highpoint;
	double slope1, slope2;
	double defaultslope,defaultint;
	double bestslope,bestint;
	double tempslope,tempint;
	double rmsvalue;
	int bestposi	= 0;
	int bestposj	= 0;

	midpoint = (int)((double)m_sfrData.rectHeight/2);
	lowpoint = (int)((double)m_sfrData.rectHeight/10);
	highpoint = (int)((double)m_sfrData.rectHeight/10*9);
	slope1= (loc[midpoint]-loc[lowpoint])/(double)(midpoint-lowpoint);	
	slope2= (loc[highpoint]-loc[midpoint])/(double)(highpoint-midpoint);	

	defaultslope = (loc[highpoint]-loc[lowpoint])/(double)(highpoint-lowpoint);

	defaultint = loc[lowpoint];

	bestslope = defaultslope;
	bestint = defaultint;


	double previousrmsvalue;
	int count = 0,tcount = 0;
	while(1) {
		for(int i = -3; i < 3; i++){
			tempslope = defaultslope+(i*0.001/(pow(2.0,count)));	//
			for (int j = -3; j < 3; j++){
				rmsvalue = 0;
				tempint = defaultint+(j*0.01/(pow(2.0,count)));		//
				for (int k=0;k<m_sfrData.rectHeight;k++) {
					rmsvalue += (loc[k]-(tempslope*(double)k+tempint))*(loc[k]-(tempslope*(double)k+tempint));
				}
				rmsvalue = sqrt(rmsvalue);
				if(i == -3 && j == -3){
					previousrmsvalue = rmsvalue;
				}
				if(rmsvalue < previousrmsvalue){
					previousrmsvalue = rmsvalue;
					bestslope = tempslope;
					bestint = tempint;
					bestposi = i;
					bestposj = j; 
				}
			}
		}
		if (bestposi == 0 && bestposj == 0) {
			count++;
		}
		if (count > 17) {
			break;
		}		
		if (tcount > 70) {
			break;
		}
		tcount++;

		defaultslope = bestslope;
		defaultint = bestint;
	}

	fitme[0] = bestint;
	fitme[1] = bestslope;

	return fitme;
}

/************************************************************************/
/* Projects data														*/	
/*																		*/
/* Projects the data in array bb along the direction defined by			*/
/*  npix = (1/slope)*nlin.  Used by sfrmat11 and sfrmat2 functions.		*/
/* Data is accumulated in 'bins' that have a width (1/fac) pixel.		*/
/* The smooth, supersampled one-dimensional vector is returned.			*/
/*  bb = input data array												*/
/*  slope and loc are from the least-square fit to edge					*/
/*    x = loc + slope*cent(x)											*/
/*  fac = oversampling (binning) factor, default = 4					*/
/*  Note that this is the inverse of									*/
/*  the usual cent(x) = int + slope*xstatus =1;							*/
/*  point = output vector												*/
/*  status = 1, OK														*/
/*  status = 0, zero counts encountered in binning operation, warning is*/
/*           printed, but execution continues							*/
/************************************************************************/
double* CSfr::project(double *point,double *loc, double slope, double fac)
{
	int i = 0;

	double status = 1;	

	double big = 0;
	double nn = m_sfrData.rectWidth *fac ;

	// smoothing window
	double *win;
	win = new double[(int)nn];
	ahamming(win,nn, fac*loc[0]);



	slope =  1/slope;
	double offset = (double)(int)(((fac*(0-(m_sfrData.rectHeight-1)/slope))*2+1)/2);

	double del = fabs(offset);
	if (offset>0) {
		offset=0;
	}

	double **barray;
	
	barray = new double*[2];
	barray[0] =  (double *)calloc((int)nn + (int)del+100 , sizeof(double));
	barray[1] = (double *)calloc((int)nn + (int)del+100 , sizeof(double));

	double **positionarray;
	positionarray = new double*[m_sfrData.rectHeight];
	for (i =0;i<m_sfrData.rectHeight;i++) {
		positionarray[i] = (double *)calloc(m_sfrData.rectWidth , sizeof(double));
	}
	
	double x= 0, y= 0;
	double ling = 0;

	//% Projection and binning
	for (int n = 0; n < m_sfrData.rectHeight; n++){
		for (int m = 0; m < m_sfrData.rectWidth; m++){
			x = n;
			y = m;
			ling =  ceil((x-y/slope)*fac) + 1 - offset;
			barray[0][(int)ling] = barray[0][(int)ling] + 1;
			barray[1][(int)ling] = barray[1][(int)ling] + m_sfrData.ImageData[m][n];
			positionarray[n][m]	 = ling;	
		}
	}

	 //point = zeros(nn,1);
	 double start = 1+(double)(int)((0.5*m_sfrData.del)*2+1)/2; 

	//% Check for zero counts
	 double nz =0;
	 for (i = (int)start;i<(int)start+(int)nn-1;i++) { 

		 if (barray[0][i] ==0) {
			   nz = nz +1;
			   status = 0;  
			   if (i==1) {
					barray[0][i] = barray[0][i+1];
			   } else {
					barray[0][i] = (barray[0][i-1] + barray[0][i+1])/2;
			   }
		 }
	 }

 
	 if (status ==0) {
		/*
		disp('                            WARNING');
		disp('      Zero count(s) found during projection binning. The edge ')
		disp('      angle may be large, or you may need more lines of data.');
		disp('      Execution will continue, but see Users Guide for info.'); 
		disp(nz);
		*/
	 }

	// Combine in single edge profile, point

	
	 for (i = 0;i<nn;i++) {
	  point[i] = barray[1][i+(int)start]/ barray[0][i+(int)start];
	 }

	delete [] barray;
	for (i =0;i<m_sfrData.rectHeight;i++) {
		free(positionarray[i]);
	}
	delete [] positionarray;
	delete [] win;
	//free(win);
	return point;
}

/************************************************************************/
/*  Array shift for centering data										*/
/*  Matlab function cent: shift of one-dimensional						*/
/*  array, so that a(center) is located at b(round((n+1)/2).			*/
/*  Written to shift a line-spread function array prior to				*/
/*  applying a smoothing window.										*/
/*   a      = input array												*/
/*   center = location of signal center to be shifted					*/
/*   b      = output shifted array										*/
/************************************************************************/
double* CSfr::cent(double *a, int center,int n) {
	int i;
	double* b;


	b = (double *)calloc((int)n , sizeof(double));

	double mid = (double)(int)(((((double)n+1)/2)*2+1)/2);

	int del = (int)(((double)center - mid)*2+1)/2;

	if (del > 0) {
		for (i = 0;i<n-del;i++) {
		   b[i] = a[i + del];
		}

	} else if (del < 1) {
		for (i = -del+1;i<n;i++) {
		   b[i] = a[i + del];
		}
	} else {
		for (i=0;i<n;i++) {
			b[i] = a[i];
		}
	}


	return b;
}

/************************************************************************/
/* Fast Fourier Transform 
/* Calculates the Fourier transform of a set of n real-valued data		*/
/* points.  Replaces this data (which is store in array data[1..n]) by	*/
/* the positive frequency half of its complex Fourier transform.  The	*/	
/* real-valued first and last components of the complex transform are	*/
/* returned as elements data[1] and data[2], respectively.  n must be a	*/
/* power of 2.  This routine also calculates the inverse transform of a	*/
/* complex data array if it is the transform of real data. (Result in	*/
/* this case must be multiplied by 2/n.)								*/	
/************************************************************************/
void CSfr::realft(double data[], unsigned long n, int isign)
{
   unsigned long i,i1,i2,i3,i4,np3;
   double c1=0.5,c2,h1r,h1i,h2r,h2i;
   double wr,wi,wpr,wpi,wtemp,theta;

   theta=M_PI/(double) (n>>1);
   if (isign == 1) {
      c2 = -0.5;
      four1(data,n>>1,1);
   } else {
      c2=0.5;
      theta = -theta;
   }
   wtemp=sin(0.5*theta);
   wpr = -2.0*wtemp*wtemp;
   wpi=sin(theta);
   wr=1.0+wpr;
   wi=wpi;
   np3=n+3;
   for (i=2;i<=(n>>2);i++) {
      i4=1+(i3=np3-(i2=1+(i1=i+i-1)));
      h1r=c1*(data[i1]+data[i3]);
      h1i=c1*(data[i2]-data[i4]);
      h2r = -c2*(data[i2]+data[i4]);
      h2i=c2*(data[i1]-data[i3]);
      data[i1] = h1r+wr*h2r-wi*h2i;
      data[i2] = h1i+wr*h2i+wi*h2r;
      data[i3] = h1r-wr*h2r+wi*h2i;
      data[i4] = -h1i+wr*h2i+wi*h2r;
      wr=(wtemp=wr)*wpr-wi*wpi+wr;
      wi=wi*wpr+wtemp*wpi+wi;
   }
   if (isign == 1) {
      data[1] = (h1r=data[1])+data[2];
      data[2] = h1r-data[2];
   } else {
      data[1]=c1*((h1r=data[1])+data[2]);
      data[2]=c1*(h1r-data[2]);
      four1(data,n>>1,-1);
   }
}

/************************************************************************/
/* Replaces data[1..2*nn] by its discrete Fourier transform, if isign is*/
/* input as 1; or replaces data[1..2*nn] by nn times its inverse		*/
/* discrete Fourier transform, if isign is input as -1.  data is a		*/
/* complex array of length nn or, equivalently, a real array of length	*/
/* 2*nn.  nn MUST be an integer power of 2  (this is not checked for!)	*/
/************************************************************************/
void CSfr::four1(double data[], unsigned long nn, int isign)
{
   unsigned long n,mmax,m,j,istep,i;
   double wtemp,wr,wpr,wpi,wi,theta;
   double tempr,tempi;

   n=nn << 1;
   j=1;
   for (i=1;i<n;i+=2) {
      if (j > i) {
	 SWAP(data[j],data[i]);
	 SWAP(data[j+1],data[i+1]);
      }
      m=n >> 1;
      while (m >= 2 && j > m) {
	 j -= m;
	 m >>= 1;
      }
      j += m;
   }
   /* Here begins the Danielson-Lanczos section of the routine */
   mmax=2;
   while (n > mmax) {
      istep=mmax << 1;
      theta=isign*(6.28318530717959/mmax);
      wtemp=sin(0.5*theta);
      wpr = -2.0*wtemp*wtemp;
      wpi=sin(theta);
      wr=1.0;
      wi=0.0;
      for (m=1;m<mmax;m+=2) {
	 for (i=m;i<=n;i+=istep) {
	    j=i+mmax;
	    tempr=wr*data[j]-wi*data[j+1];
	    tempi=wr*data[j+1]+wi*data[j];
	    data[j]=data[i]-tempr;
	    data[j+1]=data[i+1]-tempi;
	    data[i] += tempr;
	    data[i+1] += tempi;
	 }
	 wr=(wtemp=wr)*wpr-wi*wpi+wr;
	 wi=wi*wpr+wtemp*wpi+wi;
      }
      mmax=istep;
   }
}

/************************************************************************/
/*  Direct fourier transform											*/
/************************************************************************/
int CSfr::DFT(int dir,int m,double *x1,double *y1)
{
   long i,k;
   double arg;
   double cosarg,sinarg;
   double *x2=NULL,*y2=NULL;

   x2 = (double *)calloc(m , sizeof(double));
   y2 = (double *)calloc(m , sizeof(double));
   if (x2 == NULL || y2 == NULL)
      return(FALSE);

   for (i=0;i<m;i++) {
      x2[i] = 0;
      y2[i] = 0;
      arg = - dir * 2.0 * 3.141592654 * (double)i / (double)m;
      for (k=0;k<m;k++) {
         cosarg = cos(k * arg);
         sinarg = sin(k * arg);
         x2[i] += (x1[k] * cosarg - y1[k] * sinarg);
         y2[i] += (x1[k] * sinarg + y1[k] * cosarg);
      }
   }

   /* Copy the data back */
   if (dir == 1) {
      for (i=0;i<m;i++) {
         x1[i] = x2[i] / (double)m;
         y1[i] = y2[i] / (double)m;
      }
   } else {
      for (i=0;i<m;i++) {
         x1[i] = x2[i];
         y1[i] = y2[i];
      }
   }

   free(x2);
   free(y2);
   return(TRUE);
}
