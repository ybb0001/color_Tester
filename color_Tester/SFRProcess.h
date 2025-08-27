/*******************************************/
/*                                         */
/* - title : SFRProcess.h                  */
/* - description : SFR Calculation Header  */
/* - author : Jae-Jin Kim                  */
/*            (R&D Center, Samsung Electro-*/
/*			   Mechanics)                  */
/* - date : 2006-01-17                     */
/*                                         */
/*******************************************/
#pragma once

typedef signed long  int32;
typedef signed short int16;
typedef signed char  int8;

typedef unsigned long	uint32;
typedef unsigned short	uint16;
typedef unsigned char	uint8;

typedef unsigned long  unsigned32;
typedef unsigned short unsigned16;
typedef unsigned char  unsigned8;

typedef long Fixed;
typedef long Fract;
typedef long (*ProcPtr)();

// system header files
#include <stdlib.h>
#include <String>
#include <stdio.h>

#define RES_MAX_ROI	200

typedef struct tagImageInfo
{
	int process_step;
	double **ImageData; 
	int32 rectHeight;
	int32 rectWidth;
	uint16 oecfdata[256];
	int depth;
	double del;
	double *SFRGraph[RES_MAX_ROI];
	int SFRLength;
	double MeasureFreq;
	double MeasureFreq2;
	double **DerivImage;
} sImageInfo;

class CSfr {
public:
	sImageInfo m_sfrData;

	CSfr();
	virtual ~CSfr();
	void Init(int size);
	void Dealloc();
	sImageInfo* GetSFRDataHandle();
	double sfr(int position,double lpfreq,int minlevel,int maxlevel,bool norm);

private:
	int clipping(int low,int high,double thresh1, uint16 rectHeight,uint16 rectWidth);
	void getoecf();
	int rotatev();
	void ahamming(double* win,double n,double mid);
	void deriv1(double *fil,int fillength);
	double centroid_image(double *win1,int ypos);
	double centroid(double *x,int length);
	double* findedge(double *loc,double *fitme);
	double* project(double *point,double *loc, double slope, double nbin);
	double* cent(double *a, int center, int length);
	int DFT(int dir,int m,double *x1,double *y1);
	void four1(double* data,unsigned long nn,int isign);
	void realft(double *data, unsigned long n, int isign);
};