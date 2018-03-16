// Version 0.5.0 CUDA-C: Malbec
// Dr. Gonzalo Damián Quiroga
// Universidad Industrial de Santander

void rhs(double rot, double x0, double x1, double x2, double x3, double px0, double px1, double px2, double px3,
	double *dx0, double *dx1, double *dx2, double *dx3, double *dpx0, double *dpx1, double *dpx2, double *dpx3)
{
	//Schwarzschild
	if(rot==0)
	{
		#include "metrics/sch.inc"		
	}
	
	//Kerr
	else if(rot!=0)
	{
		#include "metrics/kerr.inc"
	}
}
