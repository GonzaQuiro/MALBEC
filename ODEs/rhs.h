/*
 * rhs.h
 *
 *  Created on: 01/03/2018
 *      Author: Dr. Gonzalo Damián Quiroga
 */

#ifndef RHS_H_
#define RHS_H_

__device__ void rhs(double rot, double x0, double x1, double x2, double x3, double px0, double px1, double px2, double px3,
	double *dx0, double *dx1, double *dx2, double *dx3, double *dpx0, double *dpx1, double *dpx2, double *dpx3);
#endif /* RHS_H_ */
