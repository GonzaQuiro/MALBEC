/*
 * kernel.h
 *
 *  Created on: 01/03/2018
 *      Author: Gonzalo Damian Quiroga
 */

#ifndef KERNEL_H_
#define KERNEL_H_

__global__ void rk4(double Horizon, double *var0, double *var1, double *var2, double *var3, double *var4, double *var5, double *var6, double *var7, int n_ic, double h);
//__global__ void rkf45(double Rot,double *var0, double *var1, double *var2, double *var3, double *var4, double *var5, double *var6, double *var7, int n_ic, double *h,double errorTol);

#endif /* KERNEL_H_ */
