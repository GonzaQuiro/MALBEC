// Version 0.5.0 CUDA-C: Malbec
// Dr. Gonzalo Damián Quiroga
// Universidad Industrial de Santander
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include "../ODEs/rhs.h"

__global__ 
#include "../solvers/rk4.cu"

//__global__ 
//#include "rkf45.cu"

__device__ 
#include "../ODEs/rhs.c"

