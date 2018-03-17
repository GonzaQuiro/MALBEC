// Version 0.5.0 CUDA-C: Malbec
// Dr. Gonzalo Damián Quiroga
// Universidad Industrial de Santander
#include <cuda.h>
#include <stdio.h>
#include "common/handle.h"
#include "common/menu.h"
#include "kernel/kernel.h"
#include "kernel/kernel.cu"
#include "common/menu.c"



int main(int argc,char **argv)
{		
	//Setting the Spacetime Parameters and Initial conditions
	double Rot;
	menu(&Rot);
	int n_ic;
	printf("Set the number of initial conditions: ");
	scanf("%d", &n_ic);

	//Solver settings
	double h, final_time;
	double evol_time = 0.0;
	//Integration steps
	printf("Final Time: ");
	scanf("%lf", &final_time);
	printf("Initial step size: ");
	scanf("%lf", &h);

	// Host input/output vectors	
	double *h_x0, *h_x1, *h_x2, *h_x3,*h_px0, *h_px1,*h_px2, *h_px3;
	 
	// Device input/output vectors  
	double *d_x0, *d_x1, *d_x2, *d_x3,*d_px0, *d_px1,*d_px2, *d_px3;
	
	// Size, in bytes, of each vector
	double nBytes = n_ic*sizeof(double);

	// Allocate memory for each vector on host
	h_x0 = (double *)malloc(nBytes);
	h_x1= (double *)malloc(nBytes);
	h_x2 = (double *)malloc(nBytes);
	h_x3 = (double *)malloc(nBytes);
	h_px0= (double *)malloc(nBytes);
	h_px1 = (double *)malloc(nBytes);
	h_px2= (double *)malloc(nBytes);
	h_px3 = (double *)malloc(nBytes);	
	
	// Allocate memory for each vector on GPU
	printf("Allocating device memory on host..\n");
	HANDLE_ERROR(cudaMalloc((void **)&d_x0,nBytes));
	HANDLE_ERROR(cudaMalloc((void **)&d_x1,nBytes));
	HANDLE_ERROR(cudaMalloc((void **)&d_x2,nBytes));
	HANDLE_ERROR(cudaMalloc((void **)&d_x3,nBytes));
	HANDLE_ERROR(cudaMalloc((void **)&d_px0,nBytes));
	HANDLE_ERROR(cudaMalloc((void **)&d_px1,nBytes));
	HANDLE_ERROR(cudaMalloc((void **)&d_px2,nBytes));
	HANDLE_ERROR(cudaMalloc((void **)&d_px3,nBytes));
	
	// Initial conditions on host
	FILE *fic = fopen("ic/ic.txt", "r");
	if (fic == NULL)
	{
		perror("Error: can't open ic.txt.");
		return -1;
	}

	//Read the initial conditions
	int count;
	for (count = 0; count < n_ic; count++)
	{
		fscanf(fic, "%lf %lf %lf %lf %lf %lf %lf %lf", &h_x0[count], &h_x1[count], &h_x2[count], &h_x3[count], &h_px0[count], &h_px1[count], &h_px2[count], &h_px3[count]);
	}
	//Close the file
	fclose(fic);

	//Set the Block and grid Size
	int blockSize, gridSize,minGridSize; 

	// Number of threads in each thread block.
	//Suggested block size to achieve maximum occupancy.
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, rk4, 0, n_ic);
	gridSize = n_ic/blockSize;
	dim3 dimBlock(blockSize,1,1);
	dim3 dimGrid(gridSize,1,1);
	 
	// Copy host vectors to device 
	printf("Copying to device..\n");
	HANDLE_ERROR(cudaMemcpy(d_x0,h_x0,nBytes,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_x1,h_x1,nBytes,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_x2,h_x2,nBytes,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_x3,h_x3,nBytes,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_px0,h_px0,nBytes,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_px1,h_px1,nBytes,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_px2,h_px2,nBytes,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_px3,h_px3,nBytes,cudaMemcpyHostToDevice));

	//Evol: Time Start
   	clock_t start_d=clock();
	
	//System Evolution
	printf("Executing Kernel.. \n");
	printf("Output file created.. \n");
	//Open fout
	FILE *fout = fopen("output/output.txt", "w");
	if (fout == NULL)
	{
		perror("Error: the output folder does not exist.");
		return -1;
	}
	//Print the initial conditions in the output file
	int i;	
	for (i = 0; i < n_ic; i++)
	{
		fprintf(fout, "%.8lf %.16lf %.16lf %.16lf %.16lf \n", evol_time, h_x0[i], sqrt(h_x1[i]*h_x1[i]+Rot*Rot)*sin(h_x2[i])*cos(h_x3[i]), 				
		sqrt(h_x1[i]*h_x1[i]+Rot*Rot)*sin(h_x2[i])*sin(h_x3[i]), h_x1[i]*cos(h_x2[i]));	
	}
	evol_time=h+evol_time;
	printf("Evolving the systems.. \n");
	do
	{
		// Executing kernel
		rk4<<<gridSize,blockSize>>>(Rot, d_x0,d_x1,d_x2,d_x3,d_px0,d_px1,d_px2,d_px3, n_ic, h);
		cudaThreadSynchronize();
		// Copy array back to host	
		HANDLE_ERROR(cudaMemcpy(h_x0,d_x0,nBytes,cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(h_x1,d_x1,nBytes,cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(h_x2,d_x2,nBytes,cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(h_x3,d_x3,nBytes,cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(h_px0,d_px0,nBytes,cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(h_px1,d_px1,nBytes,cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(h_px2,d_px2,nBytes,cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(h_px3,d_px3,nBytes,cudaMemcpyDeviceToHost));

		//Print in file the initial conditions in Cartesian Coordinates
		for (i = 0; i < n_ic; i++)
		{
			fprintf(fout, "%.8lf %.16lf %.16lf %.16lf %.16lf \n", evol_time, h_x0[i], sqrt(h_x1[i]*h_x1[i]+Rot*Rot)*sin(h_x2[i])*cos(h_x3[i]), 				
			sqrt(h_x1[i]*h_x1[i]+Rot*Rot)*sin(h_x2[i])*sin(h_x3[i]), h_x1[i]*cos(h_x2[i]));
			
		}	
		evol_time=h+evol_time;
	}while(evol_time <= final_time);

	
	
	
	//Evol: Time Ends
   	clock_t end_d = clock();
   	double time_spent = (double)(end_d-start_d)/CLOCKS_PER_SEC;

	//Close the output file
	fclose(fout);
			
	// Release device memory
   	cudaFree(d_x0);
	cudaFree(d_x1);
	cudaFree(d_x2);
	cudaFree(d_x3);
   	cudaFree(d_px0);
	cudaFree(d_px1);
	cudaFree(d_px2);
	cudaFree(d_px3);

	// Release host memory
	free(h_x0);	
	free(h_x1);
	free(h_x2);
	free(h_x3);
	free(h_px0);	
	free(h_px1);
	free(h_px2);
	free(h_px3);

	//Log file.
	printf("Printing info.log file.. \n");
	FILE *flog = fopen("info.log", "w");
	fprintf(flog, "Mass: 1, Rot: %lf \n", Rot);
	fprintf(flog, "Initial step size: %lf, Final time: %lf, Initial Conditions: %d \n", h, final_time, n_ic);		
	fprintf(flog, "Integrator: RK4 Fixed Step. Number of Thread per block: %d \n", blockSize);
	fprintf(flog, "Runtime: %f sec.", time_spent);
	fclose(flog); 
	printf("Runtime: %f sec \n", time_spent);
		
	return 0;
}
