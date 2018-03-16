// Version 0.5.0 CUDA-C: Malbec. Runge-Kutta 4 method
// Dr. Gonzalo Damián Quiroga
// Universidad Industrial de Santander

void rk4(double Rot, double *x0, double *x1, double *x2, double *x3, double *px0, double *px1, double *px2, double *px3, int n_ic, double h)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	double k1[8], k2[8], k3[8], k4[8];
	double dx0, dx1, dx2, dx3, dpx0, dpx1, dpx2, dpx3;

	if (id < n_ic)
	{	
		//Coefficinet k1:
		rhs(Rot, x0[id], x1[id], x2[id], x3[id], px0[id], px1[id], px2[id], px3[id], &dx0, &dx1, &dx2, &dx3, &dpx0, &dpx1, &dpx2, &dpx3);

		k1[0] = h*dx0;
		k1[1] = h*dx1;
		k1[2] = h*dx2;
		k1[3] = h*dx3;
		k1[4] = h*dpx0;
		k1[5] = h*dpx1;
		k1[6] = h*dpx2;
		k1[7] = h*dpx3;

		//Coefficinet k2:
		rhs(Rot, x0[id] + 0.5*k1[0], x1[id] + 0.5*k1[1], x2[id] + 0.5*k1[2], 
		x3[id] + 0.5*k1[3],px0[id] + 0.5*k1[4], px1[id] + 0.5*k1[5], px2[id] + 0.5*k1[6], px3[id] + 0.5*k1[7],
		&dx0, &dx1, &dx2, &dx3, &dpx0, &dpx1, &dpx2, &dpx3);

		k2[0] = h*dx0;
		k2[1] = h*dx1;
		k2[2] = h*dx2;
		k2[3] = h*dx3;
		k2[4] = h*dpx0;
		k2[5] = h*dpx1;
		k2[6] = h*dpx2;
		k2[7] = h*dpx3;
		
		//Coefficinet k3:
		rhs(Rot, x0[id] + 0.5*k2[0], x1[id] +0.5*k2[1], x2[id]+0.5*k2[2], 
		x3[id]+0.5*k2[3], px0[id]+0.5*k2[4], px1[id]+0.5*k2[5], px2[id]+0.5*k2[6], px3[id]+0.5*k2[7],
		&dx0, &dx1, &dx2, &dx3, &dpx0, &dpx1, &dpx2, &dpx3);

		k3[0] = h*dx0;
		k3[1] = h*dx1;
		k3[2] = h*dx2;
		k3[3] = h*dx3;
		k3[4] = h*dpx0;
		k3[5] = h*dpx1;
		k3[6] = h*dpx2;
		k3[7] = h*dpx3;

		//Coefficinet k4:
		rhs(Rot, x0[id]+k3[0],x1[id]+k3[1], x2[id]+k3[2], x3[id]+k3[3],
		px0[id]+k3[4], px1[id]+k3[5],px2[id]+k3[6], px3[id]+k3[7], 
		&dx0, &dx1, &dx2, &dx3, &dpx0, &dpx1, &dpx2, &dpx3);

		k4[0] = h*dx0;
		k4[1] = h*dx1;
		k4[2] = h*dx2;
		k4[3] = h*dx3;
		k4[4] = h*dpx0;
		k4[5] = h*dpx1;
		k4[6] = h*dpx2;
		k4[7] = h*dpx3;

		
		if((x1[id]+(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])/6.0) >= (1+sqrt(1-Rot*Rot)))
		{
			x0[id]=x0[id]+(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])/6.0;
			x1[id]=x1[id]+(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])/6.0;
			x2[id]=x2[id]+(k1[2]+2.0*k2[2]+2.0*k3[2]+k4[2])/6.0;
			x3[id]=x3[id]+(k1[3]+2.0*k2[3]+2.0*k3[3]+k4[3])/6.0;
			px0[id]=px0[id]+(k1[4]+2.0*k2[4]+2.0*k3[4]+k4[4])/6.0;
			px1[id]=px1[id]+(k1[5]+2.0*k2[5]+2.0*k3[5]+k4[5])/6.0;
			px2[id]=px2[id]+(k1[6]+2.0*k2[6]+2.0*k3[6]+k4[6])/6.0;
			px3[id]=px3[id]+(k1[7]+2.0*k2[7]+2.0*k3[7]+k4[7])/6.0;
		}
		else
		{
			return;
		}
	}
}

