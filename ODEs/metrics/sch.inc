*dx0= px0/(1 - (2)/x1);
*dx1= px1*(1 - (2)/x1);
*dx2= px2/pow(x1,2);
*dx3= (px3*pow(sin(x2),-2))/pow(x1,2);
*dpx0=0;
*dpx1=-((pow(px0,2))/(pow(1 - (2)/x1,2)*pow(x1,2))) +((2*pow(px2,2))/pow(x1,3) - (2*pow(px1,2))/pow(x1,2) +(2*pow(px3,2)*pow(sin(x2),-2))/pow(x1,3))/2.0;
*dpx2= (pow(px3,2)*cos(x2)*pow(sin(x2),-2))/(sin(x2)*pow(x1,2));
*dpx3=0;
