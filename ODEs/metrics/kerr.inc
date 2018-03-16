*dx0= px0 + (4*x1*(pow(rot,2)*px0 - rot*px3 + px0*pow(x1,2)))/((pow(rot,2) + x1*(-2 + x1))*(pow(rot,2) + 2*pow(x1,2) + pow(rot,2)*cos(2*x2)));
*dx1= (px1*(pow(rot,2) - 2*x1 + pow(x1,2)))/(pow(x1,2) + pow(rot,2)*pow(cos(x2),2));
*dx2= px2/(pow(x1,2) + pow(rot,2)*pow(cos(x2),2));
*dx3= (-2*rot*(rot*px3 - 2*px0*x1) + 2*px3*(pow(rot,2) + x1*(-2 + x1))*pow(sin(x2),-2))/((pow(rot,2) + x1*(-2 + x1))*(pow(rot,2) + 2*pow(x1,2) + pow(rot,2)*cos(2*x2)));
*dpx0=0;
*dpx1= (-2*(-(pow(rot,6)*pow(px0,2)) - pow(rot,6)*pow(px1,2) + 2*pow(rot,5)*px0*px3 - pow(rot,4)*pow(px3,2) - pow(rot,6)*pow(px1,2)*x1 + 
    	4*pow(rot,4)*pow(px1,2)*x1 - 2*pow(rot,4)*pow(px2,2)*x1 + 3*pow(rot,4)*pow(px3,2)*x1 + 4*pow(rot,4)*pow(px1,2)*pow(x1,2) - 
       	4*pow(rot,2)*pow(px1,2)*pow(x1,2) + 8*pow(rot,2)*pow(px2,2)*pow(x1,2) - 6*pow(rot,3)*px0*px3*pow(x1,2) - 
       	6*pow(rot,2)*pow(px3,2)*pow(x1,2) - 4*pow(rot,2)*pow(px0,2)*pow(x1,3) - 2*pow(rot,4)*pow(px1,2)*pow(x1,3) - 
       	8*pow(rot,2)*pow(px1,2)*pow(x1,3) - 4*pow(rot,2)*pow(px2,2)*pow(x1,3) - 8*pow(px2,2)*pow(x1,3) + 16*rot*px0*px3*pow(x1,3) + 
       	4*pow(rot,2)*pow(px3,2)*pow(x1,3) + 3*pow(rot,2)*pow(px0,2)*pow(x1,4) + 7*pow(rot,2)*pow(px1,2)*pow(x1,4) + 8*pow(px1,2)*pow(x1,4) + 
       	8*pow(px2,2)*pow(x1,4) - 12*rot*px0*px3*pow(x1,4) - pow(rot,2)*pow(px1,2)*pow(x1,5) - 8*pow(px1,2)*pow(x1,5) - 2*pow(px2,2)*pow(x1,5) + 
       	2*pow(px0,2)*pow(x1,6) + 2*pow(px1,2)*pow(x1,6) - pow(rot,6)*pow(px0,2)*cos(2*x2) - pow(rot,6)*pow(px1,2)*cos(2*x2) + 
       	2*pow(rot,5)*px0*px3*cos(2*x2) - pow(rot,4)*pow(px3,2)*cos(2*x2) + pow(rot,6)*pow(px1,2)*x1*cos(2*x2) + 4*pow(rot,4)*pow(px1,2)*x1*cos(2*x2) + 
       	pow(rot,4)*pow(px3,2)*x1*cos(2*x2) - 2*pow(rot,4)*pow(px0,2)*pow(x1,2)*cos(2*x2) - 6*pow(rot,4)*pow(px1,2)*pow(x1,2)*cos(2*x2) - 
       	4*pow(rot,2)*pow(px1,2)*pow(x1,2)*cos(2*x2) - 2*pow(rot,3)*px0*px3*pow(x1,2)*cos(2*x2) + 4*pow(rot,2)*pow(px0,2)*pow(x1,3)*cos(2*x2) + 
       	2*pow(rot,4)*pow(px1,2)*pow(x1,3)*cos(2*x2) + 8*pow(rot,2)*pow(px1,2)*pow(x1,3)*cos(2*x2) - pow(rot,2)*pow(px0,2)*pow(x1,4)*cos(2*x2) - 
       	5*pow(rot,2)*pow(px1,2)*pow(x1,4)*cos(2*x2) + pow(rot,2)*pow(px1,2)*pow(x1,5)*cos(2*x2) - 2*pow(rot,4)*pow(px3,2)*x1*pow(sin(x2),-2) + 
       	8*pow(rot,2)*pow(px3,2)*pow(x1,2)*pow(sin(x2),-2) - 4*pow(rot,2)*pow(px3,2)*pow(x1,3)*pow(sin(x2),-2) - 
       	8*pow(px3,2)*pow(x1,3)*pow(sin(x2),-2) + 8*pow(px3,2)*pow(x1,4)*pow(sin(x2),-2) - 2*pow(px3,2)*pow(x1,5)*pow(sin(x2),-2)))/
       	(pow(pow(rot,2) - 2*x1 + pow(x1,2),2)*pow(pow(rot,2) + 2*pow(x1,2) + pow(rot,2)*cos(2*x2),2));
*dpx2= (-2*(-2*pow(rot,2)*pow(px3,2)*pow(x1,2)*cos(x2)*pow(sin(x2),-3) + 4*pow(px3,2)*pow(x1,3)*cos(x2)*pow(sin(x2),-3) - 
       	2*pow(px3,2)*pow(x1,4)*cos(x2)*pow(sin(x2),-3) - 2*pow(rot,4)*pow(px3,2)*cos(2*x2)*cos(x2)*pow(sin(x2),-3) + 
       	4*pow(rot,2)*pow(px3,2)*x1*cos(2*x2)*cos(x2)*pow(sin(x2),-3) - 2*pow(rot,2)*pow(px3,2)*pow(x1,2)*cos(2*x2)*cos(x2)*pow(sin(x2),-3) + 
       	pow(rot,6)*pow(px1,2)*sin(2*x2) + pow(rot,4)*pow(px2,2)*sin(2*x2) - pow(rot,4)*pow(px3,2)*sin(2*x2) - 2*pow(rot,4)*pow(px0,2)*x1*sin(2*x2) - 
       	4*pow(rot,4)*pow(px1,2)*x1*sin(2*x2) - 2*pow(rot,2)*pow(px2,2)*x1*sin(2*x2) + 4*pow(rot,3)*px0*px3*x1*sin(2*x2) + 
       	2*pow(rot,4)*pow(px1,2)*pow(x1,2)*sin(2*x2) + 4*pow(rot,2)*pow(px1,2)*pow(x1,2)*sin(2*x2) + pow(rot,2)*pow(px2,2)*pow(x1,2)*sin(2*x2) - 
       	2*pow(rot,2)*pow(px0,2)*pow(x1,3)*sin(2*x2) - 4*pow(rot,2)*pow(px1,2)*pow(x1,3)*sin(2*x2) + pow(rot,2)*pow(px1,2)*pow(x1,4)*sin(2*x2)))/
      	((pow(rot,2) - 2*x1 + pow(x1,2))*pow(pow(rot,2) + 2*pow(x1,2) + pow(rot,2)*cos(2*x2),2));
*dpx3=0;
