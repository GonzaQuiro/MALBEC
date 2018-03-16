// Version 0.5.0 CUDA-C: Malbec
// Dr. Gonzalo Damián Quiroga
// Universidad Industrial de Santander
#include <stdio.h>

void menu(double *Rot)
{
	printf("*****************************************\n");
	printf("*		Malbec v0.5.0		*\n");
	printf("*****************************************\n");
	double rot;
	int choice;
 	reboot:
	/*Displaing on screen*/
	printf("-------Menu-------\n");
	printf("1) Schwarzschild BH. \n");
	printf("2) Kerr BH. \n");
 
	/*getting input*/
	scanf("%d", &choice);
 
 	switch (choice)
        {
		//Sch BH
		case 1:
			*Rot=0.0;
			break;
		//Kerr BH
		case 2:
			do
			{
			printf("Enter the Kerr parameter: ");
			scanf("%lf", &rot);
			if(rot>1.0 || rot<-1.0 || rot==0.0)
				printf("The kerr parameter must be between -1 and 1 and not null. \n");
			else
				*Rot=rot;
			}while(rot>1.0 || rot<-1.0 || rot==0.0);

			break;
		default:
			printf("\n");
			printf("Error: Is not a valid option. \n");
			printf("\n");
			goto reboot;
        }
}
