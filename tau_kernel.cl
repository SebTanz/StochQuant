#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable

double doubleWellSol(double t, double t0);
double doubleWellPot(double a);


double clas(double a, double w, int pot);
double ddPot(double a, int pot);
double intConst (int potID);
double boundary(int rl, int pot);
double absol(double a);
double random(__global ulong * seed, int gid);
__constant double eta = 1.;
__constant double lbda = 100.;
__constant double m = 1.;


__kernel void time_dev(__global double *f,
                       __global double *x,
                       __global double *xx0,
                       __global double *newf,
                       __global double *newx,
                       __global double *newxx0,
                       __global double *omega,
                       __global ulong *rand1,
                       __global int *stable,
                       __constant double *deltaTau,
                       __global int *lrgEl,
                       __global double *lrgVl,
                       __global int *initRun,
                       __constant int *LIST_SIZE,
                       __constant double *deltaT,
                       __global int *runs,
                       __constant int *potential,
                       __constant double *C,
                       __constant int *Loops
                      ) 
{
    
    
    // Get the index of the current element
    int i = get_global_id(0);
    int M = get_global_size(0);
    int N = *LIST_SIZE;
    double deltatau = *deltaTau;
    double deltat = *deltaT;
    double om = *omega;
//     double m = 1.;
    int potID = *potential;
    double c = *C;
    int loops = *Loops;
    double newomega;
    int boundaryConditions = 1;
    double dw;
    double max = 1000;
    int midpt = N/2;
    
    for(int j=0; j<loops; j++){
        om = *omega;
    //     double dw = random(rand1, i);
        
    //     double dw = c*(double)sqrt((float)(2.*deltatau/deltat))*rand1[i];

        
        if(i==0){
            dw = c*(double)sqrt((float)(2.*deltatau/deltat))*random(rand1, i);
            if(boundaryConditions == 2){
                newf[i] = 0;
            }
            if(boundaryConditions == 1){
                newf[i] = f[0]+m*deltatau*(f[1]+boundary(-1, potID)-clas(-1.*deltat, om, potID)-2*f[0])/(double)pown((float)deltat,2)-ddPot(clas((double)i*deltat, om, potID), potID) * f[0] *deltatau + dw;
                
                
            }
            if(boundaryConditions == 0){
                newf[i] = f[0]+m*deltatau*(f[1]+f[N-1]-2*f[0])/(double)pown((float)deltat,2)-ddPot(clas((double)i*deltat, om, potID), potID) * f[0] *deltatau + dw;
                
            }
            
            
            
        }
        if(i==N-1){
            dw = c*(double)sqrt((float)(2.*deltatau/deltat))*random(rand1, i);
            if(boundaryConditions == 2){
                newf[i] = 0;
            }
            if(boundaryConditions == 1){
                newf[i] = f[N-1]+m*deltatau*(f[N-2]+boundary(1, potID)-clas((double)N*deltat, om, potID)-2*f[N-1])/(double)pown((float)deltat,2)-ddPot(clas((double)i*deltat, om, potID), potID)* f[N-1] *deltatau + dw;
            
            }
            if(boundaryConditions == 0){
                newf[i] = f[N-1]+m*deltatau*(f[0]+f[N-2]-2*f[N-1])/(double)pown((float)deltat,2)+ddPot(clas((double)i*deltat, om, potID), potID)* f[N-1] *deltatau + dw;
            
            }
            
            
                
        }
        if(i==N){
            
//             rand1[i]+=1;
            dw = c*(double)sqrt((float)(2.*deltatau))*random(rand1, i);
//                 printf("%f\n", dw);
            newomega = om + intConst(3)*dw;
//                 if(rand1[i]>0)
//             printf("%lu\n", rand1[i-1]);
            
            
        }
        if(i<N-1&&i>0){
            dw = c*(double)sqrt((float)(2.*deltatau/deltat))*random(rand1, i);
//             printf("%f ",dw);
            
            newf[i] = f[i]+m*deltatau*(f[i+1]+f[i-1]-2*f[i])/(double)pown((float)deltat,2)-ddPot(clas((double)i*deltat, om, potID), potID) * f[i] *deltatau + dw;
            
            
        }
            
//         if(i==15){
//             for(int k=0; k<N+1; k++){
//                 printf("%lu,", rand1[k]);
//             }
//             printf("%d, %d\n", i,N);
//             
//         }
        if(i<N){
//             if(i==*lrgEl){
//                 if(absol(newf[i]-f[i]-dw)>*lrgVl){
// //                     newf[i]=f[i];
//                     *stable=0;
//                 }
//                 else{
//                     *stable=1;
//                 }
//             }
            
            if(newf[i]>max){
                newf[i] = max;
                
            }
            if(newf[i]<-max){
                newf[i] = -max;
                
            }
            if (isinf((float)newf[i])==1||isnan((float)newf[i])==1){
                newf[i] = max;
//                 newf[i] = 0;
//                 *stable=-1;
            }
            
            
            if(newf[i]+clas((double)i*deltat, om, potID)>newf[*lrgEl]+clas((double)*lrgEl*deltat, om, potID)){
                *lrgEl=i;
                if(absol(newf[i]-f[i]-dw)>*lrgVl){
//                     newf[i]=f[i];
                    *stable=0;
                }
            }
            if(absol(newf[i]+clas((double)i*deltat, om, potID))>*lrgVl){
                *lrgVl=absol(newf[i]+clas((double)i*deltat, om, potID));
            }
            newxx0[i] = xx0[i] + ((f[i]+clas((double)i*deltat, om, potID))*(f[midpt]+clas((double)midpt*deltat, om, potID))-xx0[i])/((double)(*runs+j+1));
            newx[i] = x[i] + ((f[i]+clas((double)i*deltat, om, potID))-x[i])/((double)(*runs+j+1));
            
            if(j<loops-1){
                f[i] = newf[i];
                xx0[i] = newxx0[i];
                x[i] = newx[i];
                
            }
            
            
        }
        else{
            if(newomega>(double)(N-1)*deltat){
                *omega = 2*(double)(N-1)*deltat - newomega;
            }
            else{
                if(newomega<0){
                    *omega = -newomega;
                }
                else{
                    *omega = newomega;
                }
            }
//             for(int k=0; k<M; k++){
//                 printf("%lu, ", rand1[k]);
//             }
//             printf("%d, %d\n", i,M);
            
//             printf("%f\n",*omega);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        if(*stable!=1){
            break;
        }
        
    }
    
//     barrier(CLK_GLOBAL_MEM_FENCE);
//     newf[i]=dw;
    
}








double doubleWellSol(double t, double t0){
//     float x0=1.;
//     double m = 1.;
//     double eta = 10.;
//     double lbda = .5;
    return eta * (double)tanh((float)(eta*(double)sqrt((float)(2.*lbda/m))*(t-t0)));
    
}
double doubleWellPot(double a){
//     float x0=1.;
//     double lbda = .5;
//     double eta = 10.;
    return 12.*lbda*a*a-4.*lbda*eta*eta;
    
}


double clas(double a, double w, int pot){
    
    if(pot==3){
        
        return doubleWellSol(a, w);
    }
    else{
        return doubleWellSol(a, w);
    }

    
}
double ddPot(double a, int pot){

    if(pot==3){
        
        return doubleWellPot(a);
    }
    else{
        return doubleWellPot(a);
    }

    
}
double intConst(int potID){
//     double eta = 10.;
//     double lbda = .5;
//     double m = 1.;
    return 1./(double)sqrt((float)((double)pown((float)eta,3)*(double)pow((float)(lbda/m),(float)(3./2.))*(double)sqrt((float)2.)*4./3.));
    
}
double boundary(int rl, int pot){
//     double eta = 10.;
    if (rl==1){
        return eta;
    }
    if (rl==-1){
        return -eta;
    }
    
}


double absol(double a){
    
    if (a<=0){
        return -a;
    }
    else{
        return a;
    }
}
 
double random(__global ulong * seed, int gid){
    double result;
    ulong temp;
    do{
        temp = ((*seed+(ulong)gid) * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
        double v1 = (double)(temp >> 16) / (double)pown((float)2,32);
        temp = ((temp+(ulong)gid) * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
        double v2 = (double)(temp >> 16) / (double)pown((float)2,32);
        result = (double)cos((float)(2.*3.1415*v2))*(double)sqrt((float)(-2.*(double)log((float)v1)));
        if(*seed<(ulong)pown((float)2,31)&&temp<(ulong)pown((float)2,31))
            *seed +=temp;
        else
            *seed = temp-(ulong)pown((float)2,31);
    }while(isinf((float)result));
    return result;
}

