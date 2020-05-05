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
 

__kernel void time_dev(__global double *f,
                       __global double *fh,
                       __global double *newf,
                       __global double *newfh,
                       __global double *omega,
                       __global ulong *rand1,
                       __global int *stable,
                       __constant double *deltaTau,
                       __global int *lrgEl,
                       __global double *lrgVl,
                       __global int *sameness,
                       __constant int *LIST_SIZE,
                       __constant double *deltaT,
                       __constant double *hPar,
                       __constant int *potential,
                       __constant double *C,
                       __constant int *Loops
                      ) 
{
    
    
    // Get the index of the current element
    int i = get_global_id(0);
//     int N = get_global_size(0);
    int N = *LIST_SIZE;
    double deltatau = *deltaTau;
    double deltat = *deltaT;
    double om = *omega;
    double m = 1.;
    double h = *hPar;
    int potID = *potential;
    double c = *C;
    int loops = *Loops;
    double newomega;
    int boundaryConditions = 2;
    double dw;
    
    for(int j=0; j<loops; j++){
        om = *omega;
    //     double dw = random(rand1, i);
        
    //     double dw = c*(double)sqrt((float)(2.*deltatau/deltat))*rand1[i];

        
        if(i==0){
            dw = c*(double)sqrt((float)(2.*deltatau/deltat))*random(rand1, i);
            if(boundaryConditions == 2){
                newf[i] = boundary(-1, potID);
            }
            if(boundaryConditions == 1){
                newf[i] = f[0]+m*deltatau*(f[1]+boundary(-1, potID)-clas(-1.*deltat, om, potID)-2*f[0])/(double)pown((float)deltat,2)-ddPot(clas((double)i*deltat, om, potID), potID) * f[0] *deltatau + dw;
                
                newfh[i] = fh[0]+m*deltatau*(fh[1]+boundary(-1, potID)-clas(-1.*deltat, om, potID)-2*fh[0])/(double)pown((float)deltat,2) -ddPot(clas((double)i*deltat, om, potID), potID)* fh[0] *deltatau + dw ;
                
            }
            if(boundaryConditions == 0){
                newf[i] = f[0]+m*deltatau*(f[1]+f[N-1]-2*f[0])/(double)pown((float)deltat,2)-ddPot(clas((double)i*deltat, om, potID), potID) * f[0] *deltatau + dw;
                
                newfh[i] = fh[0]+m*deltatau*(fh[1]+fh[N-1]-2*fh[0])/(double)pown((float)deltat,2) -ddPot(clas((double)i*deltat, om, potID), potID)* fh[0] *deltatau + dw ;
            }
            
            
//             newf[i] = f[0]+m*deltatau*(f[1]-f[0])/deltat-ddPot(clas((double)i*deltat, om, potID)+f[0], potID) * f[0] *deltatau + dw;
//             newf[i] = boundary(-1, potID)-clas((double)i*deltat, om, potID) + dw;
    //         newfh[i] = fh[0]+m*deltatau*(fh[1]-fh[0])/deltat -ddPot(clas((double)i*deltat, om, potID)+fh[0], potID)* fh[0] *deltatau + dw ;
    //         newfh[i] = boundary(-1, potID)-clas((double)i*deltat, om, potID) + dw;
            
            
        }
        if(i==N-1){
            dw = c*(double)sqrt((float)(2.*deltatau/deltat))*random(rand1, i);
            if(boundaryConditions == 2){
                newf[i] = boundary(1, potID);
            }
            if(boundaryConditions == 1){
                newf[i] = f[N-1]+m*deltatau*(boundary(1, potID)-clas((double)N*deltat, om, potID)+f[N-2]-2*f[N-1])/(double)pown((float)deltat,2)-ddPot(clas((double)i*deltat, om, potID), potID)* f[N-1] *deltatau + dw;
            
                newfh[i] = fh[N-1]+m*deltatau*(boundary(1, potID)-clas((double)N*deltat, om, potID)+fh[N-2]-2*fh[N-1])/(double)pown((float)deltat,2) -ddPot(clas((double)i*deltat, om, potID), potID)* fh[N-1] *deltatau + dw;
            }
            if(boundaryConditions == 0){
                newf[i] = f[N-1]+m*deltatau*(f[0]+f[N-2]-2*f[N-1])/(double)pown((float)deltat,2)+ddPot(clas((double)i*deltat, om, potID), potID)* f[N-1] *deltatau + dw;
            
                newfh[i] = fh[N-1]+m*deltatau*(fh[0]+fh[N-2]-2*fh[N-1])/(double)pown((float)deltat,2) -ddPot(clas((double)i*deltat, om, potID), potID)* fh[N-1] *deltatau + dw;
            }
            
//             newf[i] = f[N-1]+m*deltatau*(f[N-2]-f[N-1])/deltat+ddPot(clas((double)i*deltat, om, potID)+f[N-1], potID)* f[N-1] *deltatau + dw;
//             newf[i] = boundary(1, potID)-clas((double)i*deltat, om, potID) + dw;
            
//             newfh[i] = fh[N-1]+m*deltatau*(fh[N-2]-fh[N-1])/deltat -ddPot(clas((double)i*deltat, om, potID)+fh[N-1], potID)* fh[N-1] *deltatau + dw;
//             newfh[i] = boundary(1, potID)-clas((double)i*deltat, om, potID) + dw;
                
        }
        if(i==N){
            dw = c*(double)sqrt((float)(2.*deltatau))*random(rand1, i);
            
            newomega = om + intConst(3)*dw;
//             printf("%f\n", newomega);
        }
        if(i<N-1&&i>0){
            dw = c*(double)sqrt((float)(2.*deltatau/deltat))*random(rand1, i);
            
            newf[i] = f[i]+m*deltatau*(f[i+1]+f[i-1]-2*f[i])/(double)pown((float)deltat,2)-ddPot(clas((double)i*deltat, om, potID), potID) * f[i] *deltatau + dw;
            
            newfh[i] = fh[i]+m*deltatau*(fh[i+1]+fh[i-1]-2*fh[i])/(double)pown((float)deltat,2) -ddPot(clas((double)i*deltat, om, potID), potID)* fh[i] *deltatau + dw;
        }
            
        
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
            
            if (isinf((float)newf[i])==1||isnan((float)newf[i])==1||isinf((float)newfh[i])==1||isnan((float)newfh[i])==1){
//                 newf[i] = f[i];
//                 newfh[i] = fh[i];
                newf[i] = 0;
                newfh[i] = 0;
//                 *stable=-1;
            }
            if (newf[i]==newfh[i]&&i!=0&&i!=N-1){
                newfh[i]=newf[i]+h;
        //         *stable=-2;
            }
            if(newfh[i]-newf[i]==fh[i]-f[i]){
                *sameness*=1;
            }
            else{
                *sameness=0;
            }
            
            if(newf[i]>newf[*lrgEl]){
                *lrgEl=i;
                if(absol(newf[i]-f[i]-dw)>*lrgVl){
//                     newf[i]=f[i];
                    *stable=0;
                }
            }
            if(absol(newf[i])>*lrgVl){
                *lrgVl=absol(newf[i]);
            }
            if(j<loops-1){
                f[i] = newf[i];
                fh[i] = newfh[i];
            }
            else{
                if(i==N-1){
//                     printf("%f\n",newf[i]);
                }
            }
        }
        else{
            *omega = newomega;            
//             printf("%f\n",*omega);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        if(*stable!=1){
            break;
        }
        
    }
//     barrier(CLK_GLOBAL_MEM_FENCE);
//     newf[i]=dw;
//     newfh[i]=h;
    
}








double doubleWellSol(double t, double t0){
//     float x0=1.;
    double m = 1.;
    double eta = 10.;
    double lbda = .5;
    return eta * (double)tanh((float)(eta*(double)sqrt((float)(2.*lbda/m))*(t-t0)));
    
}
double doubleWellPot(double a){
//     float x0=1.;
    double lbda = .5;
    double eta = 10.;
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
    double eta = 10.;
    double lbda = .5;
    double m = 1.;
    return 1./(double)sqrt((float)((double)pown((float)eta,3)*(double)pow((float)(lbda/m),(float)(3./2.))*(double)sqrt((float)2.)*4./3.));
    
}
double boundary(int rl, int pot){
    double eta = 10.;
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
    
    seed[gid] = ((seed[gid]+(ulong)gid) * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
    double v1 = (double)(seed[gid] >> 16) / (double)pown((float)2,32);
    seed[gid] = ((seed[gid]+(ulong)gid) * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
    double v2 = (double)(seed[gid] >> 16) / (double)pown((float)2,32);
    
//     return v1;
    return (double)cos((float)(2.*3.14*v2))*(double)sqrt((float)(-2.*(double)log((float)v1)));
}

