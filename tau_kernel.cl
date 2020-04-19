#pragma OPENCL EXTENSION cl_khr_fp64 : enable



double dharmosc(double a);
double poeschlTeller(double a);
double quartic(double a);

double doubleWellSol(double t, double t0);
double doubleWellPot(double a);


double clas(double a, double w, int pot);
double ddPot(double a, int pot);
double absol(double a);
 


__kernel void time_dev(__global double *f,
                       __global double *fh,
                       __global double *newf,
                       __global double *newfh,
                       __global double *omega,
                       __global double *rand1,
                       __global int *stable,
                       __global double *deltaTau,
                       __global int *lrgEl,
                       __global double *lrgVl,
                       __global int *sameness,
                       __global int *LIST_SIZE,
                       __global double *deltaT,
                       __global double *hPar,
                       __global int *potential,
                       __global double *C
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

    double dw = c*(double)sqrt((float)(2.*deltatau/deltat))*rand1[i];

    
    
    if(i==0){
        newf[i] = f[0]+m*deltatau*(f[1]+f[N-1]-2*f[0])/(double)pown((float)deltat,2)-ddPot(clas((double)i*deltat, om, potID)+f[0], potID) * f[0] *deltatau + dw;
        
        newfh[i] = fh[0]+m*deltatau*(fh[1]+fh[N-1]-2*fh[0])/(double)pown((float)deltat,2) -ddPot(clas((double)i*deltat, om, potID)+fh[0], potID)* fh[0] *deltatau + dw ;
        
        
    }
    else{
        if(i==N-1){
            newf[i] = f[N-1]+m*deltatau*(f[0]+f[N-2]-2*f[N-1])/(double)pown((float)deltat,2)+ddPot(clas((double)i*deltat, om, potID)+f[N-1], potID)* f[N-1] *deltatau + dw;
            
            newfh[i] = fh[N-1]+m*deltatau*(fh[0]+fh[N-2]-2*fh[N-1])/(double)pown((float)deltat,2) -ddPot(clas((double)i*deltat, om, potID)+fh[N-1], potID)* fh[N-1] *deltatau + dw;
            
        }
        else{
            newf[i] = f[i]+m*deltatau*(f[i+1]+f[i-1]-2*f[i])/(double)pown((float)deltat,2)-ddPot(clas((double)i*deltat, om, potID)+f[i], potID) * f[i] *deltatau + dw;
            
            newfh[i] = fh[i]+m*deltatau*(fh[i+1]+fh[i-1]-2*fh[i])/(double)pown((float)deltat,2) -ddPot(clas((double)i*deltat, om, potID)+fh[i], potID)* fh[i] *deltatau + dw;
        }
        
    }
    if(i==*lrgEl){
        if(absol(newf[i]-f[i]-dw)>*lrgVl){
            *stable=0;
        }
        else{
            *stable=1;
        }
    }
    
    if (isinf((float)newf[i])==1||isnan((float)newf[i])==1||isinf((float)newfh[i])==1||isnan((float)newfh[i])==1){
        *stable=-1;
    }
    if (newf[i]==newfh[i]){
        newfh[i]=newf[i]+h;
        *stable=-2;
    }
    if(newfh[i]-newf[i]==fh[i]-f[i]){
        *sameness*=1;
    }
    else{
        *sameness=0;
    }
    
    if(newf[i]>newf[*lrgEl]){
        *lrgEl=i;
    }
    if(absol(newf[i])>*lrgVl){
        *lrgVl=absol(newf[i]);
    }
    
    
//     newf[i]=0;
//     newfh[i]=h;
    
}








double harmoscSol(double a){
    double omega=1.;
    double mass =1.;
    return mass*(double)pown((float)omega,2)*a;
}
double harmoscPot(double a){
    double omega=1.;
    double mass =1.;
    return mass*(double)pown((float)omega,2);
}
double poeschlTeller(double a){
    double b=0.1;
    double V0=1.;
    return 2.*b*V0*(double)tanh((float)(b*a))/((double)pown(cosh((float)(b*a)),2));
    
}
double quartic(double a){
    double mu=1;
    return 4*mu*((double)pown((float)a,3));
    
}

double doubleWellSol(double t, double t0){
//     float x0=1.;
    
    return (double)tanh((t-t0)/(double)sqrt((float)2));
    
}
double doubleWellPot(double a){
//     float x0=1.;
    
    return 6*a*a-2;
    
}


double clas(double a, double w, int pot){
    
//     if(pot==0){
//         return dharmosc(a);
//         
//     }
//     if(pot==1){
//         return poeschlTeller(a);
//     }
//     if(pot==2){
//         return quartic(a);
//     }
    if(pot==3){
        
        return doubleWellSol(a, w);
    }
    else{
        return doubleWellSol(a, w);
    }

    
}
double ddPot(double a, int pot){
//     if(pot==0){
//         return dharmosc(a);
//         
//     }
//     if(pot==1){
//         return poeschlTeller(a);
//     }
//     if(pot==2){
//         return quartic(a);
//     }
    if(pot==3){
        
        return doubleWellPot(a);
    }
    else{
        return doubleWellPot(a);
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
 


