#pragma OPENCL EXTENSION cl_khr_fp64 : enable



double dharmosc(double a){
    double omega=1.;
    double mass =1.;
    return mass*(double)pown((float)omega,2)*a;
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
double linearPot(double a){
    if(a<=0){
        return -1.;
    }else{
        return 1;
    }
    
}
double doubleWell(double a){
    float x0=1.;
    return 1/2.* a* (-1. + (double)pown((float)a,2.)/(double)pown(x0,2.));
    
}
double expSq(double a){
    return 2*a*(double)exp((float)(a*a));
    
}
/*
double oddifier(double (*f)(), double a){
    return f(a);
    
}*/

double func(double a, int pot){
    
    if(pot==0){
        return dharmosc(a);
        
    }
    if(pot==1){
        return poeschlTeller(a);
    }
    if(pot==2){
        return quartic(a);
    }
    if(pot==3){
        
        return doubleWell(a);
    }
    else{
        return 0.;
    }
//     return dharmosc(a);
    
//     return poeschlTeller(a);
//     return expSq(a);
//     return atan(a);
//     return sinh(a);
//     return sin(a);
//     return linearPot(a);
    
}
double absol(double a){
    
    if (a<=0){
        return -a;
    }
    else{
        return a;
    }
}
 


__kernel void time_dev(__global double *x,
                       __global double *xh,
                       __global double *newx,
                       __global double *newxh,
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
    double m = 1.;
    double h = *hPar;
    int potID = *potential;
    double c = *C;
//     double c=1.;
//     double c=31.62277660168379332e-3;
//     double newx;
//     double newxh;
//     return a normally distributed random value
    // Do the operation
    
//     double dw = c*sqrt(2.*deltatau/deltat)*cos(2.*3.14*v2)*sqrt(-2.*log(v1));
    double dw = c*(double)sqrt((float)(2.*deltatau/deltat))*rand1[i];
//     newx[i] = 0.;
//     dw = 0;
    
    
    if(i==0){
        
        newx[i] = x[0]+m*deltatau*(x[1]+x[N-1]-2*x[0])/(double)pown((float)deltat,2)- func(x[0], potID)*deltatau + dw;
        
        newxh[i] = xh[0]+m*deltatau*(xh[1]+xh[N-1]-2*xh[0])/(double)pown((float)deltat,2)-func(xh[0], potID)*deltatau + dw + deltatau*h;
        
        
    }
    else{
        if(i==N-1){
            newx[i] = x[N-1]+m*deltatau*(x[0]+x[N-2]-2*x[N-1])/(double)pown((float)deltat,2)- func(x[N-1], potID)*deltatau + dw;
            newxh[i] = xh[N-1]+m*deltatau*(xh[0]+xh[N-2]-2*xh[N-1])/(double)pown((float)deltat,2)-func(xh[N-1], potID)*deltatau + dw;
            
        }
        else{
            newx[i] = x[i]+m*deltatau*(x[i+1]+x[i-1]-2*x[i])/(double)pown((float)deltat,2)-func(x[i], potID)*deltatau + dw;
            
            newxh[i] = xh[i]+m*deltatau*(xh[i+1]+xh[i-1]-2*xh[i])/(double)pown((float)deltat,2)-func(xh[i], potID)*deltatau + dw;
        }
        
    }
    if(i==*lrgEl){
        if(absol(newx[i]-x[i]-dw)>*lrgVl){
            *stable=0;
        }
        else{
            *stable=1;
        }
    }
    
    if (isinf((float)newx[i])==1||isnan((float)newx[i])==1||isinf((float)newxh[i])==1||isnan((float)newxh[i])==1){
        *stable=-1;
    }
    if (newx[i]==newxh[i]){
        newxh[i]=newx[i]+h;
        *stable=-2;
    }
    if(newxh[i]-newx[i]==xh[i]-x[i]){
        *sameness*=1;
    }
    else{
        *sameness=0;
    }
    
    if(newx[i]>newx[*lrgEl]){
        *lrgEl=i;
    }
    if(absol(newx[i])>*lrgVl){
        *lrgVl=absol(newx[i]);
    }
//     newx[i]=1;
//     newxh[i]=2;
//     x[i]=newx[i];
//     xh[i]=newxh[i];
    
}

