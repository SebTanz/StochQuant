#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE (0x100000)


double absol(double a){
    
    if (a<=0){
        return -a;
    }
    else{
        return a;
    }
}
int min(int m, int n){
    if (m<n){
        return m;
    }else{
        return n;
    }
    
}

 
int main(int argc, char **argv) {
    // Create the two input vectors
    
    const int LIST_SIZE     = atoi(argv[1]);
    const double deltat     = atof(argv[2]);
    const double deltatau   = atof(argv[3]);
    const double h          = atof(argv[4]);
    const int parisi        = atoi(argv[5]);
    const int frames        = atoi(argv[6]);
    const int potID         = atoi(argv[7]);
    const double C          = atof(argv[8]);
    const int dev           = atoi(argv[9]);
    const int fps           = atoi(argv[10]);
    const int inTime        = atoi(argv[11]);
    const char * startFile  = argv[12];
    const char * endFile    = argv[13];
    const int endAccuracy   = atoi(argv[14]);
//     printf("%d, %f, %f, %f\n", LIST_SIZE, deltat, deltatau, h);
    int recSimlgth;
    
    double *x       = (double*)malloc(sizeof(double)*LIST_SIZE);
    double *xh      = (double*)malloc(sizeof(double)*LIST_SIZE);
    double *newx    = (double*)malloc(sizeof(double)*LIST_SIZE);
    double *newxh   = (double*)malloc(sizeof(double)*LIST_SIZE);
    
    double *rand1   = (double*)malloc(sizeof(double)*LIST_SIZE);
    
    int stable = 1;
    
    double dtautmp = deltatau;
    
    int lrgEl = 0;
    double lrgVl = 0;
    
    int sameness=1;
    
    double v1;
    double v2;
    
    double xsum [LIST_SIZE];
    double xhsum [LIST_SIZE];
    
    int i;
    FILE *fp;
    char *source_str;
    size_t source_size;
    
    if(strcmp(startFile, "0")==0){
//         printf("no file !");
        for(i = 0; i < LIST_SIZE; i++) {
            x[i]=0;
            xh[i] = x[i]+h;
            newx[i] = x[i];
            newxh[i] = xh[i];
            xsum[i]=x[i];
            xhsum[i]=xh[i];
            
            v1 = (double)(rand() + 1. )/( (double)(RAND_MAX) + 1.);
            v2 = (double)(rand() + 1. )/( (double)(RAND_MAX) + 1.);
            rand1[i] = cos(2.*3.14*v2)*sqrt(-2.*log(v1));
            
            recSimlgth = 0;
        }
    }
    else{
//         printf("file open");
        fp = fopen(startFile, "r");
        if (!fp) {
            fprintf(stderr, "Failed to read Input.\n");
            exit(1);
        }
        char line[2];
        char * token;
        char * string = (char*)malloc(sizeof(line));
        char * tmp = (char*)malloc(sizeof(line));
        i = 0;
        int length = 0;
        int strsize = 0;
        while(fgets(line, sizeof(line), fp)){
//             printf("%s",line);
            
            if (line[0]=='\n'){
                char litstr[length];
                for(int n=0; n<length; n++)
                    litstr[n] = string[n];
                if(i==LIST_SIZE){
                    recSimlgth = atoi(litstr);
//                     printf("%d\n",recSimlgth);
                }
                if(i==LIST_SIZE+1){
                    
                    dtautmp = atof(litstr);
                    if(dtautmp>deltatau)
                        dtautmp=deltatau;
//                     printf("%d\n",recSimlgth);
                }
                if (i<LIST_SIZE){
                    token = strtok(litstr, "|");
                    xsum[i] = (double)atof(token);
                    token = strtok(NULL, "|");
                    xhsum[i] = (double)atof(token);
//                     printf("%f|%f\n",xsum[i],xhsum[i]);
                    x[i] = xsum[i]/(double)recSimlgth;
                    xh[i] = xhsum[i]/(double)recSimlgth;
                    newx[i] = x[i];
                    newxh[i] = xh[i];
                    
                    v1 = (double)(rand() + 1. )/( (double)(RAND_MAX) + 1.);
                    v2 = (double)(rand() + 1. )/( (double)(RAND_MAX) + 1.);
                    rand1[i] = cos(2.*3.14*v2)*sqrt(-2.*log(v1));
                    
                    length=0;
                }
                i++;
                
                
            }
            else{
                length++;
                if(length>strsize){
                    tmp = realloc(string, length*sizeof(line));
                    string = tmp;
                    strsize = length;
                }
                string[length-1] = line[0];
                
            }
            
            
        }
        free(string);
//         free(tmp);
        i=0;
        fclose( fp );
        
    }
    // Load the kernel source code into the array source_str
    
    fp = fopen("time_dev_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
 
    // Get platform and device information
    cl_platform_id *platform_id;
    cl_device_id *device_id;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    
    cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    platform_id = (cl_platform_id*)malloc(sizeof(cl_platform_id)*ret_num_platforms);
    ret = clGetPlatformIDs(ret_num_platforms, platform_id, NULL);
    
    cl_platform_id *platform0 = &platform_id[dev];
    
    ret = clGetDeviceIDs( *platform0, CL_DEVICE_TYPE_DEFAULT, 0, NULL, &ret_num_devices);
    device_id = (cl_device_id*) malloc(sizeof(cl_device_id) * ret_num_devices);
    ret = clGetDeviceIDs( *platform0, CL_DEVICE_TYPE_DEFAULT, ret_num_devices, device_id, NULL);
    
    //print device information
    size_t valueSize;
    char* value;
    cl_uint maxComputeUnits;
    size_t maxWorkGrSize;
    
    cl_device_id *device0 = &device_id[0];
    
    cl_int abc = clGetDeviceInfo(*device0, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    abc = clGetDeviceInfo(*device0, CL_DEVICE_NAME, valueSize, value, NULL);
//     printf("Device: %s\n", value);
    free(value);
    
    abc = clGetDeviceInfo(*device0, CL_DEVICE_VERSION, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    abc = clGetDeviceInfo(*device0, CL_DEVICE_VERSION, valueSize, value, NULL);
//     printf("Hardware Version: %s\n", value);
    free(value);
    
    abc = clGetDeviceInfo(*device0, CL_DRIVER_VERSION, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    abc = clGetDeviceInfo(*device0, CL_DRIVER_VERSION, valueSize, value, NULL);
//     printf("Software Version: %s\n", value);
    free(value);
    
    abc = clGetDeviceInfo(*device0, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    abc = clGetDeviceInfo(*device0, CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
//     printf("OpenCL C Version: %s\n", value);
    free(value);
   
    abc = clGetDeviceInfo(*device0, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, NULL);
//     printf("Parallel compute units: %d\n", maxComputeUnits);
    
    abc = clGetDeviceInfo(*device0, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGrSize, NULL);
//     printf("Max work group size: %d\n", maxWorkGrSize);
    
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, device0, NULL, NULL, &ret);
 
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, *device0, 0, &ret);
 
    // Create memory buffers on the device for each vector 
    cl_mem x_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            LIST_SIZE * sizeof(double), NULL, &ret);
    
    cl_mem xh_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            LIST_SIZE * sizeof(double), NULL, &ret);
    
    cl_mem nx_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            LIST_SIZE * sizeof(double), NULL, &ret);
    
    cl_mem nxh_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            LIST_SIZE * sizeof(double), NULL, &ret);
    
    
    cl_mem r1_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            LIST_SIZE * sizeof(double), NULL, &ret);
    
    
    cl_mem st_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            sizeof(int), NULL, &ret);
    
    cl_mem dt_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            sizeof(double), NULL, &ret);
    
    
    cl_mem lE_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            sizeof(int), NULL, &ret);
    
    cl_mem lV_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            sizeof(double), NULL, &ret);
    
    
    cl_mem sm_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            sizeof(int), NULL, &ret);
    
    
    cl_mem cN_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            sizeof(int), NULL, &ret);
    
    cl_mem cdt_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            sizeof(double), NULL, &ret);
    
    cl_mem ch_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            sizeof(double), NULL, &ret);
    
    cl_mem cPi_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            sizeof(int), NULL, &ret);
    
    cl_mem cC_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            sizeof(double), NULL, &ret);
    
    
 
    // Copy the lists x and xh to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, x_mem_obj, CL_TRUE, 0,
                               LIST_SIZE * sizeof(double), x, 0, NULL, NULL);
    
    ret = clEnqueueWriteBuffer(command_queue, xh_mem_obj, CL_TRUE, 0,
                               LIST_SIZE * sizeof(double), xh, 0, NULL, NULL);
    
    ret = clEnqueueWriteBuffer(command_queue, nx_mem_obj, CL_TRUE, 0,
                               LIST_SIZE * sizeof(double), newx, 0, NULL, NULL);
    
    ret = clEnqueueWriteBuffer(command_queue, nxh_mem_obj, CL_TRUE, 0, 
                               LIST_SIZE * sizeof(double), newxh, 0, NULL, NULL);
    
    
    ret = clEnqueueWriteBuffer(command_queue, r1_mem_obj, CL_TRUE, 0, 
                               LIST_SIZE * sizeof(double), rand1, 0, NULL, NULL);
    
    ret = clEnqueueWriteBuffer(command_queue, st_mem_obj, CL_TRUE, 0, 
                               sizeof(int), &stable, 0, NULL, NULL);
    
    ret = clEnqueueWriteBuffer(command_queue, dt_mem_obj, CL_TRUE, 0, 
                               sizeof(double), &dtautmp, 0, NULL, NULL);
    
    
    ret = clEnqueueWriteBuffer(command_queue, lE_mem_obj, CL_TRUE, 0, 
                               sizeof(int), &lrgEl, 0, NULL, NULL);
    
    ret = clEnqueueWriteBuffer(command_queue, lV_mem_obj, CL_TRUE, 0, 
                               sizeof(double), &lrgVl, 0, NULL, NULL);
    
    
    ret = clEnqueueWriteBuffer(command_queue, sm_mem_obj, CL_TRUE, 0, 
                               sizeof(int), &sameness, 0, NULL, NULL);
    
    
    ret = clEnqueueWriteBuffer(command_queue, cN_mem_obj, CL_TRUE, 0, 
                               sizeof(int), &LIST_SIZE, 0, NULL, NULL);
    
    ret = clEnqueueWriteBuffer(command_queue, cdt_mem_obj, CL_TRUE, 0, 
                               sizeof(double), &deltat, 0, NULL, NULL);
    
    ret = clEnqueueWriteBuffer(command_queue, ch_mem_obj, CL_TRUE, 0, 
                               sizeof(double), &h, 0, NULL, NULL);
    
    ret = clEnqueueWriteBuffer(command_queue, cPi_mem_obj, CL_TRUE, 0, 
                               sizeof(int), &potID, 0, NULL, NULL);
    
    ret = clEnqueueWriteBuffer(command_queue, cC_mem_obj, CL_TRUE, 0, 
                               sizeof(double), &C, 0, NULL, NULL);
    
 
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &ret);
    
    
    // Build the program
    ret = clBuildProgram(program, 1, device0, NULL, NULL, NULL);
    
    if (ret == CL_BUILD_PROGRAM_FAILURE) {
    // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, *device0, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, *device0, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
    }
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "time_dev", &ret);
    /*
    if(ret==CL_INVALID_PROGRAM)printf("Fail: CL_INVALID_PROGRAM\n");
    else printf("Success: clCreateKernel\n");
    if(ret==CL_INVALID_PROGRAM_EXECUTABLE)printf("Fail: CL_INVALID_PROGRAM_EXECUTABLE\n");
    else printf("Success: clCreateKernel\n");
    if(ret==CL_INVALID_KERNEL_NAME)printf("Fail: CL_INVALID_KERNEL_NAME\n");
    else printf("Success: clCreateKernel\n");
    if(ret==CL_INVALID_KERNEL_DEFINITION)printf("Fail: CL_INVALID_VALUE\n");
    else printf("Success: clCreateKernel\n");
    if(ret==CL_OUT_OF_HOST_MEMORY)printf("Fail: CL_OUT_OF_HOST_MEMORY\n");
    else printf("Success: clCreateKernel\n");
    */
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&x_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&xh_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&nx_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&nxh_mem_obj);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&r1_mem_obj);
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&st_mem_obj);
    ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&dt_mem_obj);
    ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&lE_mem_obj);
    ret = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&lV_mem_obj);
    ret = clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&sm_mem_obj);
    ret = clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&cN_mem_obj);
    ret = clSetKernelArg(kernel, 11, sizeof(cl_mem), (void *)&cdt_mem_obj);
    ret = clSetKernelArg(kernel, 12, sizeof(cl_mem), (void *)&ch_mem_obj);
    ret = clSetKernelArg(kernel, 13, sizeof(cl_mem), (void *)&cPi_mem_obj);
    ret = clSetKernelArg(kernel, 14, sizeof(cl_mem), (void *)&cC_mem_obj);
 
    // Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE; // Process the entire lists
    
    // Divide work items into groups of 64
//     size_t local_item_size = 1;

    size_t local_item_size = maxWorkGrSize;
    
    if (maxWorkGrSize<global_item_size){
        while(global_item_size%local_item_size!=0){
            local_item_size--;
        }
        
    }
    else{
        local_item_size=global_item_size;
    }
    
    
//     printf("%d, %d",(int)maxWorkGrSize, (int)local_item_size);
        
    
    
    

//     size_t local_item_size = LIST_SIZE;
    double expec ;
    double expec2;
    double tmp;
    
    
    int stabCnt=0;
    
    
    
    double samRd=1;
    int outp=0;
    double aver1;
    double aver2;
    
    int runs = 0;
    for(int j=0; j<frames; j++){
        
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
                                        &global_item_size, &local_item_size, 0, NULL, NULL);
        
        for(i=0; i<LIST_SIZE; i++){
//             printf("% -.10f ", x[i]);
            v1 = (double)(rand() + 1. )/( (double)(RAND_MAX) + 1.);
            v2 = (double)(rand() + 1. )/( (double)(RAND_MAX) + 1.);
            rand1[i] = cos(2.*3.14*v2)*sqrt(-2.*log(v1));
            
            if(i!=0&&(j)%fps==0){
                
                if (parisi==1){
                    aver1=(xhsum[i]-xsum[i])/h/(double)(runs+1+recSimlgth);
                    aver2=(xhsum[i-1]-xsum[i-1])/h/(double)(runs+1+recSimlgth);
                    printf(" % -.20f |", (log(aver1)-log(aver2))/deltat);
                }
                else{
                    aver1=(xsum[i]*xsum[0])/(double)(runs+1+recSimlgth);
                    aver2=(xsum[i-1]*xsum[0])/(double)(runs+1+recSimlgth);
                    printf(" % -.20f |", (log(aver1)-log(aver2))/deltat);
                    
                }
                if(i==LIST_SIZE-1){
                    printf("% -.2f\n", 100.*((double)j+1)/(double)frames);
                }
            }
            
        }
        
        // Read the memory buffer x, xh on the device to the local variable x, xh
        
        ret = clEnqueueReadBuffer(command_queue, st_mem_obj, CL_TRUE, 0, 
                                  sizeof(int), &stable, 0, NULL, NULL);
        if(stable==1){
//             printf("stable %d", stable);
            ret = clEnqueueReadBuffer(command_queue, nx_mem_obj, CL_TRUE, 0, 
                                        LIST_SIZE * sizeof(double), x, 0, NULL, NULL);
            ret = clEnqueueReadBuffer(command_queue, nxh_mem_obj, CL_TRUE, 0, 
                                        LIST_SIZE * sizeof(double), xh, 0, NULL,NULL);
            if(j>=inTime){
                runs+=1;
                for(i=0; i<LIST_SIZE; i++){
                    xsum[i]+=x[i];
                    xhsum[i]+=xh[i];
                }
            }
            
//             dtautmp=deltatau;
//             ret = clEnqueueWriteBuffer(command_queue, dt_mem_obj, CL_TRUE,   0,sizeof(double), &dtautmp, 0, NULL, NULL);
            
            
        }
        else{
//             printf("unstable %d\n", stable);
            ret = clEnqueueReadBuffer(command_queue, dt_mem_obj, CL_TRUE, 0, 
                                        sizeof(double), &dtautmp, 0, NULL, NULL);
            
            dtautmp*=0.950;
            ret = clEnqueueWriteBuffer(command_queue, dt_mem_obj, CL_TRUE, 0, 
                               sizeof(double), &dtautmp, 0, NULL, NULL);
            stable=1;
            ret = clEnqueueWriteBuffer(command_queue, st_mem_obj, CL_TRUE, 0, 
                               sizeof(int), &stable, 0, NULL, NULL);
        }
        ret = clEnqueueReadBuffer(command_queue, sm_mem_obj, CL_TRUE, 0, 
                                  sizeof(int), &sameness, 0, NULL, NULL);
        
        if (sameness==1){
//             printf("sameness %d", sameness);
            
            for(i=0; i<LIST_SIZE; i++){
//                 samRd = h*((double)(rand() + 1. )/( (double)(RAND_MAX) + 1.)+0.5);
                x[i]*=samRd;
                xh[i]*=samRd;
            }
            
            
        }
        else{
            sameness=1;
            ret = clEnqueueWriteBuffer(command_queue, sm_mem_obj, CL_TRUE, 0, 
                                           sizeof(int), &sameness, 0, NULL, NULL);
        }
        
        
        ret = clEnqueueWriteBuffer(command_queue, x_mem_obj, CL_TRUE, 0,
                                   LIST_SIZE * sizeof(double), x, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, xh_mem_obj, CL_TRUE, 0, 
                                   LIST_SIZE * sizeof(double), xh, 0, NULL, NULL);
        
        
        ret = clEnqueueWriteBuffer(command_queue, r1_mem_obj, CL_TRUE, 0, 
            LIST_SIZE * sizeof(double), rand1, 0, NULL, NULL);
        
        fflush(stdout);
        
    }
//     printf("%d\n",recSimlgth);
    
    if(strcmp(endFile, "0")!=0){
        fp = fopen(endFile, "w");
        if (!fp) {
            fprintf(stderr, "Failed to write to Output.\n");
            exit(1);
        }
        
        for (i=0; i<LIST_SIZE; i++){
//             printf("%f|%f\n",xsum[i],xhsum[i]);
            fprintf(fp, "% -*a|% -*a\n",endAccuracy,xsum[i], endAccuracy, xhsum[i]);
            
        }
        fprintf(fp, "%*d|N\n", endAccuracy, runs+1+recSimlgth);
        fprintf(fp, "% -*e|deltaTau\n", endAccuracy, dtautmp);
        fclose( fp );
        
    }
    

    
    
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(x_mem_obj);
    ret = clReleaseMemObject(xh_mem_obj);
    ret = clReleaseMemObject(r1_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(x);
    free(xh);
    free(newx);
    free(newxh);
    free(rand1);
    free(device_id);
    free(platform_id);
    
    return 0;
}
