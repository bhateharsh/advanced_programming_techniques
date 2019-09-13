#pragma once

// Defining my Kernel
__global__ void grid3D(float *tOld, float *tCurr, float *k, int *H, int *W, int *D, bool *heatElement){
    int h = *H;
    int w = *W;
    int d = *D;
    float K = *k;
    float t1, t2, t3, t4, t5, t6;
    // Indexing
    const int i = blockIdx.x * blockDim.x + threadIdx.x;    // Height
    const int j = blockIdx.y * blockDim.y + threadIdx.y;    // Weight 
    const int l = blockIdx.z * blockDim.z + threadIdx.z;    // Depth
    if ((i >= h) || (j >= w) || (l >= d)){
        return;
    }
    // Width
    if (j-1 < 0){
        t4 = 0;
    }
    else{
        t4 = tOld[l*h*w + i*w + (j-1)] - tOld[l*h*w + i*w + j];
    }
    if (j+1 >= w){
        t2 = 0;
    }
    else{
        t2 = tOld[l*h*w + i*w + (j+1)] - tOld[l*h*w + i*w + j];
    }
    // Height
    if (i-1 < 0){
        t1 = 0;
    }
    else{
        t1 = tOld[l*h*w + (i-1)*w + (j)] - tOld[l*h*w + i*w + j];
    }
    if (i+1 >= h){
        t3 = 0;
    }
    else{
        t3 = tOld[l*h*w + (i+1)*w + (j)] - tOld[l*h*w + i*w + j];
    }
    // Depth
    if (l-1 < 0){
        t5 = 0;
    }
    else{
        t5 = tOld[(l-1)*h*w + i*w + j] - tOld[l*h*w + i*w + j];
    }
    if (l+1 >= d){
        t6 = 0;
    }
    else{
        t6 = tOld[(l+1)*h*w + i*w + j] - tOld[l*h*w + i*w + j];
    }
    
    // Assignment
    if (heatElement[l*h*w + i*w + j]){
        tCurr[l*h*w + i*w + j] = tOld[l*h*w + i*w + j];
    }
    else{
        tCurr[l*h*w + i*w + j] = tOld[l*h*w + i*w + j] + K*(t1 + t2 + t3 + t4 + t5 + t6);
    }
}