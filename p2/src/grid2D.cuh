#pragma once

// Defining my Kernel
__global__ void grid2D(float *tOld, float *tCurr, float *k, int *H, int *W, bool *heatElement){
    int h = *H;
    int w = *W;
    float K = *k;
    float t1, t2, t3, t4;
    s// Indexing
    const int i = blockIdx.x * blockDim.x + threadIdx.x;    // Height
    const int j = blockIdx.y * blockDim.y + threadIdx.y;    //Weight 
    
    if ((i >= h) || (j >= w)){
        return;
    }
    
    // Width
    if (j-1 < 0){
        t4 = 0;
    }
    else{
        t4 = tOld[i*w + (j-1)] - tOld[i*w + j];
    }
    if (j+1 >= w){
        t2 = 0;
    }
    else{
        t2 = tOld[i*w + (j+1)] - tOld[i*w + j];
    }
    // Height
    if (i-1 < 0){
        t1 = 0;
    }
    else{
        t1 = tOld[(i-1)*w + (j)] - tOld[i*w + j];
    }
    if (i+1 >= h){
        t3 = 0;
    }
    else{
        t3 = tOld[(i+1)*w + (j)] - tOld[i*w + j];
    }
    // Assignment
    if (heatElement[i*w + j]){
        tCurr[i*w + j] = tOld[i*w + j];
    }
    else{
        tCurr[i*w + j] = tOld[i*w + j] + K*(t1 + t2 + t3 + t4);
    }
}