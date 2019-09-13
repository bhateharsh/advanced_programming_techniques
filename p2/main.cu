/*
    2D HEAT TRANSFER FUNCTION
    Script to achieve 2D heat transfer on one timestep
*/

#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include "src/helper.h"
#include "src/grid2D.cuh"
#include "src/grid3D.cuh"
#include "src/configParser.h"

#define GRID_H 8
#define GRID_W 8
#define GRID_D 8

int main(int argc, char* argv[]){
    if (argc != 2){
        printf("Please fix the path input to the Code!\n");
        return 0;
    }
    std::string configPath(argv[1]);
    Config conf(configPath);
    conf.parseConfig();
    conf.generateReport();
    // Defining Some Commons
    float temp [conf.gridHeight*conf.gridWidth*conf.gridDepth];
    bool stencil [conf.gridHeight*conf.gridWidth*conf.gridDepth] = {false};
    // Computing Kernel Parameters
        // Case 2D 
    int *threadCount2D = threadCounter(conf.gridHeight,
                                    conf.gridWidth);
    dim3 threadsPerBlock2D (GRID_H, GRID_W);
    dim3 numBlocks2D(threadCount2D[0], threadCount2D[1]);
    if (!conf.flag3D){
        printf("----------THREAD AND BLOCK REPORT----------\n");
        printf("BLOCKS: x = %d, y = %d \n", numBlocks2D.x, numBlocks2D.y);
        printf("-------------------------------------------\n");
    }
        // Case 3D
    int *threadCount3D = threadCounter(conf.gridHeight,
                                    conf.gridWidth, 
                                    conf.gridDepth);
    dim3 threadsPerBlock3D (GRID_H, GRID_W, GRID_D);
    dim3 numBlocks3D(threadCount3D[0], 
                    threadCount3D[1],
                    threadCount3D[2]);
    if (conf.flag3D){
        printf("----------THREAD AND BLOCK REPORT----------\n");
        printf("BLOCKS: x = %d, y = %d, z = %d\n", numBlocks3D.x, numBlocks3D.y, numBlocks3D.z);
        printf("-------------------------------------------\n");
    }
    // Size Variables
    size_t size_grid;
    size_t size_stencil;
    // Case Specific Stuff
    if (conf.flag3D){
        // Initializing the Grid and Stencil
        initialize3DGrid (temp,
            conf.gridHeight,
            conf.gridWidth,
            conf.gridDepth,
            conf.initTemp);
        heatPlacement3D(conf.heatingSource,
            temp,
            stencil,
            conf.gridHeight,
            conf.gridWidth, 
            conf.gridDepth);
        // Displaying the grid, comment to not see the output
        // display3DGrid(temp,
        //     conf.gridHeight,
        //     conf.gridWidth, 
        //     conf.gridDepth);
        // Initializing the Kernel Parameters
        // Defining Device Specific Memory Size
        size_grid = sizeof(float)*conf.gridHeight*conf.gridWidth*conf.gridDepth;
        size_stencil = sizeof(bool)*conf.gridHeight*conf.gridWidth*conf.gridDepth;
    }
    else{
        // Initializing the Grid and Stencil
        initialize2DGrid (temp,
            conf.gridHeight,
            conf.gridWidth,
            conf.initTemp);
        heatPlacement2D(conf.heatingSource,
            temp,
            stencil,
            conf.gridHeight,
            conf.gridWidth);
        // Displaying the grid, comment to not see the output
        // m//     conf.gridWidth);
        // Initializing the Kernel Parameters
        // Defining Device Specific Memory Size
        size_grid = sizeof(float)*conf.gridHeight*conf.gridWidth;
        size_stencil = sizeof(bool)*conf.gridHeight*conf.gridWidth; 
    }
    
    // Initializing Device Variables
    int *dev_H, *dev_W, *dev_D;
    float *dev_k;
    float *dev_tempCurr, *dev_tempOld;
    bool *dev_heatElem;

    cudaMalloc ((void **)&dev_tempCurr, size_grid);
    cudaMalloc ((void **)&dev_tempOld,  size_grid);
    cudaMalloc ((void **)&dev_k,        sizeof(float));
    cudaMalloc ((void **)&dev_H,        sizeof(int));
    cudaMalloc ((void **)&dev_W,        sizeof(int));
    cudaMalloc ((void **)&dev_D,        sizeof(int));
    cudaMalloc ((void **)&dev_heatElem, size_stencil);

    cudaMemcpy (dev_tempCurr, 
                &temp, 
                size_grid, 
                cudaMemcpyHostToDevice);
    cudaMemcpy (dev_tempOld, 
                &temp, 
                size_grid, 
                cudaMemcpyHostToDevice);
    cudaMemcpy (dev_k, 
                &conf.k, 
                sizeof (float), 
                cudaMemcpyHostToDevice);
    cudaMemcpy (dev_W, 
                &conf.gridWidth, 
                sizeof (int), 
                cudaMemcpyHostToDevice);
    cudaMemcpy (dev_H, 
                &conf.gridHeight, 
                sizeof (int), 
                cudaMemcpyHostToDevice);
    cudaMemcpy (dev_D, 
                &conf.gridDepth, 
                sizeof (int), 
                cudaMemcpyHostToDevice);
    cudaMemcpy (dev_heatElem, 
                &stencil, 
                size_stencil, 
                cudaMemcpyHostToDevice);

    // Running the time steps
    for (int i = 0; i < conf.timeStep; i ++){
        if (conf.flag3D){
            if (i%2 == 0){
                grid3D<<<numBlocks3D, threadsPerBlock3D>>> (dev_tempOld, 
                    dev_tempCurr, 
                    dev_k, 
                    dev_H, 
                    dev_W,
                    dev_D, 
                    dev_heatElem);
            }
            else{
                grid3D<<<numBlocks3D, threadsPerBlock3D>>> (dev_tempCurr, 
                    dev_tempOld, 
                    dev_k, 
                    dev_H, 
                    dev_W,
                    dev_D, 
                    dev_heatElem);
            }
        }
        else{
            if (i%2 == 0){
                grid2D<<<numBlocks2D, threadsPerBlock2D>>> (dev_tempOld, 
                                                    dev_tempCurr, 
                                                    dev_k, 
                                                    dev_H, 
                                                    dev_W, 
                                                    dev_heatElem);
            }
            else{
                grid2D<<<numBlocks2D, threadsPerBlock2D>>> (dev_tempCurr, 
                    dev_tempOld, 
                    dev_k, 
                    dev_H, 
                    dev_W, 
                    dev_heatElem);
            }
        }
        cudaDeviceSynchronize();
    }
    printf("Run Complete!\n");
    if (conf.timeStep%2 == 0){
        cudaMemcpy (temp, 
                dev_tempOld, 
                size_grid, 
                cudaMemcpyDeviceToHost);
    }
    else{
        cudaMemcpy (temp, 
            dev_tempCurr, 
            size_grid, 
            cudaMemcpyDeviceToHost);
    }
    if (conf.flag3D){
        // display3DGrid(temp, conf.gridHeight, conf.gridWidth, conf.gridDepth);
        record3DGrid (temp, conf.gridHeight, conf.gridWidth, conf.gridDepth);
    }
    else{
        // display2DGrid(temp, conf.gridHeight, conf.gridWidth);
        record2DGrid (temp, conf.gridHeight, conf.gridWidth);
    }
    // Freeing the CUDA Memory Allocation
    cudaFree(dev_tempCurr);
    cudaFree(dev_tempOld);
    cudaFree(dev_k);
    cudaFree(dev_H);
    cudaFree(dev_W);
    cudaFree(dev_D);
    cudaFree (dev_heatElem);
    return 0;
}