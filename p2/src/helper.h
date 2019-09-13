#pragma once

/*
    Helper Functions Header Files
*/

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>

#define GRID_H 8
#define GRID_W 8
#define GRID_D 8

int* threadCounter (int H, int W, int D = 0){
    /*
        Function to count the threads per iteration
    */
    int threadCount[3];
    threadCount[0] = (H + (GRID_H-1))/GRID_H;
    threadCount[1] = (W + (GRID_W-1))/GRID_W;
    threadCount[2] = (D + (GRID_D-1))/GRID_D;
    return threadCount;
}
/*-----------------------------------------------------------------------------
                            2D Helper Functions
-----------------------------------------------------------------------------*/
void initialize2DGrid (float temp[],int H, int W, float T){
    /*
        Function to initialize grid 
    */
    for (int i = 0; i < (W*H); i++)
    {
        temp[i] = T;
    }
}

void display2DGrid (float temp[], int H, int W){
    /*
        Function to Display 2D Grid
    */
    int idx = 0;
    for (int i = 0; i < H; i++){
        for (int j = 0; j < W; j++){
            idx = i*W + j;
            printf("%.2f \t", temp[idx]);
        }
        printf("\n");
    }   
}

void heatPlacement2D(std::vector <float> heater, float temp[], bool stencil[], int H, int W){
    /*
        Functions to place heat
    */
    int idx = 0;
    int startX, startY, endX, endY;
    int numEntries = heater.size()/5;
    for (int entry=0; entry < numEntries; entry++){
        startX = (int) heater[entry*5+0];
        startY = (int) heater[entry*5+1];
        endX = startX + (int) heater[entry*5+2];
        endY = startY + (int) heater[entry*5+3];
        for (int i = startY; i<endY; i++ ){
            for (int j = startX; j < endX; j++){
                idx = i*W + j;
                temp[idx] = heater[entry*5+4];
                stencil[idx] = true;
            }
        }
    } 
}

void record2DGrid(float temp[], int H, int W){
    std::ofstream grid2DLog;
    grid2DLog.open ("heatOutput.csv");
    int idx = 0;
    for (int i = 0; i < H; i++){
        for (int j = 0; j < (W-1); j++){
            idx = i*W + j;
            grid2DLog << temp[idx] << ", ";
        }
        grid2DLog << temp[idx+1] <<"\n";
    } 
    grid2DLog.close();
}

/*-----------------------------------------------------------------------------
                            3D Helper Functions
-----------------------------------------------------------------------------*/
void initialize3DGrid (float temp[],int H, int W, int D, float T){
    /*
        Function to initialize grid 
    */
    for (int i = 0; i < (W*H*D); i++)
    {
        temp[i] = T;
    }
}

void display3DGrid (float temp[], int H, int W, int D){
    /*
        Function to Display 2D Grid
    */
    int idx = 0;
    for (int k = 0; k < D; k++){
        for (int i = 0; i < H; i++){
            for (int j = 0; j < W; j++){
                idx = k*(H*W) + i*W + j;
                printf("%.2f \t", temp[idx]);
            }
            printf("\n");
        }
        printf("\n\n");  
    } 
}

void heatPlacement3D(std::vector <float> heater, float temp[], bool stencil[], int H, int W, int D){
    /*
        Functions to place heat
    */
    int idx = 0;
    int startX, startY, startZ, endX, endY, endZ;
    int numEntries = heater.size()/7;
    for (int entry=0; entry < numEntries; entry++){
        startX = (int) heater[entry*7+0];
        startY = (int) heater[entry*7+1];
        startZ = (int) heater[entry*7+2];
        endX = startX + (int) heater[entry*7+3];
        endY = startY + (int) heater[entry*7+4];
        endZ = startZ + (int) heater[entry*7+5];
        for (int k = startZ; k<endZ; k++){
            for (int i = startY; i<endY; i++ ){
                for (int j = startX; j < endX; j++){
                    idx = k*(H*W) + i*W + j;
                    temp[idx] = heater[entry*7+6];
                    stencil[idx] = true;
                }
            }
        }
    } 
}

void record3DGrid(float temp[], int H, int W, int D){
    std::ofstream grid3DLog;
    grid3DLog.open ("heatOutput.csv");
    int idx = 0;
    for (int k = 0; k < D; k++){
        for (int i = 0; i < H; i++){
            for (int j = 0; j < (W-1); j++){
                idx = k*(H*W) + i*W + j;
                grid3DLog << temp[idx] << ", ";
            }
            grid3DLog << temp[idx+1] <<"\n";
        } 
        grid3DLog<<"\n";
    }
    grid3DLog.close();
}