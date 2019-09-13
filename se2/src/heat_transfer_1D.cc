#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include "mpi.h"

#define ROOT  0

int main(int argc, char**argv)
{
    if (argc != 5){
        return 0;
    }

    // Equation Constants
    const double alpha = 1;
    const double K = 1;
    const double h = 2;
    const double r = K/(h*h);
    
    // Inputing the Initial Conditions
    const double tempT1 = std::atof(argv[1]);
    const double tempT2 = std::atof(argv[2]);
    const int gridPts = std::atoi(argv[3]);
    const int timeSteps = std::atoi(argv[4]);
    std::ofstream tempLog;

    // Outputting the Inputs
    // printf("T1 = %f deg \n",tempT1);
    // printf("T2 = %f deg \n",tempT2);
    // printf("Grid Points = %d points \n",gridPts);
    // printf("Time = %d steps \n",timeSteps);
    // MPI Initialization
    int runCode, rank, procs, tag, usedProcs;
    MPI_Status Stat;
    
    //Initializing MPI
    runCode = MPI_Init(&argc,&argv);
    if (runCode != MPI_SUCCESS) {
        std::cout<<"MPI Failed\n";
        MPI_Finalize();
        exit(1);
    }

    // Setting up Task size and Rank
    MPI_Comm_size(MPI_COMM_WORLD,&procs);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    // Resource Distribution
    int chunkSize = (gridPts+2)/procs;
    int rem = (gridPts+2)%procs;
    if (chunkSize < 2){
        chunkSize = 2;
        usedProcs = (gridPts+2)/chunkSize;
        rem = (gridPts+2)%usedProcs;
    }
    else{
        usedProcs = procs;
    }
    int dest, source, size;
    
    // Initializing Array
    int maxSize = chunkSize + rem + 2; 
    int recvCounts [procs];
    int displacement [procs];
    for (int i = 0; i < procs; i ++){
        if (i == 0){
            recvCounts[i] = chunkSize + rem-1;
            displacement[i] = 0;
        }
        else if (i == procs - 1){
            recvCounts[i] = chunkSize - 1;
            displacement[i] = displacement[i -1] + recvCounts[i-1];
        }
        else{
            recvCounts[i] = chunkSize;
            displacement[i] = displacement[i -1] + recvCounts[i-1];
        }
        // std::cout<<"recvCounts["<<i<<"] = "<<recvCounts[i]<<std::endl;
    }
    double currGrid[maxSize];
    double nextGrid[maxSize];
    double finalGrid[procs*maxSize];
    for (int i = 0; i < maxSize; i++){
        currGrid[i] = 0;
        nextGrid[i] = 0;
    }
    // Assigning Buffer Size, Index    
    if (rank == 0){
        size = chunkSize + rem + 1;
        currGrid[0] = tempT1;
        nextGrid[0] = tempT1;
    }
    else if (rank == usedProcs-1){
        size = chunkSize + 1;
        currGrid[size-1] = tempT2;
        nextGrid[size-1] = tempT2;
    }
    else{
        size = chunkSize + 2;
    }

    for (int time = 1; time <= timeSteps; time++){
        // Assorted Send and Recieve
        if (rank%2 == 0 && rank < usedProcs){
            dest = rank + 1;
            if (dest < usedProcs){
                // std::cout<<"Attempting Rank Even Phase 1"<<std::endl;
                // std::cout<<"rank = "<<rank<<std::endl;
                runCode = MPI_Send(&currGrid[size-2], 1, MPI_DOUBLE, dest, 123, MPI_COMM_WORLD);
                runCode = MPI_Recv(&currGrid[size-1], 1, MPI_DOUBLE, dest, 123, MPI_COMM_WORLD,&Stat);
                // std::cout<<"Rank Even Phase 1 Complete"<<std::endl;
            }
            source = rank - 1;
            if (source >= 0 && rank < usedProcs){
                // std::cout<<"Attempting Rank Even Phase 2"<<std::endl;
                // std::cout<<"rank = "<<rank<<std::endl;
                runCode = MPI_Recv(&currGrid[0], 1, MPI_DOUBLE, source, 123, MPI_COMM_WORLD,&Stat);  
                runCode = MPI_Send(&currGrid[1], 1, MPI_DOUBLE, source, 123, MPI_COMM_WORLD);  
                // std::cout<<"Rank Even Phase 2 Complete"<<std::endl;
            }
            // std::cout<<"Ranks in EVEN = "<<rank<<std::endl;
        }
        else if (rank%2 != 0 && rank < usedProcs){
            source = rank-1;
            if (source >= 0 && source < usedProcs){
                // std::cout<<"Attempting Rank Odd Phase 1"<<std::endl;
                // std::cout<<"rank = "<<rank<<std::endl;
                runCode = MPI_Recv(&currGrid[0], 1, MPI_DOUBLE, rank-1, 123, MPI_COMM_WORLD,&Stat);     
                runCode = MPI_Send(&currGrid[1], 1, MPI_DOUBLE, rank-1, 123, MPI_COMM_WORLD);
                // std::cout<<"Rank Odd Phase 1 Complete"<<std::endl;
            }
            dest = rank + 1;
            if (dest < usedProcs)
            {
                // std::cout<<"Attempting Rank Odd Phase 2"<<std::endl;
                // std::cout<<"rank = "<<rank<<std::endl;
                runCode = MPI_Send(&currGrid[size-2], 1, MPI_DOUBLE, dest, 123, MPI_COMM_WORLD);
                runCode = MPI_Recv(&currGrid[size-1], 1, MPI_DOUBLE, dest, 123, MPI_COMM_WORLD,&Stat);
                // std::cout<<"Rank Odd Phase 2 Complete"<<std::endl;
            }
            // std::cout<<"Rank in Odd = "<<rank<<std::endl;
        }
        else{
            continue;
        }
        // std::cout<<"Send Recieve Cycle Complete"<<std::endl;
        
        // Diffusion Stuff
        for (int i = 1; i < size-1; i ++){
            nextGrid[i] = (1-2*r)*currGrid[i] + r*currGrid[i-1] + r*currGrid[i+1];
        }
        for (int i = 0; i < size; i ++){
            currGrid[i] = nextGrid[i];
        }
    }
    // tempLog.open("temp_log.csv", std::ios_base::app);
    MPI_Gatherv(&currGrid[1], recvCounts[rank], MPI_DOUBLE, &finalGrid[0], recvCounts, displacement,  MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // for (int i = 1; i < size-1; i ++){
        // std::cout<<"Rank = "<<rank<<" currGrid["<<i<<" ]"<<currGrid[i]<<std::endl;
    // }
    if (rank == 0){
        // std::cout<<"Gatherstuff"<<std::endl;
        tempLog.open("heat1Doutput.csv");

        for (int i = 0; i < gridPts-1; i ++){
            tempLog << finalGrid[i] << ", ";
            std::cout<<"finalGrid[ "<<i<<"] ="<<finalGrid[i]<<std::endl;
        }
        tempLog<<finalGrid[gridPts-1];
        std::cout<<"finalGrid ["<<gridPts-1<<"] ="<<finalGrid[gridPts-1]<<std::endl;
        tempLog.close();
    }
    // tempLog.close();
    MPI_Finalize();
}