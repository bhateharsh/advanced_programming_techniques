/*
    Config File Reader
*/

#include "configParser.h"
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>

// Constructor
Config::Config(const std::string& filePath){
    confPath = filePath;
    printf ("Input File: %s \n", confPath.c_str());
}
// Destructor
Config::~Config(){
}

void Config::parseConfig(){
    /*
        Function to parse the config file
    */
    std::ifstream configFile;
    configFile.open(confPath.c_str());
    std::string line;
    printf ("Parsing the config file.\n");
    if (!configFile){
        printf("Unable to Open the Config file! \n");
        exit(1);
    }
    int count = 0;
    while (!configFile.eof())
    {
        std::getline(configFile, line);
        line.erase(std::remove( line.begin(), line.end(), ' ' ), line.end() );
        if (*line.begin() == '#'){
            line.clear();
        }
        if (!line.empty()){
            switch(count){
                case 0:{
                    if (line == "2D"){
                        flag3D = false;
                    }
                    else if (line == "3D"){
                        flag3D = true;
                    }
                    else{
                        exit(0);
                    }
                    count = count + 1;
                    break;
                }
                case 1:{
                    k = std::stof(line);
                    count = count + 1;
                    break;
                }
                case 2:{
                    timeStep = std::stoi(line);
                    count = count + 1;
                    break;
                }
                case 3:{
                    int s=0;
                    int temp;
                    std::stringstream lineStream(line);
                    while (lineStream >> temp){
                        switch (s){
                            case (0):{
                                s = s + 1;
                                gridWidth = temp;
                                break;
                            }
                            case (1):{
                                s = s + 1;
                                gridHeight = temp;
                                break;
                            }
                            case (2):{
                                gridDepth = temp;
                                break;
                            }
                        }
                        if (lineStream.peek()==','){
                            lineStream.ignore();
                        }
                    }
                    count = count + 1;
                    break;
                }
                case 4:{
                    initTemp = std::stof(line);
                    count = count + 1;
                    break;
                }
                default:{
                    float temp;
                    std::stringstream lineStream(line);
                    while (lineStream >> temp){
                        heatingSource.push_back(temp);
                        if (lineStream.peek()==','){
                            lineStream.ignore();
                        }
                    }
                    count = count + 1;
                    break;
                }
            }
        }
    }
    printf("Parsing Complete!\n");
    configFile.close();  
}

void Config::generateReport(){
    /*
        Script to generate report
    */
    int fields = 5;
    if (flag3D){
        fields = 7;
    }
    int numEntry = heatingSource.size()/fields;
    printf("\n------------CONFIG REPORT------------\n");
    printf ("3D: %d\n", flag3D);
    printf ("Flow Constant, k = %f \n", k);
    printf ("Time Steps: %d \n", timeStep);
    printf ("Grid Dimensions\n");
    printf ("H = %d, W = %d, D = %d\n", 
            gridHeight, 
            gridWidth, 
            gridDepth);
    printf ("Initial Temperature, T_o = %f \n", initTemp);
    printf ("Heating Source:\n");
    printf ("numEntry: %d \n", numEntry);
    for (int i = 0; i < numEntry; i++){
        if (flag3D){
            printf("x: %f, y: %f, z: %f, W: %f, H: %f, D: %f, Temp: %f\n", 
                heatingSource[i*fields+0],
                heatingSource[i*fields+1],
                heatingSource[i*fields+2],
                heatingSource[i*fields+3],
                heatingSource[i*fields+4],
                heatingSource[i*fields+5],
                heatingSource[i*fields+6]);
        }
        else{
            printf("x: %f, y: %f, W: %f, H: %f, Temp: %f\n", 
                heatingSource[i*fields+0],
                heatingSource[i*fields+1],
                heatingSource[i*fields+2],
                heatingSource[i*fields+3],
                heatingSource[i*fields+4]);
        }
    }
    printf("------------END OF REPORT------------\n\n");
}