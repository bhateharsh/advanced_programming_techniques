#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

class Config{
    private:
        std::string confPath;
    public:
        Config(const std::string& filePath);
        ~Config();
        void parseConfig();
        void generateReport();
        bool flag3D;
        float k;
        int timeStep;
        int gridWidth = 0;
        int gridHeight = 0;
        int gridDepth = 1;
        float initTemp;
        std::vector<float> heatingSource;
};