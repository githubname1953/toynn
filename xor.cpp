#include<iostream>
#include<random>
#include <cmath>
#include<vector>
#include<ctime>
#include "Eigen/Dense"
#include "Eigen/Core"
#include "nn.cpp"
#include "xoshiro.hpp"
using namespace std;
#define nsize 512
#define PI 3.14159265359f

int main()
{
    int i,j,k;
    random_device rd; xoshiro128plus32 xg(rd()); 
    vector<int> configuration={1,16,16,1};
    float inputs[nsize][1];
    float outputs[nsize][1];
    float expected[nsize][1];
    for(i=0;i<nsize;i++) { inputs[i][0]=(float)i/(float)nsize; expected[i][0]=sin(inputs[i][0]*PI); } 
    Eigen::MatrixXf inputv = Eigen::MatrixXf::Zero(configuration[0],1);
    Eigen::MatrixXf targetv = Eigen::MatrixXf::Zero(configuration[configuration.size()-1],1);
    Eigen::MatrixXf outputv = Eigen::MatrixXf::Zero(configuration[configuration.size()-1],1);
    
    
    NN nn1(configuration);
    
    //nn1.train(m,n,0.1);
    for(i=0;i<10000000;i++)
    {
        j = xg() % nsize;
        inputv = Eigen::Map<Eigen::MatrixXf>(inputs[j],configuration[0],1);
        targetv = Eigen::Map<Eigen::MatrixXf>(expected[j],configuration[configuration.size()-1],1);
        nn1.train(inputv,targetv,0.1);
    }
    
    for(i=0;i<10;i++)
    {
        float it = (float)xg()/(float)xg.max();
        inputv = Eigen::Map<Eigen::MatrixXf>(&it,configuration[0],1);
        nn1.predict(inputv,outputv);
        cout<<outputv(0,0)<<'\t'<<sin(it*PI)<<endl;
    }
	
	
    
    return 0;
}
