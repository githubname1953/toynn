#include "Eigen/Dense"
#include <iostream>
#include <cmath>
#include <random>
#include<ctime>
#include<vector>
#include<algorithm>
using namespace std;

class Node{
public:
    Eigen::MatrixXf weight;
    Eigen::MatrixXf bias;
    Eigen::MatrixXf I;
    Eigen::MatrixXf nout;
    Eigen::MatrixXf dweight;
    Eigen::MatrixXf grad;
    Eigen::MatrixXf error;

    Node(int in, int out)
    {
        weight = Eigen::MatrixXf::Random(out,in);
        bias = Eigen::MatrixXf::Random(out,1);
        I = Eigen::MatrixXf::Constant(out,1,1.0f);
        nout = Eigen::MatrixXf::Constant(out,1,0.0f);
        dweight = Eigen::MatrixXf::Constant(out,in,0.0f);
        //grad = Eigen::MatrixXf::Constant(out,1,0.0f);
        error = Eigen::MatrixXf::Constant(out,1,0.0f);
    }

    void propagate( Eigen::MatrixXf &i )
    {
        nout.noalias() = weight*i + bias;
        nout = (1 + Eigen::exp(-nout.array())).inverse().matrix(); // sigmoid
        //nout = nout.unaryExpr([](float x){return max(x,0.0f);}); // relu
    }

    void backprop( Eigen::MatrixXf &i, float lrate )
    {
        error = (error.cwiseProduct(nout.cwiseProduct(I-nout)));
        //error = (error.cwiseProduct(nout.unaryExpr([](float x){return float(x>0.0f);})));
        dweight.noalias() = lrate*(error*i.transpose());
        weight += dweight;
        bias += lrate*error;
    }
};

class NN {
public:
    vector<int> num;
    int size;
    vector<Node> v; 
    
    NN(vector<int> &config)
    {
        num = config;
        for(int i=0; i<config.size()-1; i++)
        {
            v.emplace_back(config[i],config[i+1]);
        }

        size = config.size()-1; // equal to v.size()
    }
    
    void predict(Eigen::MatrixXf &input, Eigen::MatrixXf &output)
    {   
        v[0].propagate(input);
        for(int i=1; i<v.size(); i++)
        {
            v[i].propagate(v[i-1].nout);
        }

        output = v[size-1].nout;
        //cout<<outputv<<endl;
        //Eigen::Map<Eigen::MatrixXf>(output, outputv.rows(), outputv.cols()) = outputv;
    }
    
    void train(Eigen::MatrixXf &input, Eigen::MatrixXf &target, float lr)
    {
        v[0].propagate(input);
        for(int i=1; i<v.size(); i++)
        {
            v[i].propagate(v[i-1].nout);
        }

        v[size-1].error = target - v[size-1].nout;
        v[size-1].backprop(v[size-2].nout, lr);

        for(int i=size-2; i>0; i--)
        {
            v[i].error.noalias() = v[i+1].weight.transpose()*v[i+1].error;
            v[i].backprop(v[i-1].nout, lr);
        }

        v[0].error = v[1].weight.transpose()*v[1].error;
        v[0].backprop(input, lr);
        //Eigen::MatrixXf inputv = Eigen::Map<Eigen::MatrixXf>(input,num[0],1);
        
        // Eigen::MatrixXf hiddenv = weightsh*inputv + biash;
        // hiddenv = (1 + Eigen::exp(-hiddenv.array())).inverse().matrix();
        // //hiddenv = hiddenv.unaryExpr( [](float x){return x>0 ? x : 0;}  );
        // //cout<<hiddenv<<endl;
        // Eigen::MatrixXf outputv = weightso*hiddenv + biaso;
        // outputv = (1 + Eigen::exp(-outputv.array())).inverse().matrix();
        // //outputv = outputv.unaryExpr( [](float x){return x>0 ? x : 0;}  );
        
        
        // Eigen::MatrixXf targetv = Eigen::Map<Eigen::MatrixXf>(target,on,1);
        
        // Eigen::MatrixXf error = targetv - outputv;
        // Eigen::MatrixXf errorh = weightso.transpose()*error;
        
        // Eigen::MatrixXf grad = lr*(error.cwiseProduct(outputv.cwiseProduct(Io-outputv)));
        // Eigen::MatrixXf dweightso = (grad*hiddenv.transpose());
        // weightso += dweightso;
        // biaso += grad;
        
        // Eigen::MatrixXf gradh = lr*(errorh.cwiseProduct(hiddenv.cwiseProduct(Ih-hiddenv)));
        // Eigen::MatrixXf dweightsh = (gradh*inputv.transpose());
        // weightsh += dweightsh;
        // biash += gradh;
        
        //cout<<grad<<endl; cout<<endl; cout<<gradh<<endl;
        
        //Eigen::Map<Eigen::MatrixXf>(output, outputv.rows(), outputv.cols()) = outputv;
    }
    
    
};
