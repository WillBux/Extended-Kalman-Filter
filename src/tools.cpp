#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    //Calculate the error using the Root-Mean-Squared Error
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    //making sure that the RMSE calculations do not break
    if(estimations.size() == 0){
        cout << "estimations.size() == 0" << endl;
        return rmse;
    }
    if(estimations.size() != ground_truth.size()){
        cout << "estimations.size() != ground_truth.size()" << endl;
        return rmse;
    }
    // Get the squared residuals
    for(unsigned int i=0; i < estimations.size(); ++i){
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array()*residual.array();
        rmse += residual;
    }

    //Get the mean
    rmse = rmse / estimations.size();
    rmse = rmse.array().sqrt();
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    //calculate Jacobian matrix
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);
    MatrixXd J(3,4);

    //pre compute to save computing time
    float c1 = px*px+py*py;
    // make sure the program does not return /0 error
    if(fabs(c1) < 0.0000001){
        c1 = 0.0000001;//use a small number close to 0 instead
    }
    float c2 = sqrt(c1);
    float c3 = (c1*c2);
    // Compute the Jacobian matrix
    J << (px/c2), (py/c2), 0, 0,
            -(py/c1), (px/c1), 0, 0,
            py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;
    return J;

}
