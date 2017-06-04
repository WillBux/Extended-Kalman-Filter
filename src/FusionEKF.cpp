#include "FusionEKF.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
* Constructor.
*/
FusionEKF::FusionEKF() {

    is_initialized_ = false;
    previous_timestamp_ = 0;

    // Initialize
    H_laser_ = MatrixXd(2, 4);
    H_laser_ << 1, 0, 0, 0,
            0, 1, 0, 0;

    R_laser_ = MatrixXd(2, 2); //laser covariance
    R_laser_ << 0.0225, 0,
            0, 0.0225;

    R_radar_ = MatrixXd(3, 3); //radar covariance
    R_radar_ << 0.0775, 0, 0,
            0, 0.0775, 0,
            0, 0, 0.0775;


}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {

        ekf_.x_ = VectorXd(4);

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

            float rho = measurement_pack.raw_measurements_[0]; // range
            float phi = measurement_pack.raw_measurements_[1]; // bearing
            float rho_dot = measurement_pack.raw_measurements_[2]; // velocity of rho
            //convert to cartesian
            float x = rho * cos(phi);
            float y = rho * sin(phi);
            float vx = rho_dot * cos(phi);
            float vy = rho_dot * sin(phi);
            ekf_.x_ << x, y, vx , vy;
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
        }
        if (fabs(ekf_.x_(0)) < 0.00001 and fabs(ekf_.x_(1)) < 0.00001){
            ekf_.x_(0) = 0.00001;
            ekf_.x_(1) = 0.00001;
        }
        //covariance matrix
        ekf_.P_ = MatrixXd(4, 4);
        ekf_.P_ << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1000, 0,
                0, 0, 0, 1000;

        //getting timestamp
        previous_timestamp_ = measurement_pack.timestamp_;
        //make sure not to intialize again
        is_initialized_ = true;
        return;
    }
    /*****************************************************************************
    *  Prediction
    ****************************************************************************/

    //turn the time stamps (measured in ms) into a change in time in seconds
    float dt = (measurement_pack.timestamp_ - previous_timestamp_);
    dt /= 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;

    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1;

    //noise numbers given by Udacity
    float noise_ax = 9.0;
    float noise_ay = 9.0;
    // Precomputed values to speed up processing
    float dtSq = dt * dt; //dt^2
    float dtCu = dtSq * dt; //dt^3
    float dt4th = dtCu * dt; //dt^4
    float dt4th_4 = dt4th / 4; //dt^4/4
    float dtCu_2 = dtCu / 2; //dt^3/2
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ << dt4th_4 * noise_ax, 0, dtCu_2 * noise_ax, 0,
            0, dt4th_4 * noise_ay, 0, dtCu_2 * noise_ay,
            dtCu_2 * noise_ax, 0, dtSq * noise_ax, 0,
            0, dtCu_2 * noise_ay, 0, dtSq * noise_ay;

    ekf_.Predict();

    /*****************************************************************************
     *  Update
     ****************************************************************************/


    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.R_ = R_radar_;
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    } else {
        // Laser updates
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }
    // print output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}