///////////////////////////////////////////////////////////////////////////////
// kalman_filter.hpp: Generic Kalman Filter implementation using Eigen
// Clean implementation of standard Kalman filter equations
// Supports arbitrary state and measurement dimensions
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Eigen/Dense>

namespace sort
{

class KalmanFilter
{
public:
  using MatrixXf = Eigen::MatrixXf;
  using VectorXf = Eigen::VectorXf;

  /**
   * @brief Construct Kalman filter with specified dimensions
   * @param dim_x Dimension of state vector
   * @param dim_z Dimension of measurement vector
   */
  KalmanFilter(int dim_x, int dim_z);

  /**
   * @brief Predict next state using motion model
   */
  void predict();

  /**
   * @brief Update state with new measurement
   * @param z Measurement vector
   */
  void update(const VectorXf & z);

  /**
   * @brief Get current state estimate
   * @return Current state vector
   */
  const VectorXf & getState() const;

  /**
   * @brief Set state vector
   * @param x New state vector
   */
  void setState(const VectorXf & x);

  /**
   * @brief Get state covariance matrix
   * @return Current covariance matrix
   */
  const MatrixXf & getCovariance() const;

  // Public access to system matrices for configuration
  MatrixXf F;  // State transition matrix
  MatrixXf H;  // Observation matrix
  MatrixXf Q;  // Process noise covariance
  MatrixXf R;  // Measurement noise covariance

private:
  VectorXf x_;  // State vector
  MatrixXf P_;  // State covariance matrix

  int dim_x_;   // State dimension
  int dim_z_;   // Measurement dimension
};

} // namespace sort
