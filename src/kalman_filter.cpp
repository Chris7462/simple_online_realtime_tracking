#include "sort/kalman_filter.hpp"


namespace sort
{

KalmanFilter::KalmanFilter(int dim_x, int dim_z)
: dim_x_(dim_x), dim_z_(dim_z)
{

	// Initialize state and covariance
	x_ = VectorXf::Zero(dim_x_);
	P_ = MatrixXf::Identity(dim_x_, dim_x_);

	// Initialize system matrices
	F = MatrixXf::Identity(dim_x_, dim_x_);
	H = MatrixXf::Zero(dim_z_, dim_x_);
	Q = MatrixXf::Identity(dim_x_, dim_x_);
	R = MatrixXf::Identity(dim_z_, dim_z_);
}

void KalmanFilter::predict()
{
	// State prediction: x = F * x
	x_ = F * x_;

	// Covariance prediction: P = F * P * F' + Q
	P_ = F * P_ * F.transpose() + Q;
}

void KalmanFilter::update(const VectorXf& z)
{
	// Innovation: y = z - H * x
	VectorXf y = z - H * x_;

	// Innovation covariance: S = H * P * H' + R
	MatrixXf S = H * P_ * H.transpose() + R;

	// Kalman gain: K = P * H' * S^-1
	MatrixXf K = P_ * H.transpose() * S.inverse();

	// State update: x = x + K * y
	x_ = x_ + K * y;

	// Covariance update: P = P - K * H * P
	P_ = P_ - K * H * P_;
}

} // namespace sort
