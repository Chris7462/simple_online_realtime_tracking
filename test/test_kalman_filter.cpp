// C++ standard library includes
#include <vector>
#include <cmath>
#include <limits>

// Eigen includes
#include <Eigen/Dense>

// Google Test includes
#include <gtest/gtest.h>

// Local includes
#include "sort_backend/kalman_filter.hpp"


using namespace Eigen;

class KalmanFilterTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Setup SORT-specific 7x4 filter configuration
    filter_ = std::make_unique<sort::KalmanFilter>(7, 4);
    setupSortMatrices();
  }

  void TearDown() override
  {
    // Clean up if needed
  }

  void setupSortMatrices()
  {
    // Configure matrices to match SORT's constant velocity model
    // State: [x, y, s, r, dx, dy, ds]

    // State transition matrix F (constant velocity model)
    filter_->F << 1, 0, 0, 0, 1, 0, 0,  // x = x + dx
      0, 1, 0, 0, 0, 1, 0,  // y = y + dy
      0, 0, 1, 0, 0, 0, 1,  // s = s + ds
      0, 0, 0, 1, 0, 0, 0,  // r = r (constant aspect ratio)
      0, 0, 0, 0, 1, 0, 0,  // dx = dx (constant velocity)
      0, 0, 0, 0, 0, 1, 0,  // dy = dy (constant velocity)
      0, 0, 0, 0, 0, 0, 1;  // ds = ds (constant scale velocity)

    // Observation matrix H (observe position, scale, aspect ratio)
    filter_->H << 1, 0, 0, 0, 0, 0, 0,  // observe x
      0, 1, 0, 0, 0, 0, 0,  // observe y
      0, 0, 1, 0, 0, 0, 0,  // observe s
      0, 0, 0, 1, 0, 0, 0;  // observe r

    // Process noise covariance Q
    filter_->Q = MatrixXf::Identity(7, 7) * 0.1f;

    // Measurement noise covariance R
    filter_->R = MatrixXf::Identity(4, 4) * 1.0f;
  }

  // Helper method to check if two values are approximately equal
  bool isApproxEqual(float a, float b, float tolerance = 1e-5f) const
  {
    return std::abs(a - b) < tolerance;
  }

  // Helper method to check if matrices are approximately equal
  bool isApproxEqual(const MatrixXf & a, const MatrixXf & b, float tolerance = 1e-5f) const
  {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
      return false;
    }

    for (int i = 0; i < a.rows(); ++i) {
      for (int j = 0; j < a.cols(); ++j) {
        if (!isApproxEqual(a(i, j), b(i, j), tolerance)) {
          return false;
        }
      }
    }
    return true;
  }

  // Helper method to check if vector is approximately equal
  bool isApproxEqual(const VectorXf & a, const VectorXf & b, float tolerance = 1e-5f) const
  {
    if (a.size() != b.size()) {
      return false;
    }

    for (int i = 0; i < a.size(); ++i) {
      if (!isApproxEqual(a[i], b[i], tolerance)) {
        return false;
      }
    }
    return true;
  }

  // Test data
  std::unique_ptr<sort::KalmanFilter> filter_;
};

// Basic construction and initialization tests
TEST_F(KalmanFilterTest, TestConstruction)
{
  // Test SORT dimensions
  sort::KalmanFilter filter_7x4(7, 4);

  EXPECT_EQ(filter_7x4.getState().size(), 7) << "State vector should have 7 elements";
  EXPECT_EQ(filter_7x4.getCovariance().rows(), 7) << "Covariance should be 7x7";
  EXPECT_EQ(filter_7x4.getCovariance().cols(), 7) << "Covariance should be 7x7";

  // Test other valid dimensions
  sort::KalmanFilter filter_1x1(1, 1);
  sort::KalmanFilter filter_10x5(10, 5);

  EXPECT_EQ(filter_1x1.getState().size(), 1);
  EXPECT_EQ(filter_10x5.getState().size(), 10);
}

TEST_F(KalmanFilterTest, TestInitialState)
{
  // Test that initial state is zero vector
  VectorXf initial_state = filter_->getState();

  EXPECT_EQ(initial_state.size(), 7) << "SORT state should have 7 elements";

  VectorXf expected_zero = VectorXf::Zero(7);
  EXPECT_TRUE(isApproxEqual(initial_state, expected_zero))
    << "Initial state should be zero vector";

  // Test initial covariance is identity
  MatrixXf initial_cov = filter_->getCovariance();
  EXPECT_EQ(initial_cov.rows(), 7);
  EXPECT_EQ(initial_cov.cols(), 7);

  MatrixXf expected_cov = MatrixXf::Identity(7, 7);
  EXPECT_TRUE(isApproxEqual(initial_cov, expected_cov))
    << "Initial covariance should be identity";
}

TEST_F(KalmanFilterTest, TestSystemMatricesInitialization)
{
  // Verify system matrices are initialized correctly
  EXPECT_EQ(filter_->F.rows(), 7);
  EXPECT_EQ(filter_->F.cols(), 7);
  EXPECT_EQ(filter_->H.rows(), 4);
  EXPECT_EQ(filter_->H.cols(), 7);
  EXPECT_EQ(filter_->Q.rows(), 7);
  EXPECT_EQ(filter_->Q.cols(), 7);
  EXPECT_EQ(filter_->R.rows(), 4);
  EXPECT_EQ(filter_->R.cols(), 4);

  // Check that F is initially identity (before our setup)
  sort::KalmanFilter fresh_filter(7, 4);
  MatrixXf expected_F = MatrixXf::Identity(7, 7);
  EXPECT_TRUE(isApproxEqual(fresh_filter.F, expected_F))
    << "Default F matrix should be identity";

  // Check that H is initially zero (before our setup)
  MatrixXf expected_H = MatrixXf::Zero(4, 7);
  EXPECT_TRUE(isApproxEqual(fresh_filter.H, expected_H))
    << "Default H matrix should be zero";
}

// State management tests
TEST_F(KalmanFilterTest, TestSetAndGetState)
{
  // Test setting custom state for SORT tracking
  VectorXf custom_state(7);
  custom_state << 100.0f, 200.0f, 5000.0f, 1.5f, 2.0f, -1.0f, 10.0f;
  // [x=100, y=200, s=5000, r=1.5, dx=2, dy=-1, ds=10]

  filter_->setState(custom_state);
  VectorXf retrieved_state = filter_->getState();

  EXPECT_TRUE(isApproxEqual(retrieved_state, custom_state))
    << "Set/get state should preserve values exactly";
}

TEST_F(KalmanFilterTest, TestStateWithExtremeValues)
{
  // Test with very large values
  VectorXf large_state(7);
  large_state << 1e6f, 1e6f, 1e12f, 100.0f, 1e3f, 1e3f, 1e6f;

  filter_->setState(large_state);
  VectorXf result = filter_->getState();
  EXPECT_TRUE(isApproxEqual(result, large_state, 1e-2f))
    << "Should handle large state values";

  // Test with very small values
  VectorXf small_state(7);
  small_state << 1e-6f, 1e-6f, 1e-12f, 0.01f, 1e-9f, 1e-9f, 1e-12f;

  filter_->setState(small_state);
  result = filter_->getState();
  EXPECT_TRUE(isApproxEqual(result, small_state, 1e-12f))
    << "Should handle small state values";

  // Test with negative values
  VectorXf negative_state(7);
  negative_state << -50.0f, -100.0f, -1000.0f, -2.0f, -5.0f, -3.0f, -50.0f;

  filter_->setState(negative_state);
  result = filter_->getState();
  EXPECT_TRUE(isApproxEqual(result, negative_state))
    << "Should handle negative state values";
}

// Prediction tests
TEST_F(KalmanFilterTest, TestPredictConstantVelocity)
{
  // Test prediction with constant velocity model
  VectorXf initial_state(7);
  initial_state << 100.0f, 200.0f, 5000.0f, 1.5f, 10.0f, -5.0f, 100.0f;
  // Position=(100,200), scale=5000, ratio=1.5, velocity=(10,-5,100)

  filter_->setState(initial_state);

  // Predict one step
  filter_->predict();
  VectorXf predicted_state = filter_->getState();

  // After prediction: position should move by velocity
  VectorXf expected_state(7);
  expected_state << 110.0f, 195.0f, 5100.0f, 1.5f, 10.0f, -5.0f, 100.0f;

  EXPECT_TRUE(isApproxEqual(predicted_state, expected_state, 1e-4f))
    << "Constant velocity prediction incorrect.\nExpected: " << expected_state.transpose()
    << "\nActual: " << predicted_state.transpose();
}

TEST_F(KalmanFilterTest, TestPredictZeroVelocity)
{
  // Test prediction with zero velocities (stationary object)
  VectorXf stationary_state(7);
  stationary_state << 50.0f, 75.0f, 2500.0f, 2.0f, 0.0f, 0.0f, 0.0f;

  filter_->setState(stationary_state);

  VectorXf state_before = filter_->getState();
  filter_->predict();
  VectorXf state_after = filter_->getState();

  // Position and scale should remain the same with zero velocity
  EXPECT_TRUE(isApproxEqual(state_after[0], state_before[0])) << "X position should not change";
  EXPECT_TRUE(isApproxEqual(state_after[1], state_before[1])) << "Y position should not change";
  EXPECT_TRUE(isApproxEqual(state_after[2], state_before[2])) << "Scale should not change";
  EXPECT_TRUE(isApproxEqual(state_after[3], state_before[3])) << "Aspect ratio should not change";
}

TEST_F(KalmanFilterTest, TestPredictMultipleSteps)
{
  // Test prediction over multiple time steps
  VectorXf initial_state(7);
  initial_state << 0.0f, 0.0f, 1000.0f, 1.0f, 5.0f, 3.0f, 20.0f;

  filter_->setState(initial_state);

  // Predict multiple steps and verify linear progression
  std::vector<VectorXf> states;
  states.push_back(filter_->getState());

  for (int step = 1; step <= 5; ++step) {
    filter_->predict();
    states.push_back(filter_->getState());

    // Verify linear progression of position
    EXPECT_TRUE(isApproxEqual(states[step][0], step * 5.0f, 1e-4f))
      << "X position progression incorrect at step " << step;
    EXPECT_TRUE(isApproxEqual(states[step][1], step * 3.0f, 1e-4f))
      << "Y position progression incorrect at step " << step;
    EXPECT_TRUE(isApproxEqual(states[step][2], 1000.0f + step * 20.0f, 1e-4f))
      << "Scale progression incorrect at step " << step;

    // Velocities should remain constant
    EXPECT_TRUE(isApproxEqual(states[step][4], 5.0f)) << "X velocity should remain constant";
    EXPECT_TRUE(isApproxEqual(states[step][5], 3.0f)) << "Y velocity should remain constant";
    EXPECT_TRUE(isApproxEqual(states[step][6], 20.0f)) << "Scale velocity should remain constant";
  }
}

TEST_F(KalmanFilterTest, TestPredictCovarianceIncrease)
{
  // Test that covariance increases with prediction (uncertainty grows)
  VectorXf initial_state(7);
  initial_state << 50.0f, 50.0f, 2500.0f, 1.0f, 1.0f, 1.0f, 5.0f;
  filter_->setState(initial_state);

  MatrixXf cov_before = filter_->getCovariance();
  filter_->predict();
  MatrixXf cov_after = filter_->getCovariance();

  // Covariance should generally increase (or at least not decrease significantly)
  // Check diagonal elements (variances)
  for (int i = 0; i < 7; ++i) {
    EXPECT_GE(cov_after(i, i), cov_before(i, i) - 1e-6f)
      << "Variance should not decrease significantly at index " << i;
  }

  // At least some variances should increase due to process noise
  bool some_increased = false;
  for (int i = 0; i < 7; ++i) {
    if (cov_after(i, i) > cov_before(i, i) + 1e-6f) {
      some_increased = true;
      break;
    }
  }
  EXPECT_TRUE(some_increased) << "Some variances should increase due to process noise";
}

// Update tests
TEST_F(KalmanFilterTest, TestUpdateBasic)
{
  // Test basic update with measurement
  VectorXf initial_state(7);
  initial_state << 100.0f, 200.0f, 5000.0f, 1.5f, 5.0f, -2.0f, 50.0f;
  filter_->setState(initial_state);

  // Create measurement vector [x, y, s, r]
  VectorXf measurement(4);
  measurement << 95.0f, 205.0f, 4800.0f, 1.6f;

  VectorXf state_before = filter_->getState();
  filter_->update(measurement);
  VectorXf state_after = filter_->getState();

  // State should move toward measurement (exact amount depends on Kalman gain)
  // X should move toward measurement
  EXPECT_LT(std::abs(state_after[0] - measurement[0]), std::abs(state_before[0] - measurement[0]))
    << "X position should move toward measurement";

  // Y should move toward measurement
  EXPECT_LT(std::abs(state_after[1] - measurement[1]), std::abs(state_before[1] - measurement[1]))
    << "Y position should move toward measurement";

  // Scale should move toward measurement
  EXPECT_LT(std::abs(state_after[2] - measurement[2]), std::abs(state_before[2] - measurement[2]))
    << "Scale should move toward measurement";

  // Velocities should be adjusted but not directly set
  // They remain part of the internal state estimation
}

TEST_F(KalmanFilterTest, TestUpdateCovarianceDecrease)
{
  // Test that update decreases covariance (uncertainty reduces with measurement)
  VectorXf initial_state(7);
  initial_state << 50.0f, 50.0f, 2500.0f, 1.0f, 0.0f, 0.0f, 0.0f;
  filter_->setState(initial_state);

  MatrixXf cov_before = filter_->getCovariance();

  VectorXf measurement(4);
  measurement << 52.0f, 48.0f, 2600.0f, 1.1f;

  filter_->update(measurement);
  MatrixXf cov_after = filter_->getCovariance();

  // Observed state variances should decrease
  for (int i = 0; i < 4; ++i) {  // First 4 states are observed
    EXPECT_LT(cov_after(i, i), cov_before(i, i) + 1e-6f)
      << "Observed state variance should decrease at index " << i;
  }
}

TEST_F(KalmanFilterTest, TestUpdateWithIdenticalMeasurement)
{
  // Test update when measurement exactly matches predicted state
  VectorXf state(7);
  state << 50.0f, 50.0f, 2500.0f, 1.0f, 0.0f, 0.0f, 0.0f;
  filter_->setState(state);

  // Measurement that exactly matches current estimate
  VectorXf measurement(4);
  measurement << 50.0f, 50.0f, 2500.0f, 1.0f;

  VectorXf state_before = filter_->getState();
  filter_->update(measurement);
  VectorXf state_after = filter_->getState();

  // State should not change much when measurement matches prediction
  EXPECT_TRUE(isApproxEqual(state_after[0], state_before[0], 1e-3f))
    << "X should not change much with identical measurement";
  EXPECT_TRUE(isApproxEqual(state_after[1], state_before[1], 1e-3f))
    << "Y should not change much with identical measurement";
  EXPECT_TRUE(isApproxEqual(state_after[2], state_before[2], 1e-3f))
    << "Scale should not change much with identical measurement";
  EXPECT_TRUE(isApproxEqual(state_after[3], state_before[3], 1e-3f))
    << "Aspect ratio should not change much with identical measurement";
}

TEST_F(KalmanFilterTest, TestUpdateWithExtremeValues)
{
  // Test update with very different measurement
  VectorXf initial_state(7);
  initial_state << 50.0f, 50.0f, 2500.0f, 1.0f, 0.0f, 0.0f, 0.0f;
  filter_->setState(initial_state);

  // Measurement very different from current state
  VectorXf extreme_measurement(4);
  extreme_measurement << 500.0f, 500.0f, 25000.0f, 3.0f;

  filter_->update(extreme_measurement);
  VectorXf updated_state = filter_->getState();

  // State should move toward measurement but not completely
  EXPECT_GT(updated_state[0], initial_state[0])
    << "X should move toward large measurement";
  EXPECT_LT(updated_state[0], extreme_measurement[0])
    << "X should not reach measurement completely";

  EXPECT_GT(updated_state[1], initial_state[1])
    << "Y should move toward large measurement";
  EXPECT_LT(updated_state[1], extreme_measurement[1])
    << "Y should not reach measurement completely";
}

// Predict-update cycle tests
TEST_F(KalmanFilterTest, TestPredictUpdateCycle)
{
  // Test complete predict-update cycle
  VectorXf initial_state(7);
  initial_state << 100.0f, 200.0f, 5000.0f, 1.5f, 10.0f, -5.0f, 100.0f;
  filter_->setState(initial_state);

  // Store initial state
  VectorXf state_0 = filter_->getState();

  // Predict step
  filter_->predict();
  VectorXf state_1 = filter_->getState();

  // Verify prediction moved state according to velocity
  EXPECT_TRUE(isApproxEqual(state_1[0], state_0[0] + state_0[4], 1e-4f))
    << "Prediction should update X by velocity";
  EXPECT_TRUE(isApproxEqual(state_1[1], state_0[1] + state_0[5], 1e-4f))
    << "Prediction should update Y by velocity";
  EXPECT_TRUE(isApproxEqual(state_1[2], state_0[2] + state_0[6], 1e-4f))
    << "Prediction should update scale by scale velocity";

  // Update with measurement
  VectorXf measurement(4);
  measurement << 105.0f, 190.0f, 5200.0f, 1.4f;  // Close to predicted position

  filter_->update(measurement);
  VectorXf state_2 = filter_->getState();

  // State should be influenced by both prediction and measurement
  // Should be between predicted state and measurement
  EXPECT_GT(state_2[0], std::min(state_1[0], measurement[0]) - 1.0f)
    << "Updated X should be reasonable";
  EXPECT_LT(state_2[0], std::max(state_1[0], measurement[0]) + 1.0f)
    << "Updated X should be reasonable";
}

TEST_F(KalmanFilterTest, TestMultiplePredictUpdateCycles)
{
  // Test multiple predict-update cycles to verify stability
  VectorXf initial_state(7);
  initial_state << 0.0f, 0.0f, 1000.0f, 1.0f, 5.0f, 3.0f, 10.0f;
  filter_->setState(initial_state);

  // Simulate tracking with consistent measurements
  std::vector<VectorXf> measurements = {
    (VectorXf(4) << 4.8f, 3.1f, 1008.0f, 1.05f).finished(),
    (VectorXf(4) << 9.9f, 5.8f, 1022.0f, 0.98f).finished(),
    (VectorXf(4) << 15.2f, 9.2f, 1028.0f, 1.02f).finished(),
    (VectorXf(4) << 19.8f, 11.9f, 1045.0f, 0.95f).finished()
  };

  for (size_t i = 0; i < measurements.size(); ++i) {
    filter_->predict();
    filter_->update(measurements[i]);

    VectorXf current_state = filter_->getState();

    // Verify state remains reasonable
    EXPECT_FALSE(std::isnan(current_state[0])) << "State should not become NaN at step " << i;
    EXPECT_FALSE(std::isnan(current_state[1])) << "State should not become NaN at step " << i;
    EXPECT_FALSE(std::isnan(current_state[2])) << "State should not become NaN at step " << i;
    EXPECT_FALSE(std::isnan(current_state[3])) << "State should not become NaN at step " << i;

    // Position should be roughly following the measurements
    EXPECT_LT(std::abs(current_state[0] - measurements[i][0]), 10.0f)
      << "X position should track measurement reasonably at step " << i;
    EXPECT_LT(std::abs(current_state[1] - measurements[i][1]), 10.0f)
      << "Y position should track measurement reasonably at step " << i;
  }
}

// Numerical stability tests
TEST_F(KalmanFilterTest, TestNumericalStability)
{
  // Test with ill-conditioned matrices
  VectorXf state(7);
  state << 1e6f, 1e6f, 1e12f, 100.0f, 1e3f, 1e3f, 1e6f;
  filter_->setState(state);

  // Set very small process noise
  filter_->Q = MatrixXf::Identity(7, 7) * 1e-12f;

  // Set very large measurement noise
  filter_->R = MatrixXf::Identity(4, 4) * 1e6f;

  // Should not crash or produce NaN
  for (int i = 0; i < 10; ++i) {
    filter_->predict();

    VectorXf current_state = filter_->getState();
    for (int j = 0; j < 7; ++j) {
      EXPECT_FALSE(std::isnan(current_state[j]))
        << "State should not become NaN during prediction at iteration " << i;
    }

    VectorXf measurement(4);
    measurement << 1e6f + i, 1e6f - i, 1e12f + i * 1e6f, 50.0f + i * 0.1f;

    filter_->update(measurement);
    current_state = filter_->getState();
    for (int j = 0; j < 7; ++j) {
      EXPECT_FALSE(std::isnan(current_state[j]))
        << "State should not become NaN during update at iteration " << i;
    }
  }
}

TEST_F(KalmanFilterTest, TestCovariancePositiveDefinite)
{
  // Test that covariance matrix remains positive definite
  VectorXf state(7);
  state << 50.0f, 50.0f, 2500.0f, 1.0f, 2.0f, 1.0f, 10.0f;
  filter_->setState(state);

  for (int i = 0; i < 20; ++i) {
    filter_->predict();

    MatrixXf cov = filter_->getCovariance();

    // Check diagonal elements are positive
    for (int j = 0; j < 7; ++j) {
      EXPECT_GT(cov(j, j), 0.0f)
        << "Diagonal covariance element should be positive at step " << i << ", index " << j;
    }

    // Update with noisy measurement
    VectorXf measurement(4);
    measurement << 50.0f + i, 50.0f - i * 0.5f, 2500.0f + i * 10.0f, 1.0f + i * 0.01f;
    filter_->update(measurement);

    cov = filter_->getCovariance();
    for (int j = 0; j < 7; ++j) {
      EXPECT_GT(cov(j, j), 0.0f)
        << "Diagonal covariance element should remain positive after update at step " << i;
    }
  }
}

// Edge case tests
TEST_F(KalmanFilterTest, TestZeroProcessNoise)
{
  // Test with zero process noise (deterministic system)
  filter_->Q = MatrixXf::Zero(7, 7);

  VectorXf state(7);
  state << 50.0f, 50.0f, 2500.0f, 1.0f, 5.0f, 3.0f, 20.0f;
  filter_->setState(state);

  VectorXf state_before = filter_->getState();
  filter_->predict();
  VectorXf state_after = filter_->getState();

  // Should still predict correctly even with zero process noise
  EXPECT_TRUE(isApproxEqual(state_after[0], state_before[0] + state_before[4], 1e-6f))
    << "Zero process noise should still allow prediction";
}

TEST_F(KalmanFilterTest, TestZeroMeasurementNoise)
{
  // Test with zero measurement noise (perfect measurements)
  filter_->R = MatrixXf::Zero(4, 4);

  // This creates a singular matrix, so we use very small values instead
  filter_->R = MatrixXf::Identity(4, 4) * 1e-12f;

  VectorXf state(7);
  state << 100.0f, 200.0f, 5000.0f, 1.5f, 0.0f, 0.0f, 0.0f;
  filter_->setState(state);

  VectorXf measurement(4);
  measurement << 95.0f, 205.0f, 4800.0f, 1.6f;

  filter_->update(measurement);
  VectorXf updated_state = filter_->getState();

  // With very low measurement noise, state should be very close to measurement
  EXPECT_TRUE(isApproxEqual(updated_state[0], measurement[0], 1e-6f))
    << "With low measurement noise, X should match measurement closely";
  EXPECT_TRUE(isApproxEqual(updated_state[1], measurement[1], 1e-6f))
    << "With low measurement noise, Y should match measurement closely";
}

TEST_F(KalmanFilterTest, TestMeasurementDimensionMismatch)
{
  // Test behavior with wrong measurement dimensions
  VectorXf wrong_size_measurement(3);  // Should be size 4
  wrong_size_measurement << 50.0f, 50.0f, 2500.0f;

  VectorXf state(7);
  state << 50.0f, 50.0f, 2500.0f, 1.0f, 0.0f, 0.0f, 0.0f;
  filter_->setState(state);

  // This may crash or handle gracefully depending on implementation
  // We test what actually happens
  try {
    filter_->update(wrong_size_measurement);
    // If it doesn't crash, verify state is still valid
    VectorXf result_state = filter_->getState();
    EXPECT_EQ(result_state.size(), 7) << "State size should remain correct";
  } catch (...) {
    // If it throws, that's acceptable behavior for invalid input
    SUCCEED() << "Exception thrown for wrong measurement dimension - acceptable behavior";
  }
}

// Velocity estimation tests
TEST_F(KalmanFilterTest, TestVelocityEstimation)
{
  // Test that filter estimates velocity correctly from position changes
  VectorXf initial_state(7);
  initial_state << 0.0f, 0.0f, 1000.0f, 1.0f, 0.0f, 0.0f, 0.0f;  // Zero initial velocity
  filter_->setState(initial_state);

  // Sequence of measurements showing consistent movement
  std::vector<VectorXf> measurements = {
    (VectorXf(4) << 5.0f, 3.0f, 1020.0f, 1.0f).finished(),    // +5, +3, +20
    (VectorXf(4) << 10.0f, 6.0f, 1040.0f, 1.0f).finished(),   // +5, +3, +20
    (VectorXf(4) << 15.0f, 9.0f, 1060.0f, 1.0f).finished(),   // +5, +3, +20
    (VectorXf(4) << 20.0f, 12.0f, 1080.0f, 1.0f).finished()   // +5, +3, +20
  };

  for (const auto & measurement : measurements) {
    filter_->predict();
    filter_->update(measurement);
  }

  VectorXf final_state = filter_->getState();

  // After consistent movement, velocity estimates should converge toward true values
  EXPECT_GT(final_state[4], 0.0f) << "X velocity should be positive";
  EXPECT_GT(final_state[5], 0.0f) << "Y velocity should be positive";
  EXPECT_GT(final_state[6], 0.0f) << "Scale velocity should be positive";

  // Rough check that velocities are in expected range
  EXPECT_LT(std::abs(final_state[4] - 5.0f), 3.0f) << "X velocity should be roughly 5";
  EXPECT_LT(std::abs(final_state[5] - 3.0f), 3.0f) << "Y velocity should be roughly 3";
  EXPECT_LT(std::abs(final_state[6] - 20.0f), 15.0f) << "Scale velocity should be roughly 20";
}

TEST_F(KalmanFilterTest, TestVelocityWithStaticObject)
{
  // Test velocity estimation for stationary object
  VectorXf initial_state(7);
  initial_state << 50.0f, 50.0f, 2500.0f, 1.0f, 10.0f, 5.0f, 50.0f;  // Initial non-zero velocity
  filter_->setState(initial_state);

  // Consistent stationary measurements
  VectorXf stationary_measurement(4);
  stationary_measurement << 50.0f, 50.0f, 2500.0f, 1.0f;

  for (int i = 0; i < 10; ++i) {
    filter_->predict();
    filter_->update(stationary_measurement);
  }

  VectorXf final_state = filter_->getState();

  // Velocities should converge toward zero for stationary object
  EXPECT_LT(std::abs(final_state[4]), 2.0f)
    << "X velocity should approach zero for stationary object";
  EXPECT_LT(std::abs(final_state[5]), 2.0f)
    << "Y velocity should approach zero for stationary object";
  EXPECT_LT(std::abs(final_state[6]), 20.0f)
    << "Scale velocity should approach zero for stationary object";
}

// Matrix operation tests
TEST_F(KalmanFilterTest, TestMatrixOperationsValidity)
{
  // Test that matrix operations don't produce invalid results
  VectorXf state(7);
  state << 50.0f, 50.0f, 2500.0f, 1.0f, 1.0f, 1.0f, 5.0f;
  filter_->setState(state);

  // Perform operations and check matrix validity
  for (int i = 0; i < 5; ++i) {
    MatrixXf cov_before = filter_->getCovariance();

    // Check covariance is symmetric (should be maintained by Kalman equations)
    MatrixXf cov_transpose = cov_before.transpose();
    EXPECT_TRUE(isApproxEqual(cov_before, cov_transpose, 1e-6f))
      << "Covariance should remain symmetric at step " << i;

    filter_->predict();

    VectorXf measurement(4);
    measurement << 50.0f + i, 50.0f + i * 0.5f, 2500.0f + i * 10.0f, 1.0f + i * 0.01f;
    filter_->update(measurement);
  }
}

// Performance and stress tests
TEST_F(KalmanFilterTest, TestRepeatedOperations)
{
  // Test filter stability over many operations
  VectorXf initial_state(7);
  initial_state << 100.0f, 200.0f, 5000.0f, 1.5f, 2.0f, -1.0f, 25.0f;
  filter_->setState(initial_state);

  // Run many predict-update cycles
  for (int i = 0; i < 1000; ++i) {
    filter_->predict();

    // Add some realistic noise to measurements
    VectorXf measurement(4);
    float noise_x = (i % 3 - 1) * 0.5f;  // Small random-ish noise
    float noise_y = (i % 5 - 2) * 0.3f;
    float noise_s = (i % 7 - 3) * 10.0f;
    float noise_r = (i % 11 - 5) * 0.01f;

    measurement << 100.0f + i * 2.0f + noise_x,
      200.0f - i * 1.0f + noise_y,
      5000.0f + i * 25.0f + noise_s,
      1.5f + noise_r;

    filter_->update(measurement);

    // Periodically check for stability
    if (i % 100 == 99) {
      VectorXf current_state = filter_->getState();
      MatrixXf current_cov = filter_->getCovariance();

      // Check for NaN values
      for (int j = 0; j < 7; ++j) {
        EXPECT_FALSE(std::isnan(current_state[j]))
          << "State should not become NaN after " << i << " iterations";
      }

      // Check covariance diagonal elements remain positive
      for (int j = 0; j < 7; ++j) {
        EXPECT_GT(current_cov(j, j), 0.0f)
          << "Covariance diagonal should remain positive after " << i << " iterations";
      }
    }
  }
}

TEST_F(KalmanFilterTest, TestExtremeCovariance)
{
  // Test with very large initial covariance
  VectorXf state(7);
  state << 50.0f, 50.0f, 2500.0f, 1.0f, 0.0f, 0.0f, 0.0f;
  filter_->setState(state);

  // Set very large initial uncertainty
  MatrixXf large_cov = MatrixXf::Identity(7, 7) * 1e6f;
  // Note: Can't directly set covariance in this implementation, so we test with large Q instead
  filter_->Q = MatrixXf::Identity(7, 7) * 1e3f;

  VectorXf measurement(4);
  measurement << 100.0f, 100.0f, 5000.0f, 2.0f;  // Very different from state

  // Should handle large uncertainties without numerical issues
  filter_->predict();
  filter_->update(measurement);

  VectorXf updated_state = filter_->getState();
  for (int i = 0; i < 7; ++i) {
    EXPECT_FALSE(std::isnan(updated_state[i]))
      << "State should not become NaN with large covariance";
  }
}

// Integration tests with SORT-specific scenarios
TEST_F(KalmanFilterTest, TestSortSpecificScenarios)
{
  // Test scenarios specific to bounding box tracking

  // Scenario 1: Object moving with constant velocity
  VectorXf moving_object(7);
  moving_object << 100.0f, 150.0f, 4000.0f, 1.2f, 5.0f, -2.0f, 30.0f;
  filter_->setState(moving_object);

  // Predict where object should be
  filter_->predict();
  VectorXf predicted = filter_->getState();

  EXPECT_TRUE(isApproxEqual(predicted[0], 105.0f, 1e-4f)) << "Predicted X incorrect";
  EXPECT_TRUE(isApproxEqual(predicted[1], 148.0f, 1e-4f)) << "Predicted Y incorrect";
  EXPECT_TRUE(isApproxEqual(predicted[2], 4030.0f, 1e-4f)) << "Predicted scale incorrect";
  EXPECT_TRUE(isApproxEqual(predicted[3], 1.2f, 1e-4f)) << "Aspect ratio should remain constant";

  // Scenario 2: Object changing size (zooming in/out)
  VectorXf scaling_object(7);
  scaling_object << 50.0f, 50.0f, 1000.0f, 1.0f, 0.0f, 0.0f, 100.0f;  // Growing in scale
  filter_->setState(scaling_object);

  filter_->predict();
  predicted = filter_->getState();

  EXPECT_TRUE(isApproxEqual(predicted[2], 1100.0f, 1e-4f)) << "Scale should increase by ds";
  EXPECT_TRUE(isApproxEqual(predicted[0], 50.0f, 1e-4f)) << "Position should not change";
  EXPECT_TRUE(isApproxEqual(predicted[1], 50.0f, 1e-4f)) << "Position should not change";
}

TEST_F(KalmanFilterTest, TestAspectRatioStability)
{
  // Test that aspect ratio remains stable when not observed with high noise
  VectorXf state(7);
  state << 50.0f, 50.0f, 2500.0f, 1.5f, 0.0f, 0.0f, 0.0f;
  filter_->setState(state);

  // Set high noise for aspect ratio measurement
  filter_->R(3, 3) = 100.0f;  // High noise for aspect ratio

  // Measurements with varying aspect ratios (due to noise)
  std::vector<float> noisy_ratios = {1.6f, 1.3f, 1.7f, 1.4f, 1.8f};

  for (float ratio : noisy_ratios) {
    VectorXf measurement(4);
    measurement << 50.0f, 50.0f, 2500.0f, ratio;

    filter_->predict();
    filter_->update(measurement);
  }

  VectorXf final_state = filter_->getState();

  // With high measurement noise, aspect ratio should not deviate too much from initial
  EXPECT_LT(std::abs(final_state[3] - 1.5f), 0.5f)
    << "Aspect ratio should be relatively stable with high measurement noise";
}

TEST_F(KalmanFilterTest, TestFilterConvergence)
{
  // Test that filter converges to true values with consistent measurements
  VectorXf initial_state(7);
  initial_state << 0.0f, 0.0f, 1000.0f, 1.0f, 0.0f, 0.0f, 0.0f;  // Wrong initial guess
  filter_->setState(initial_state);

  // True object state: position moving by (3,2) per frame, scale by 15
  float true_x = 50.0f, true_y = 75.0f, true_s = 2000.0f, true_r = 1.8f;
  float true_vx = 3.0f, true_vy = 2.0f, true_vs = 15.0f;

  // Generate measurements from true trajectory with small noise
  for (int step = 1; step <= 20; ++step) {
    filter_->predict();

    // True position at this step
    float true_x_step = true_x + step * true_vx;
    float true_y_step = true_y + step * true_vy;
    float true_s_step = true_s + step * true_vs;

    // Add small measurement noise
    float noise_scale = 0.1f;
    VectorXf measurement(4);
    measurement << true_x_step + (step % 3 - 1) * noise_scale,
      true_y_step + (step % 5 - 2) * noise_scale,
      true_s_step + (step % 7 - 3) * noise_scale,
      true_r + (step % 11 - 5) * 0.001f;

    filter_->update(measurement);
  }

  VectorXf converged_state = filter_->getState();

  // After many measurements, state should be close to true values
  EXPECT_LT(std::abs(converged_state[4] - true_vx), 1.0f)
    << "X velocity should converge to true value";
  EXPECT_LT(std::abs(converged_state[5] - true_vy), 1.0f)
    << "Y velocity should converge to true value";
  EXPECT_LT(std::abs(converged_state[6] - true_vs), 5.0f)
    << "Scale velocity should converge to true value";
}

// Error handling and robustness tests
TEST_F(KalmanFilterTest, TestMeasurementWithNaN)
{
  // Test behavior with NaN in measurement
  VectorXf state(7);
  state << 50.0f, 50.0f, 2500.0f, 1.0f, 1.0f, 1.0f, 5.0f;
  filter_->setState(state);

  VectorXf nan_measurement(4);
  nan_measurement << std::numeric_limits<float>::quiet_NaN(), 50.0f, 2500.0f, 1.0f;

  // Should handle NaN gracefully
  try {
    filter_->update(nan_measurement);

    VectorXf result_state = filter_->getState();

    // State should either remain valid or the NaN should propagate
    // Test what actually happens in the implementation
    bool state_has_nan = false;
    for (int i = 0; i < 7; ++i) {
      if (std::isnan(result_state[i])) {
        state_has_nan = true;
        break;
      }
    }

    // Either all values are valid, or NaN propagated (both are acceptable)
    if (state_has_nan) {
      std::cout << "NaN in measurement caused NaN in state (expected behavior)" << std::endl;
    } else {
      std::cout << "Filter handled NaN measurement gracefully" << std::endl;
    }

  } catch (...) {
    SUCCEED() << "Exception thrown for NaN measurement - acceptable behavior";
  }
}

TEST_F(KalmanFilterTest, TestMeasurementWithInfinite)
{
  // Test behavior with infinite values
  VectorXf state(7);
  state << 50.0f, 50.0f, 2500.0f, 1.0f, 1.0f, 1.0f, 5.0f;
  filter_->setState(state);

  VectorXf inf_measurement(4);
  inf_measurement << std::numeric_limits<float>::infinity(), 50.0f, 2500.0f, 1.0f;

  try {
    filter_->update(inf_measurement);

    VectorXf result_state = filter_->getState();

    // Check if infinite values are handled
    bool state_has_inf = false;
    for (int i = 0; i < 7; ++i) {
      if (std::isinf(result_state[i])) {
        state_has_inf = true;
        break;
      }
    }

    if (state_has_inf) {
      std::cout << "Infinite measurement caused infinite state (expected behavior)" << std::endl;
    }

  } catch (...) {
    SUCCEED() << "Exception thrown for infinite measurement - acceptable behavior";
  }
}

// Mathematical correctness tests
TEST_F(KalmanFilterTest, TestKalmanGainProperties)
{
  // Test properties of the Kalman filter updates
  VectorXf state(7);
  state << 100.0f, 200.0f, 5000.0f, 1.5f, 0.0f, 0.0f, 0.0f;
  filter_->setState(state);

  // With very low measurement noise, update should trust measurement more
  filter_->R = MatrixXf::Identity(4, 4) * 0.01f;  // Very low noise

  VectorXf measurement_low_noise(4);
  measurement_low_noise << 90.0f, 210.0f, 4500.0f, 1.8f;  // Different from state

  VectorXf state_before = filter_->getState();
  filter_->update(measurement_low_noise);
  VectorXf state_after_low_noise = filter_->getState();

  // Reset and test with high measurement noise
  filter_->setState(state);
  filter_->R = MatrixXf::Identity(4, 4) * 100.0f;  // Very high noise

  filter_->update(measurement_low_noise);
  VectorXf state_after_high_noise = filter_->getState();

  // With low measurement noise, state should move more toward measurement
  float movement_low_noise = std::abs(state_after_low_noise[0] - state_before[0]);
  float movement_high_noise = std::abs(state_after_high_noise[0] - state_before[0]);

  EXPECT_GT(movement_low_noise, movement_high_noise)
    << "Low measurement noise should cause larger state updates";
}

TEST_F(KalmanFilterTest, TestPredictionWithoutUpdate)
{
  // Test prediction accuracy when no updates are performed
  VectorXf initial_state(7);
  initial_state << 0.0f, 0.0f, 1000.0f, 1.0f, 10.0f, 5.0f, 50.0f;
  filter_->setState(initial_state);

  // Predict several steps without updates
  std::vector<VectorXf> predicted_states;
  predicted_states.push_back(filter_->getState());

  for (int step = 1; step <= 10; ++step) {
    filter_->predict();
    predicted_states.push_back(filter_->getState());

    // Verify linear progression
    VectorXf current = predicted_states[step];
    EXPECT_TRUE(isApproxEqual(current[0], step * 10.0f, 1e-4f))
      << "X prediction incorrect at step " << step;
    EXPECT_TRUE(isApproxEqual(current[1], step * 5.0f, 1e-4f))
      << "Y prediction incorrect at step " << step;
    EXPECT_TRUE(isApproxEqual(current[2], 1000.0f + step * 50.0f, 1e-4f))
      << "Scale prediction incorrect at step " << step;
  }
}

TEST_F(KalmanFilterTest, TestCovarianceGrowthRate)
{
  // Test that covariance grows at expected rate during prediction
  VectorXf state(7);
  state << 50.0f, 50.0f, 2500.0f, 1.0f, 1.0f, 1.0f, 5.0f;
  filter_->setState(state);

  // Set known process noise
  filter_->Q = MatrixXf::Identity(7, 7) * 0.5f;

  MatrixXf cov_initial = filter_->getCovariance();

  filter_->predict();
  MatrixXf cov_after_one = filter_->getCovariance();

  filter_->predict();
  MatrixXf cov_after_two = filter_->getCovariance();

  // Covariance should grow at each prediction step
  for (int i = 0; i < 7; ++i) {
    EXPECT_GT(cov_after_one(i, i), cov_initial(i, i))
      << "Covariance should increase after first prediction at index " << i;
    EXPECT_GT(cov_after_two(i, i), cov_after_one(i, i))
      << "Covariance should continue increasing after second prediction at index " << i;
  }
}

TEST_F(KalmanFilterTest, TestObservabilityMatrix)
{
  // Test that the observation matrix H correctly maps state to measurements
  VectorXf test_state(7);
  test_state << 123.0f, 456.0f, 7890.0f, 2.5f, 10.0f, 20.0f, 30.0f;

  // H * x should extract the observable components
  VectorXf observed = filter_->H * test_state;

  EXPECT_EQ(observed.size(), 4) << "Observation should have 4 elements";
  EXPECT_TRUE(isApproxEqual(observed[0], test_state[0])) << "Should observe X position";
  EXPECT_TRUE(isApproxEqual(observed[1], test_state[1])) << "Should observe Y position";
  EXPECT_TRUE(isApproxEqual(observed[2], test_state[2])) << "Should observe scale";
  EXPECT_TRUE(isApproxEqual(observed[3], test_state[3])) << "Should observe aspect ratio";
}

TEST_F(KalmanFilterTest, TestStateTransitionMatrix)
{
  // Test that the state transition matrix F correctly implements constant velocity
  VectorXf test_state(7);
  test_state << 10.0f, 20.0f, 1000.0f, 1.5f, 3.0f, -2.0f, 50.0f;

  // F * x should implement the constant velocity model
  VectorXf predicted = filter_->F * test_state;

  EXPECT_EQ(predicted.size(), 7) << "Predicted state should have 7 elements";

  // Check constant velocity predictions
  EXPECT_TRUE(isApproxEqual(predicted[0], 13.0f)) << "X should be x + dx";
  EXPECT_TRUE(isApproxEqual(predicted[1], 18.0f)) << "Y should be y + dy";
  EXPECT_TRUE(isApproxEqual(predicted[2], 1050.0f)) << "Scale should be s + ds";
  EXPECT_TRUE(isApproxEqual(predicted[3], 1.5f)) << "Aspect ratio should remain constant";
  EXPECT_TRUE(isApproxEqual(predicted[4], 3.0f)) << "X velocity should remain constant";
  EXPECT_TRUE(isApproxEqual(predicted[5], -2.0f)) << "Y velocity should remain constant";
  EXPECT_TRUE(isApproxEqual(predicted[6], 50.0f)) << "Scale velocity should remain constant";
}

TEST_F(KalmanFilterTest, TestFilterConsistency)
{
  // Test that repeated identical operations produce consistent results
  VectorXf initial_state(7);
  initial_state << 50.0f, 50.0f, 2500.0f, 1.0f, 2.0f, 1.0f, 10.0f;

  VectorXf measurement(4);
  measurement << 52.0f, 51.0f, 2510.0f, 1.02f;

  // Run same prediction-update cycle multiple times
  std::vector<VectorXf> final_states;

  for (int run = 0; run < 3; ++run) {
    sort::KalmanFilter test_filter(7, 4);
    setupSortMatrices();  // Reset to same configuration

    // Copy our configuration to test filter
    test_filter.F = filter_->F;
    test_filter.H = filter_->H;
    test_filter.Q = filter_->Q;
    test_filter.R = filter_->R;

    test_filter.setState(initial_state);
    test_filter.predict();
    test_filter.update(measurement);

    final_states.push_back(test_filter.getState());
  }

  // All runs should produce identical results
  for (size_t i = 1; i < final_states.size(); ++i) {
    EXPECT_TRUE(isApproxEqual(final_states[0], final_states[i], 1e-10f))
      << "Filter should produce consistent results across runs";
  }
}
