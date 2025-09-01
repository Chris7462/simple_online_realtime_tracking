#include <iostream>
#include <iomanip>

#include "sort_backend/sort.hpp"


using namespace sort;

void printMatrix(const Eigen::MatrixXf & matrix, const std::string & name)
{
  std::cout << name << " (" << matrix.rows() << "x" << matrix.cols() << "):\n";
  if (matrix.rows() == 0) {
    std::cout << "  (empty)\n";
    return;
  }

  std::cout << std::fixed << std::setprecision(2);
  for (int i = 0; i < matrix.rows(); ++i) {
    std::cout << "  ";
    for (int j = 0; j < matrix.cols(); ++j) {
      std::cout << std::setw(8) << matrix(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

int main()
{
  std::cout << "SORT Tracker Example\n";
  std::cout << "====================\n\n";

  // Initialize SORT tracker
  Sort tracker(3, 3, 0.3f);

  // Simulate detections over multiple frames
  std::vector<Eigen::MatrixXf> frame_detections = {
    // Frame 1: Two detections
    (Eigen::MatrixXf(2, 5) <<
      10, 10, 50, 50, 0.9,    // Detection 1: [x1, y1, x2, y2, confidence]
      100, 100, 140, 140, 0.8  // Detection 2
    ).finished(),

    // Frame 2: Both objects moved
    (Eigen::MatrixXf(2, 5) <<
      15, 12, 55, 52, 0.85,   // Detection 1 moved slightly
      105, 95, 145, 135, 0.9  // Detection 2 moved
    ).finished(),

    // Frame 3: Only one detection (object 2 disappeared)
    (Eigen::MatrixXf(1, 5) <<
      20, 14, 60, 54, 0.88    // Only detection 1
    ).finished(),

    // Frame 4: Both objects back
    (Eigen::MatrixXf(2, 5) <<
      25, 16, 65, 56, 0.92,   // Detection 1 continued
      110, 90, 150, 130, 0.87 // Detection 2 reappeared
    ).finished(),

    // Frame 5: No detections (empty frame)
    Eigen::MatrixXf::Zero(0, 5)
  };

  // Process each frame
  for (size_t frame = 0; frame < frame_detections.size(); ++frame) {
    std::cout << "Frame " << (frame + 1) << ":\n";
    std::cout << "--------\n";

    const auto & detections = frame_detections[frame];
    printMatrix(detections, "Input Detections");

    // Update tracker
    Eigen::MatrixXf tracks = tracker.update(detections);
    printMatrix(tracks, "Output Tracks [x1, y1, x2, y2, track_id]");

    std::cout << "Active trackers: " << tracker.getTrackerCount() << "\n";
    std::cout << "Frame count: " << tracker.getFrameCount() << "\n\n";
  }

  // Demonstrate tracker reset
  std::cout << "Resetting tracker...\n";
  tracker.reset();
  std::cout << "After reset - Active trackers: " << tracker.getTrackerCount()
            << ", Frame count: " << tracker.getFrameCount() << "\n";

  return 0;
}
