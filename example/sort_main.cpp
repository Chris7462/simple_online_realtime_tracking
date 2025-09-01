#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>

#include <opencv2/opencv.hpp>

#include "sort_backend/sort.hpp"


namespace fs = std::filesystem;
using namespace sort;

struct Arguments
{
  std::string seq_path = "/data/MOT15";
  std::string phase = "train";
  int max_age = 1;
  int min_hits = 3;
  float iou_threshold = 0.3f;
  bool display = false;
};

Arguments parseArgs(int argc, char * argv[])
{
  Arguments args;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--display") {
      args.display = true;
    } else if (arg == "--seq_path" && i + 1 < argc) {
      args.seq_path = argv[++i];
    } else if (arg == "--phase" && i + 1 < argc) {
      args.phase = argv[++i];
    } else if (arg == "--max_age" && i + 1 < argc) {
      args.max_age = std::stoi(argv[++i]);
    } else if (arg == "--min_hits" && i + 1 < argc) {
      args.min_hits = std::stoi(argv[++i]);
    } else if (arg == "--iou_threshold" && i + 1 < argc) {
      args.iou_threshold = std::stof(argv[++i]);
    }
  }

  return args;
}

// Generate random colors for visualization (equivalent to Python's np.random.rand(32, 3))
std::vector<cv::Scalar> generateColors(int num_colors = 32)
{
  std::vector<cv::Scalar> colors;
  std::mt19937 gen(0);  // Same seed as Python for consistency
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  for (int i = 0; i < num_colors; ++i) {
    float r = dis(gen);
    float g = dis(gen);
    float b = dis(gen);
    // Convert to BGR format for OpenCV (0-255 range)
    colors.push_back(cv::Scalar(b * 255, g * 255, r * 255));
  }

  return colors;
}

// Load detection file in MOT format
std::vector<std::vector<float>> loadDetections(const std::string & filename)
{
  std::vector<std::vector<float>> detections;
  std::ifstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  std::string line;
  while (std::getline(file, line)) {
    std::vector<float> row;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
      try {
        row.push_back(std::stof(cell));
      } catch (const std::exception & e) {
        // Skip invalid entries
        row.push_back(0.0f);
      }
    }

    if (row.size() >= 7) {  // Minimum required columns
      detections.push_back(row);
    }
  }

  return detections;
}

// Get detections for specific frame
Eigen::MatrixXf getFrameDetections(
  const std::vector<std::vector<float>> & all_detections, int frame)
{
  std::vector<std::vector<float>> frame_dets;

  for (const auto & det : all_detections) {
    if (std::abs(det[0] - frame) < 1e-6) {  // Frame number match (column 0)
      frame_dets.push_back(det);
    }
  }

  if (frame_dets.empty()) {
    return Eigen::MatrixXf::Zero(0, 5);
  }

  // Convert to Eigen matrix [x1, y1, x2, y2, score]
  Eigen::MatrixXf detections(frame_dets.size(), 5);

  for (size_t i = 0; i < frame_dets.size(); ++i) {
    const auto & det = frame_dets[i];
    // MOT format: frame, id, x, y, w, h, conf, ...
    // Convert [x, y, w, h] to [x1, y1, x2, y2]
    float x1 = det[2];
    float y1 = det[3];
    float w = det[4];
    float h = det[5];
    float x2 = x1 + w;
    float y2 = y1 + h;
    float conf = det[6];

    detections.row(i) << x1, y1, x2, y2, conf;
  }

  return detections;
}

// Find maximum frame number in detections
int getMaxFrame(const std::vector<std::vector<float>> & detections)
{
  int max_frame = 0;
  for (const auto & det : detections) {
    max_frame = std::max(max_frame, static_cast<int>(det[0]));
  }
  return max_frame;
}

// Write tracking results in MOT format
void writeResults(std::ofstream & out_file, int frame, const Eigen::MatrixXf & tracks)
{
  for (int i = 0; i < tracks.rows(); ++i) {
    float x1 = tracks(i, 0);
    float y1 = tracks(i, 1);
    float x2 = tracks(i, 2);
    float y2 = tracks(i, 3);
    int track_id = static_cast<int>(tracks(i, 4));

    // Convert back to [x, y, w, h] format
    float w = x2 - x1;
    float h = y2 - y1;

    // MOT output format: frame, id, x, y, w, h, conf, x, y, z
    out_file << frame << "," << track_id << ","
             << std::fixed << std::setprecision(2)
             << x1 << "," << y1 << "," << w << "," << h
             << ",1,-1,-1,-1\n";
  }
}

int main(int argc, char * argv[])
{
  Arguments args = parseArgs(argc, argv);

  std::cout << "SORT C++ Implementation\n";
  std::cout << "Parameters:\n";
  std::cout << "  seq_path: " << args.seq_path << "\n";
  std::cout << "  phase: " << args.phase << "\n";
  std::cout << "  max_age: " << args.max_age << "\n";
  std::cout << "  min_hits: " << args.min_hits << "\n";
  std::cout << "  iou_threshold: " << args.iou_threshold << "\n";
  std::cout << "  display: " << (args.display ? "true" : "false") << "\n\n";

  // Generate colors for visualization
  std::vector<cv::Scalar> colors = generateColors(32);

  // Create output directory
  if (!fs::exists("output")) {
    fs::create_directories("output");
  }

  // Find all detection files
  std::string pattern_dir = args.seq_path + "/" + args.phase;
  std::vector<std::string> det_files;

  try {
    for (const auto & seq_dir : fs::directory_iterator(pattern_dir)) {
      if (seq_dir.is_directory()) {
        std::string det_file = seq_dir.path() / "det" / "det.txt";
        if (fs::exists(det_file)) {
          det_files.push_back(det_file);
        }
      }
    }
  } catch (const fs::filesystem_error & e) {
    std::cerr << "Error accessing directory: " << e.what() << "\n";
    return 1;
  }

  if (det_files.empty()) {
    std::cerr << "No detection files found in " << pattern_dir << "\n";
    return 1;
  }

  // Sort detection files for consistent processing order
  std::sort(det_files.begin(), det_files.end());

  double total_time = 0.0;
  int total_frames = 0;

  // Process each sequence
  for (const std::string & seq_det_file : det_files) {
    std::cout << "Processing " << seq_det_file << "\n";

    // Create SORT tracker instance
    Sort mot_tracker(args.max_age, args.min_hits, args.iou_threshold);

    // Load sequence detections
    std::vector<std::vector<float>> seq_dets;
    try {
      seq_dets = loadDetections(seq_det_file);
    } catch (const std::exception & e) {
      std::cerr << "Error loading " << seq_det_file << ": " << e.what() << "\n";
      continue;
    }

    // Extract sequence name from path
    fs::path seq_path(seq_det_file);
    std::string seq_name = seq_path.parent_path().parent_path().filename().string();

    // Create output file
    std::string output_file = "output/" + seq_name + ".txt";
    std::ofstream out_file(output_file);

    if (!out_file.is_open()) {
      std::cerr << "Cannot create output file: " << output_file << "\n";
      continue;
    }

    // Get maximum frame number
    int max_frame = getMaxFrame(seq_dets);

    // Initialize OpenCV window if display is enabled
    cv::Mat display_img;
    std::string window_name;
    if (args.display) {
      window_name = seq_name + " Tracked Targets";
      cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    }

    // Process each frame
    for (int frame = 1; frame <= max_frame; ++frame) {
      // Get detections for this frame
      Eigen::MatrixXf dets = getFrameDetections(seq_dets, frame);

      total_frames++;

      // Load and display image if display mode is enabled
      if (args.display) {
        std::string img_path = "MOT15/" + args.phase + "/" + seq_name + "/img1/" +
          std::to_string(frame).insert(0, 6 - std::to_string(frame).length(), '0') + ".jpg";

        display_img = cv::imread(img_path);
        if (display_img.empty()) {
          std::cerr << "Warning: Could not load image " << img_path << "\n";
          // Create a blank image as fallback
          display_img = cv::Mat::zeros(480, 640, CV_8UC3);
        }
      }

      // Track timing
      auto start_time = std::chrono::high_resolution_clock::now();

      // Update tracker
      Eigen::MatrixXf tracks = mot_tracker.update(dets);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
      total_time += duration.count() / 1000000.0;  // Convert to seconds

      // Write results
      writeResults(out_file, frame, tracks);

      // Draw tracking results if display is enabled
      if (args.display && !display_img.empty()) {
        for (int i = 0; i < tracks.rows(); ++i) {
          int x1 = static_cast<int>(tracks(i, 0));
          int y1 = static_cast<int>(tracks(i, 1));
          int x2 = static_cast<int>(tracks(i, 2));
          int y2 = static_cast<int>(tracks(i, 3));
          int track_id = static_cast<int>(tracks(i, 4));

          // Get color for this track (cycle through available colors)
          cv::Scalar color = colors[track_id % 32];

          // Draw bounding box rectangle (thickness=3 to match Python lw=3)
          cv::rectangle(display_img, cv::Point(x1, y1), cv::Point(x2, y2), color, 3);

          // Optionally draw track ID
          std::string id_text = std::to_string(track_id);
          cv::putText(display_img, id_text, cv::Point(x1, y1 - 5),
                     cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        }

        // Display the image
        cv::imshow(window_name, display_img);

        // Wait for a short time to control display speed (1ms)
        // Press 'q' to quit or any other key to continue
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) { // 'q' or ESC to quit
          break;
        }
      }
    }

    out_file.close();
    std::cout << "  Output written to: " << output_file << "\n";

    // Close OpenCV window for this sequence
    if (args.display) {
      cv::destroyWindow(window_name);
    }
  }

  // Clean up OpenCV windows
  if (args.display) {
    cv::destroyAllWindows();
  }

  // Print timing statistics
  std::cout << "\n=== Performance Statistics ===\n";
  std::cout << "Total Tracking took: " << std::fixed << std::setprecision(3)
            << total_time << " seconds for " << total_frames << " frames\n";

  if (total_frames > 0) {
    double fps = total_frames / total_time;
    std::cout << "Average FPS: " << std::fixed << std::setprecision(1) << fps << "\n";
  }

  if (args.display) {
    std::cout << "\nNote: to get real runtime results run without the option: --display\n";
  }

  return 0;
}
