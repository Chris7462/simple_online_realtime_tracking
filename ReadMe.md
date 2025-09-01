# Simple Online and Realtime Tracking (SORT)

## Introduction
This repository provides a C++ implementation of the SORT algorithm, described in [this paper](https://arxiv.org/abs/1602.00763), packaged for ROS 2.

## Dependencies
This package depends on the Hungarian algorithm for data association. You’ll need to build [hungarian_cpp](https://github.com/Chris7462/hungarian_cpp) alongside this package.

**Note:** [hungarian_cpp](https://github.com/Chris7462/hungarian_cpp) also supports system-wide installation for non-ROS2 users.

## Building
```bash
cd ${ros2_workspace}/src
git clone https://github.com/Chris7462/hungarian_cpp.git
git clone https://github.com/Chris7462/sort_backend.git
cd ${ros2_workspace}
colcon build --symlink-install --packages-select hungarian sort_backend
```

## Running Unit Tests (optional)
```bash
colcon test --packages-select sort_backend --event-handlers console_direct+
```

## Usage Example
To compare results with the original Python implementation of [sort](https://github.com/abewley/sort):
1. Build with the example target enabled:
    ```bash
    colcon build --symlink-install --packages-select sort_backend --cmake-args -DBUILD_EXAMPLE=ON
    ```
2. Download the [MOT15 dataset](https://motchallenge.net/data/MOT15/).
2. Create a symbolic link to the dataset:
    ```bash
    cd ${ros2_workspace}/build/sort_backend
    ln -s /path/to/MOT15 .
    ```
3. Run the example:
    ```
    ./sort_main
    ```
    This will write tracking results to the `output` folder.
4. To display results with visualization:
    ```
    ./sort_main --display
    ```
