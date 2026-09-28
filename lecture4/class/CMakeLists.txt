cmake_minimum_required(VERSION 3.16.3)

project(lecture4_yolo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
# lecture2 的 CMakeLists 把这一行注释掉了，但 tools/logger.cpp 真的用了 spdlog 的编译期 sink，
# 所以必须显式打开，否则链接不过。
find_package(spdlog REQUIRED)

# OpenVINO 路径只在顶层定义一次，子目录不再各自硬编码
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")
find_package(OpenVINO REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

# tools 要在 io 之前，io 链接 tools
add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks)

# tools/logger.cpp 会往 logs/ 写文件，spdlog 在目录不存在时会抛异常。
# 在配置期就把源码树里的 logs/ 建好（程序约定从本目录运行）。
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/logs)

add_executable(main src/main.cpp)
target_link_libraries(main
    ${OpenCV_LIBS} fmt::fmt yaml-cpp spdlog::spdlog tools io auto_aim)

# 参考实现默认不编译：answer/main.cpp 里就是填好的 Task 01~05，
# 学生敲一次 cmake --build 就拿到答案，整段练习会作废。
# 教师演示时用 -DBUILD_ANSWER=ON 打开。
option(BUILD_ANSWER "Build the reference answer executable (teacher only)" OFF)
if(BUILD_ANSWER)
  add_executable(answer answer/main.cpp)
  target_link_libraries(answer
      ${OpenCV_LIBS} fmt::fmt yaml-cpp spdlog::spdlog tools io auto_aim)
endif()
