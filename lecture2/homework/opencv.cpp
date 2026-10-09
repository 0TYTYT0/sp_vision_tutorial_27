#include "opencv2/opencv.hpp"

#include "io/camera.hpp"
#include "tasks/apriltag_detector.hpp"
#include "tasks/yolo.hpp"
#include "tools/img_tools.hpp"

int main()
{
  camera cam;
  auto_charge::AprilTagDetector yolo("./configs/yolo.yaml");
  cv::Mat img;
  std::vector<cv::Point2f> points;
  while (true) {
    cam.read(img);
    if (img.empty()) {
      continue;
    }
    auto tags = yolo.detect(img);
    if (tags.empty()) {
      // continue;
    }
    for (const auto & tag : tags) {
      const auto & points = tag.corners;
      std::string name = std::to_string(tag.id);

      cv::Point2f text_point = points.front();
      tools::draw_points(img, points, cv::Scalar{0, 255, 0}, 10);
      tools::draw_text(img, name, text_point, cv::Scalar{0, 255, 255}, 2, 5);
    }

    cv::resize(img, img, cv::Size(480, 640));
    cv::imshow("img", img);
    if (cv::waitKey(1) == 'q') {
      break;
    }
  }
  return 0;
}