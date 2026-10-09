#include "io/camera.hpp"
#include "opencv2/opencv.hpp"
#include "tasks/armor.hpp"
#include "tasks/yolo.hpp"
#include "tools/img_tools.hpp"

int main()
{
  camera cam;
  auto_aim::YOLO yolo("./configs/yolo.yaml");
  cv::Mat img;
  std::vector<cv::Point2f> points;
  while (true) {
    cam.read(img);
    if (img.empty()) {
      continue;
    }
    auto armors = yolo.detect(img);
    if (armors.empty()) {
      // continue;
    }

    for (const auto & armor : armors) {
      points = armor.points;

      std::string name = "";
      switch (armor.color) {
        case auto_aim::Color::blue:
          name.append("B ");
          break;
        case auto_aim::Color::extinguish:
          name.append("DEAD ");
          break;
        case auto_aim::Color::red:
          name.append("R ");
          break;
        case auto_aim::Color::purple:
          name.append("P ");
          break;
        default:
          break;
      }
      switch (armor.name) {
        case auto_aim::ArmorName::base:
          name.append("base");
          break;
        case auto_aim::ArmorName::outpost:
          name.append("outpost");
          break;
        case auto_aim::ArmorName::not_armor:
          name.append("not_armor");
          break;
        case auto_aim::ArmorName::sentry:
          name.append("sentry");
          break;
        case auto_aim::ArmorName::one:
        case auto_aim::ArmorName::two:
        case auto_aim::ArmorName::three:
        case auto_aim::ArmorName::four:
        case auto_aim::ArmorName::five: {
          const int num =
            static_cast<int>(armor.name) - static_cast<int>(auto_aim::ArmorName::one) + 1;
          name.append(std::to_string(num));
          break;
        }
        default:
          break;
      }
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
