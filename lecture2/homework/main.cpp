#include "io/camera.hpp"
#include "opencv2/opencv.hpp"
#include "tasks/armor.hpp"
#include "tasks/yolo.hpp"
#include "tools/img_tools.hpp"

int main()
{
  // 初始化相机、yolo类

  // while (1) {
  // 调用相机读取图像

  // 调用yolo识别装甲板

  // 显示图像
  // cv::resize(img, img , cv::Size(640, 480));
  // cv::imshow("img", img);
  // if (cv::waitKey(0) == 'q') {
  //     // break;
  // }
  // }
  camera cam;
  auto_aim::YOLO yolo("./configs/yolo.yaml");
  cv::Mat img;
  std::vector<cv::Point2f> points;
  while (true) {
    if (cam.read(img) == 0) {
      auto armors = yolo.detect(img);
      auto armor1 = armors.front();
      points = armor1.points;

      std::string name = "";
      switch (armor1.color) {
        case auto_aim::Color::blue:
          name.append("B");
          break;
        case auto_aim::Color::extinguish:
          name.append("E");
          break;
        case auto_aim::Color::red:
          name.append("R");
          break;
        case auto_aim::Color::purple:
          name.append("P");
          break;
        default:
          break;
      }
      switch (armor1.name) {
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
            static_cast<int>(armor1.name) - static_cast<int>(auto_aim::ArmorName::one) + 1;
          name.append(std::to_string(num));
          break;
        }
        default:
          break;
      }
      cv::Point2f text_point = points.front();
      tools::draw_points(img, points, cv::Scalar{0, 0, 255}, 10);
      tools::draw_text(img, name, text_point, cv::Scalar{0, 255, 255}, 10, 10);
      cv::resize(img, img, cv::Size(640, 480));
      cv::imshow("img", img);
      if (cv::waitKey(0) == 'q') {
        break;
      };
    }
    return 0;
  }
}
