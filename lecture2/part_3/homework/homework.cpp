#include <iostream>
#include <opencv2/opencv.hpp>

// ======================= 作业 =======================
// 1. 读取 ../assets/demo.jpg
// 2. 使用 cvtColor 把图像转为灰度图（颜色空间：BGR2GRAY）
// 3. 使用 imwrite 把灰度图保存为 gray.jpg
// 4. 在灰度图上用 circle 画一个圆，标记你要"瞄准"的位置
// 5. 显示灰度图，按任意键退出
// ====================================================

int main()
{
  //1
  cv::Mat img = cv::imread("assets/demo.jpg");
  if (img.empty()) {
    std::cout << "读取图片失败！请检查路径" << std::endl;
    return -1;
  }
  //2
  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  //3
  cv::imwrite("gray.jpg", gray);
  //4
  cv::Point aim(gray.cols / 2, gray.rows / 2);
  cv::circle(gray, aim, 80, cv::Scalar(255), 3);
  //5
  cv::imshow("gray", gray);
  std::cout << "按任意建关闭窗口..." << std::endl;
  cv::waitKey(0);
  return 0;
}
