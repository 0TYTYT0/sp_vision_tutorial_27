#include <opencv2/opencv.hpp>

struct camera
{
public:
  camera();
  ~camera();
  int read(cv::Mat & img_out);

private:
  void * handle_;
  unsigned int ret_;
};
