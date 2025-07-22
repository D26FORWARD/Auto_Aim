#ifndef AUTO_AIM_PROCESSOR__MY_DETECTOR_HPP_
#define AUTO_AIM_PROCESSOR__MY_DETECTOR_HPP_

#include "common_types.hpp"

#include <opencv2/dnn.hpp>

namespace auto_aim_processor
{

  class MyDetector
  {
  public:
    MyDetector(const std::string& model_path);

    std::vector<Light> findLights(const cv::Mat& frame, cv::Mat* debugFrame = nullptr);
    std::vector<Armor> matchArmors(
      const std::vector<Light>& lights, int activeStrategy,
      cv::Mat* debugFrame = nullptr);
    void recognizeNumbers(
      const cv::Mat& frame, std::vector<Armor>& armors,
      cv::Mat* debugFrame = nullptr);
    void stabilizeArmorTypes(
      std::vector<Armor>& armors,
      std::map<int, ArmorTypeTracker>& type_tracker,
      cv::Mat* debugFrame = nullptr);
    std::vector<Armor> selectVehicle(const std::vector<Armor>& armors);
    void solvePoses(std::vector<Armor>& armors, cv::Mat* debugFrame = nullptr);

  private:
    char determineColor(const cv::Mat& roi);

    cv::dnn::Net number_recognition_net_;
  };

}  // namespace auto_aim_processor

#endif  // AUTO_AIM_PROCESSOR__MY_DETECTOR_HPP_
