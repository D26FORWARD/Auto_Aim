#ifndef AUTO_AIM_PROCESSOR__PROCESSOR_NODE_HPP_
#define AUTO_AIM_PROCESSOR__PROCESSOR_NODE_HPP_

#include "my_detector.hpp"
#include "my_tracker.hpp"

#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/target.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

// 添加了这一块：函数的前向声明
namespace auto_aim_processor
{
  double getYawFromRotationMatrix(const Eigen::Matrix3d & R);
}  // namespace auto_aim_processor


namespace auto_aim_processor
{
  class ProcessorNode : public rclcpp::Node
  {
  public:
    explicit ProcessorNode(const rclcpp::NodeOptions& options);

  private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

    std::unique_ptr<MyDetector> detector_;
    std::unique_ptr<Tracker> tracker_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<auto_aim_interfaces::msg::Target>::SharedPtr target_pub_;
    rclcpp::Publisher<auto_aim_interfaces::msg::Armors>::SharedPtr armors_pub_;
  };

}  // namespace auto_aim_processor

#endif  // AUTO_AIM_PROCESSOR__PROCESSOR_NODE_HPP_
