#include "processor_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
// 移除了 #include <rclcpp_components/register_node_macro.hpp>，因为不再需要它
#include <sensor_msgs/image_encodings.hpp>

namespace auto_aim_processor
{

// 函数定义保持不变
double getYawFromRotationMatrix(const Eigen::Matrix3d & R)
{
  return std::atan2(-R(2, 0), R(0, 0));
}

ProcessorNode::ProcessorNode(const rclcpp::NodeOptions & options)
: Node("auto_aim_processor", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting ProcessorNode!");

  // 构造函数的其他部分保持不变...
  detector_ = std::make_unique<MyDetector>("/home/changgeng/Auto_Aim_Developing/Models/2023_4_9_hj_num_1.onnx");
  tracker_ = std::make_unique<Tracker>();

  armors_pub_ = this->create_publisher<auto_aim_interfaces::msg::Armors>("/detector/armors", 10);
  target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Target>("/tracker/target", 10);

  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/image_raw", 10, std::bind(&ProcessorNode::image_callback, this, std::placeholders::_1));
}

void ProcessorNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  // image_callback 的内容保持不变...
  auto start_time = this->now();

  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat debug_frame = cv_ptr->image.clone();

  // 1. 视觉处理管线
  std::vector<Light> lights = detector_->findLights(cv_ptr->image, &debug_frame);
  std::vector<Armor> armors = detector_->matchArmors(lights, -1, &debug_frame);
  detector_->recognizeNumbers(cv_ptr->image, armors, &debug_frame);
  detector_->solvePoses(armors, &debug_frame);

  auto_aim_interfaces::msg::Armors armors_msg;
  for (const auto & armor : armors) {
    auto_aim_interfaces::msg::Armor armor_msg;
    armor_msg.type = armor.Type;
    armor_msg.number = armor.id;
    armors_msg.armors.push_back(armor_msg);
  }
  armors_pub_->publish(armors_msg);

  // 2. 追踪器
  std::vector<TrackerArmor> tracker_armors;
  for (const auto & armor : armors) {
    if (armor.Position.norm() > 0) {
      TrackerArmor ta;
      ta.Type = armor.Type;
      ta.Position = armor.Position;
      ta.RotationMatrix = armor.RotationMatrix;
      ta.Yaw = getYawFromRotationMatrix(armor.RotationMatrix);
      tracker_armors.push_back(ta);
    }
  }

  tracker_->update(tracker_armors, (this->now() - start_time).seconds(), msg->header.stamp.sec);

  // 3. 发布目标
  if (tracker_->isInitialized()) {
    auto_aim_interfaces::msg::Target target_msg;
    target_msg.header.stamp = this->now();
    target_msg.header.frame_id = "odom";
    target_msg.id = "0"; 
    Eigen::Vector3d target_position = tracker_->predictArmorPosition("long", 100);
    target_msg.position.x = target_position.x();
    target_msg.position.y = target_position.y();
    target_msg.position.z = target_position.z();
    target_msg.velocity.x = tracker_->getState()(3);
    target_msg.velocity.y = 0;
    target_msg.velocity.z = tracker_->getState()(4);
    target_msg.yaw = tracker_->getState()(2);
    target_msg.v_yaw = tracker_->getState()(5);
    target_msg.radius_1 = tracker_->getState()(6);
    target_msg.radius_2 = tracker_->getState()(7);
    target_msg.dz = 0;
    target_msg.tracking = true;
    target_pub_->publish(target_msg);
  }
}

}  // namespace auto_aim_processor


// 添加了标准的 main 函数作为程序入口
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto options = rclcpp::NodeOptions();
  rclcpp::spin(std::make_shared<auto_aim_processor::ProcessorNode>(options));
  rclcpp::shutdown();
  return 0;
}

// 移除了下面的宏
// RCLCPP_COMPONENTS_REGISTER_NODE(auto_aim_processor::ProcessorNode)
