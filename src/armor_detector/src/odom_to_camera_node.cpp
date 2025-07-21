#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <Eigen/Dense>
#include "auto_aim_interfaces/msg/vision.hpp" // 假设这是Vision消息的头文件
#include <tf2/LinearMath/Matrix3x3.h>

#include <opencv2/core.hpp>

class QuaternionToTFNode : public rclcpp::Node
{
public:
    QuaternionToTFNode() : Node("quaternion_to_tf_node")
    {
        // 创建订阅器
        subscription_ = this->create_subscription<auto_aim_interfaces::msg::Vision>(
                "/Vision_data", rclcpp::SensorDataQoS(),
                std::bind(&QuaternionToTFNode::visionCallback, this, std::placeholders::_1));

        // 创建TF广播器
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    }

private:

    void visionCallback(const auto_aim_interfaces::msg::Vision::ConstSharedPtr& vision_msg)
    {
        // 读取四元数
        float quaternion[4];
        for (size_t i = 0; i < 4; i++)
        {
            quaternion[i] = vision_msg->quaternion[i];
        }

        // 将四元数转换为旋转矩阵
        Eigen::Matrix3d rotationMatrix = quaternionToRotationMatrix(quaternion);

        Eigen::Matrix3d Rz;
        Rz <<  0, -1, 0,
                1, 0, 0,
                0, 0, 1;

        // Should I turn y to x?
        Eigen::Matrix3d rotationMatrix_fix = rotationMatrix * Rz;

        // 将旋转矩阵转换为tf2的四元数
        // tf2::Matrix3x3 tf2_rotation_matrix(
        //     rotationMatrix(0, 0), rotationMatrix(0, 1), rotationMatrix(0, 2),
        //     rotationMatrix(1, 0), rotationMatrix(1, 1), rotationMatrix(1, 2),
        //     rotationMatrix(2, 0), rotationMatrix(2, 1), rotationMatrix(2, 2));

        tf2::Matrix3x3 tf2_rotation_matrix(
                rotationMatrix_fix(0, 0), rotationMatrix_fix(0, 1), rotationMatrix_fix(0, 2),
                rotationMatrix_fix(1, 0), rotationMatrix_fix(1, 1), rotationMatrix_fix(1, 2),
                rotationMatrix_fix(2, 0), rotationMatrix_fix(2, 1), rotationMatrix_fix(2, 2));

        tf2::Quaternion tf2_quaternion;
        tf2_rotation_matrix.getRotation(tf2_quaternion);

        // 创建坐标变换消息
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.stamp = this->get_clock()->now();
        transformStamped.header.frame_id = "odom";
        transformStamped.child_frame_id = "camera_switch";

        // 设置平移量为0
        transformStamped.transform.translation.x = 0.0;
        transformStamped.transform.translation.y = 0.0;
        transformStamped.transform.translation.z = 0.0;

        // 设置四元数
        transformStamped.transform.rotation.x = tf2_quaternion.x();
        transformStamped.transform.rotation.y = tf2_quaternion.y();
        transformStamped.transform.rotation.z = tf2_quaternion.z();
        transformStamped.transform.rotation.w = tf2_quaternion.w();

        // 发布坐标变换
        tf_broadcaster_->sendTransform(transformStamped);
    }


    Eigen::Matrix3d quaternionToRotationMatrix(float quaternion[4]) {
        Eigen::Matrix3d R_x;
        // 四元数
        float w=quaternion[0],x=quaternion[1],y=quaternion[2],z=quaternion[3];
        R_x << 1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w,
                2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w,
                2*x*z-2*y*w, 2*y*z+2*w*x, 1-2*x*x-2*y*y;
        return R_x;
    }

    rclcpp::Subscription<auto_aim_interfaces::msg::Vision>::SharedPtr subscription_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};



int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<QuaternionToTFNode>());
    rclcpp::shutdown();
    return 0;
}