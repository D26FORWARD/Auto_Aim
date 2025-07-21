#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include "auto_aim_interfaces/msg/target.hpp"
#include "auto_aim_interfaces/msg/tracker_info.hpp"
#include "auto_aim_interfaces/msg/robot_ctrl.hpp"
#include "auto_aim_interfaces/msg/vision.hpp"
#include <cmath>

// 使用 M_PI 替换自定义的 PI 定义（可选）
// #define PI 3.141592654

class FireControlNode : public rclcpp::Node
{
public:
    FireControlNode()
        : Node("fire_control_node")
    {

        target_sub_ = this->create_subscription<auto_aim_interfaces::msg::Target>(
            "/tracker/target", 10,
            std::bind(&FireControlNode::target_sub_callback, this, std::placeholders::_1));

        vision_data_sub_ = this->create_subscription<auto_aim_interfaces::msg::Vision>(
            "/Vision_data", 10,
            std::bind(&FireControlNode::vision_callback, this, std::placeholders::_1));

        ctrl_pub_ = this->create_publisher<auto_aim_interfaces::msg::RobotCtrl>(
        "/Robot_ctrl_data", rclcpp::SensorDataQoS());
    }

private:
    rclcpp::Subscription<auto_aim_interfaces::msg::Target>::SharedPtr target_sub_;
    rclcpp::Subscription<auto_aim_interfaces::msg::Vision>::SharedPtr vision_data_sub_;
    rclcpp::Publisher<auto_aim_interfaces::msg::RobotCtrl>::SharedPtr ctrl_pub_;
    double now_pitch_;
    double now_yaw_;

    /**
     *  函数名: Barrel_Solve
     *  传入: Eigen::Vector3d position
     *  传出: Vector3d rpy
     *  功能: 通过传入的坐标计算枪管移动角度
     */
    Eigen::Vector3d Barrel_Solve(Eigen::Vector3d position) {
        Eigen::Vector3d rpy;
        rpy[2] = -atan2(position[0],position[1]) / M_PI*180.0;
        rpy[1] = atan2(position[2],position[1]) / M_PI*180.0;
        rpy[0] = atan2(position[2],position[0]) / M_PI*180.0;

        // 单独计算抬升角度 | atan2()计算角度在车旋转180度之后会解算出[90-180]
        // rpy[1] = airResistanceSolve(position);  // 暂时禁用

        // 添加枪管补偿
        rpy[0] += 0;
        rpy[1] += 0.1;
        rpy[2] += 0;

        return rpy;
    }

    void vision_callback(const auto_aim_interfaces::msg::Vision::ConstSharedPtr& vision_msg)
    {
        now_pitch_ = vision_msg->pitch;  // pitch
        now_yaw_ = vision_msg->yaw;    // yaw
        // std::cout<<"NOW: pitch:"<< now_pitch_ << " yaw: "<< now_yaw_ <<std::endl;
    }

    void target_sub_callback(const auto_aim_interfaces::msg::Target::ConstSharedPtr& target_msg) 
    {       
        RCLCPP_INFO(this->get_logger(), "INTO target_sub_callback");
        std::cout<<"INTO target_sub_callback"<<std::endl;
        auto_aim_interfaces::msg::RobotCtrl Robot_ctrl_t;

        if (target_msg->tracking) {
            double xc = target_msg->position.x, yc = target_msg->position.y, zc = target_msg->position.z;
            size_t a_n = target_msg->armors_num;
            double min_distance = std::numeric_limits<double>::max();
            Eigen::Vector3d closest_armor_xyz;

            // 遍历所有装甲板，找到距离最近的装甲板
            for (size_t i = 0; i < a_n; i++) {
                double yaw = target_msg->yaw + i * (2 * M_PI / a_n);
                double r1 = target_msg->radius_1, r2 = target_msg->radius_2;
                double za = target_msg->position.z;
                double dz = target_msg->dz;
                double r = 0;
                double p_a_z = 0;
                // Only 4 armors has 2 radius and height
                if (a_n == 4) {
                    bool is_current_pair = (i % 2 == 0);
                    r = is_current_pair ? r1 : r2;
                    p_a_z = za + (is_current_pair ? 0 : dz);
                } else {
                    r = r1;
                    p_a_z = za;
                }
                double p_a_x = xc - r * cos(yaw);
                double p_a_y = yc - r * sin(yaw);

                Eigen::Vector3d armor_xyz(p_a_x, p_a_y, p_a_z);
                double distance = (armor_xyz - Eigen::Vector3d(xc, yc, zc)).norm();
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_armor_xyz = armor_xyz;
                }
            }

            // 使用最近装甲板的位置计算枪管角度
            Eigen::Vector3d rpy = Barrel_Solve(closest_armor_xyz);
            Robot_ctrl_t.pitch = rpy[1];
            Robot_ctrl_t.yaw = rpy[2];

            if(abs(rpy[1]-now_pitch_)<=2 && abs(rpy[2]-now_yaw_)<=2){ // 当前枪口pitch yaw和目标值相差2度时，允许开火。后期加入根据装甲板宽度进行角度判定
                Robot_ctrl_t.fire_command = 1;
            }else{
                Robot_ctrl_t.fire_command = 0;
            }

        }else{
            // 如果没有敌人，控制量等于当前量并不开火
            Robot_ctrl_t.pitch = now_pitch_;
            Robot_ctrl_t.yaw = now_yaw_; // 修改为 now_yaw_
            Robot_ctrl_t.fire_command = 0;
        }


        ctrl_pub_->publish(Robot_ctrl_t);

        std::cout<<"CTRL: pitch: "<< Robot_ctrl_t.pitch << " yaw: "<< Robot_ctrl_t.yaw << " fire: "<< Robot_ctrl_t.fire_command << std::endl;

    }

};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FireControlNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}