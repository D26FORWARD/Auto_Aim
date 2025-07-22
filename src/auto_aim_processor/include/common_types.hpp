#ifndef AUTO_AIM_PROCESSOR__COMMON_TYPES_HPP_
#define AUTO_AIM_PROCESSOR__COMMON_TYPES_HPP_

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <deque>

namespace auto_aim_processor
{

// =================================================================================================
// 核心数据结构 (Core Data Structures)
// =================================================================================================
/**
 * @brief 表示一个检测到的灯条
 */
struct Light
{
    cv::Point2f orientationVector;           // 灯条主方向的归一化向量
    cv::RotatedRect rotatedRect;             // 灯条的旋转矩形
    cv::Point2f top;                         // 灯条顶点中心
    cv::Point2f bottom;                      // 灯条底点中心
    std::vector<cv::Point2f> boundingPoints; // 灯条的四个角点 (有序)
    char color;                              // 灯条颜色 ('R' 或 'B')
};

/**
 * @brief 表示一个匹配到的装甲板
 */
struct Armor
{
    int id = 0;                                  // 装甲板ID (通过数字识别, 0通常为背景或未识别)
    char color = 'N';                            // 颜色 'R', 'B', 'N' (None)
    int strategy = 0;                            // --- 新增 ---: 0:线性估算, 1:复杂补偿, 2:直接连接
    cv::Point2f center;                          // 装甲板中心点
    double radius;                               // 装甲板半径 (用于粗略表示大小)
    std::vector<cv::Point2f> numberROIPoints;    // 用于数字识别的ROI区域四个角点
    std::vector<cv::Point2f> pnpPoints;          // 用于PnP解算的角点 (8个点)
    Eigen::Vector3d Position;                    // PnP解算出的位置向量 (tvec)
    Eigen::Matrix3d RotationMatrix;              // PnP解算出的旋转矩阵 (R)
    std::deque<double> YawHistory;               // 历史存储Yaw角度记录
    std::deque<Eigen::Vector2d> PositionHistory; // 历史存储的装甲板水平面(X,Z)三维位置记录
    std::string Type;
    double Yaw;
};


/**
 * @brief 为特定车辆ID存储已确认的装甲板类型和最后已知位置
 */
struct ArmorTypeTracker
{
    std::string higher_type = "unknown"; // 物理位置较高的装甲板类型
    std::string lower_type = "unknown"; // 物理位置较低的装甲板类型

    // 新增：位置记忆锚点
    float last_higher_x = -1.0f; // 上次观察到较高装甲板的y坐标
    float last_lower_x = -1.0f; // 上次观察到较低装甲板的y坐标

    bool confirmed = false;      // 标记类型和位置是否已被确认
};

// 为追踪器定义一个独立的Armor结构，避免与主流程冲突
struct TrackerArmor
{
    std::string Type;
    double Yaw;
    Eigen::Vector3d Position;
    Eigen::Matrix3d RotationMatrix;
};

struct DebugData
{
    int frame_id;
    Eigen::VectorXd state_begin;
    Eigen::VectorXd state_predicted;
    Eigen::VectorXd state_final;
    std::vector<TrackerArmor> observations;
    std::vector<std::pair<std::string, Eigen::Vector3d>> hypotheses;
    bool match_success = false;
    std::string match_type;
    std::string match_details;
    double match_score = 0.0;
    Eigen::Vector3d final_observation;
    Eigen::Vector2d pos_innovation;
    double yaw_innovation = 0.0;
    std::string decision_pos;
    std::string decision_yaw;
    double calculated_vyaw = 0.0;
};

}  // namespace auto_aim_processor

#endif  // AUTO_AIM_PROCESSOR__COMMON_TYPES_HPP_
