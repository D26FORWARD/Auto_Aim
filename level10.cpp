#include <Eigen/Dense>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <numeric> // std::accumulate
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <string>
#include <vector>
#include"../utils/json.hpp"
#include <fstream> // 用于文件操作

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

// =================================================================================================
// 2. 追踪器模块 (Tracker Module - Integrated)
// =================================================================================================

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

// 调试器类，用于集中处理所有日志打印
class Debugger
{
public:
    Debugger(bool perf_on, bool raw_on) : performance_mode(perf_on), raw_mode(raw_on) {}

    // 函数 a: 原始信息记录
    void log_raw_info(const DebugData& data);
    // 函数 b: 性能分析摘要
    void log_performance_summary(const DebugData& data);

private:
    bool performance_mode;
    bool raw_mode;
};


// =================================================================================================
// 3. 最终稳定版自适应Alpha-Beta追踪器 (带假设检验)
// =================================================================================================

// 自适应Alpha-Beta追踪器
class Tracker
{
public:
    Tracker(bool performance_debug = false, bool raw_debug = false);

    void update(const std::vector<TrackerArmor>& armors, double dt, int frame_id);
    Eigen::VectorXd getState() const;
    Eigen::Vector3d predictArmorPosition(const std::string& armor_type, double prediction_time_ms) const;
    bool isInitialized() const { return is_initialized_; }
    double getLastObservedY() const { return last_observed_y_; }

private:
    enum ArmorTypeID { MAIN_LONG, MAIN_SHORT, SYM_LONG, SYM_SHORT };
    struct ArmorHypothesis
    {
        Eigen::Vector3d predicted_pos_yaw;
        ArmorTypeID id;
        std::string name;
    };

    void initialize(const std::vector<TrackerArmor>& armors, DebugData& debug_data);
    void predict(double dt);
    void update_filters(const Eigen::Vector3d& observation, double dt, DebugData& debug_data);
    bool process_observation(const std::vector<TrackerArmor>& armors, Eigen::Vector3d& observation, DebugData& debug_data);
    std::vector<ArmorHypothesis> generate_hypotheses() const;
    Eigen::Vector3d back_calculate_center(const Eigen::Vector2d& p_armor, double yaw_armor, ArmorTypeID armor_id) const;
    double normalizeAngle(double angle) const;
    void clampRadii();
    void swapRadiiIfNeeded();

    bool is_initialized_;
    Eigen::VectorXd x_;
    double alpha_pos_smooth_, beta_pos_smooth_;
    double alpha_yaw_smooth_, beta_yaw_smooth_;
    double alpha_pos_maneuver_, beta_pos_maneuver_;
    double alpha_yaw_maneuver_, beta_yaw_maneuver_;
    const double MANEUVER_THRESHOLD_POS = 0.1;
    const double MANEUVER_THRESHOLD_YAW = 0.3;
    const double MATCH_SCORE_THRESHOLD = 0.4;
    const double YAW_WEIGHT = 0.5;
    double alpha_r_;
    const double MIN_RADIUS = 0.15;
    const double MAX_RADIUS = 0.35;
    ArmorTypeID last_seen_id_;
    int frames_since_last_update_;
    double last_observed_y_;
    bool has_last_observation_;
    double last_observation_yaw_;
    Debugger debugger_;
};


Tracker::Tracker(bool performance_debug, bool raw_debug) :
    is_initialized_(false),
    x_(Eigen::VectorXd::Zero(8)),
    last_seen_id_(MAIN_LONG),
    frames_since_last_update_(0),
    last_observed_y_(0.0),
    has_last_observation_(false),
    last_observation_yaw_(0.0),
    debugger_(performance_debug, raw_debug) // 初始化调试器
{
    // 平滑模式 (更相信模型，用于稳定追踪)
    alpha_pos_smooth_ = 0.4;
    beta_pos_smooth_ = 0.05;
    alpha_yaw_smooth_ = 0.4;
    beta_yaw_smooth_ = 0.05;

    // 机动模式 (更相信观测，用于快速响应)
    alpha_pos_maneuver_ = 0.8;
    beta_pos_maneuver_ = 0.2;
    alpha_yaw_maneuver_ = 0.8;
    beta_yaw_maneuver_ = 0.2;

    alpha_r_ = 0.1;
}

Eigen::VectorXd Tracker::getState() const
{
    return x_;
}

void Tracker::update(const std::vector<TrackerArmor>& armors, double dt, int frame_id)
{
    if (dt <= 0) return;

    DebugData debug_data; // 创建数据托盘
    debug_data.frame_id = frame_id;

    if (!is_initialized_)
    {
        if (armors.size() >= 2)
        {
            initialize(armors, debug_data);
        }
        return;
    }

    debug_data.state_begin = x_;

    predict(dt);
    debug_data.state_predicted = x_;

    Eigen::Vector3d observation;
    if (process_observation(armors, observation, debug_data))
    {
        update_filters(observation, dt, debug_data);
        frames_since_last_update_ = 0;
    }
    else
    {
        frames_since_last_update_++;
        has_last_observation_ = false;
    }

    debug_data.state_final = x_;
    debugger_.log_raw_info(debug_data);
    debugger_.log_performance_summary(debug_data);
}

void Tracker::initialize(const std::vector<TrackerArmor>& armors, DebugData& debug_data)
{
    const TrackerArmor* long_armor = nullptr;
    const TrackerArmor* short_armor = nullptr;

    for (const auto& armor : armors)
    {
        if (armor.Type == "long") long_armor = &armor;
        else if (armor.Type == "short") short_armor = &armor;
    }

    if (!long_armor || !short_armor)
    {
        return;
    }

    double center_yaw_est = normalizeAngle(long_armor->Yaw);
    Eigen::Vector3d center1 = back_calculate_center(Eigen::Vector2d(long_armor->Position.x(), long_armor->Position.z()), center_yaw_est, MAIN_LONG);
    double expected_short_yaw = normalizeAngle(center_yaw_est + M_PI / 2.0);
    Eigen::Vector3d center2 = back_calculate_center(Eigen::Vector2d(short_armor->Position.x(), short_armor->Position.z()), expected_short_yaw, MAIN_SHORT);
    Eigen::Vector2d final_center_pos = (center1.head<2>() + center2.head<2>()) / 2.0;

    x_.setZero();
    x_(0) = final_center_pos(0);
    x_(1) = final_center_pos(1);
    x_(2) = center_yaw_est;

    double r1_obs = (Eigen::Vector2d(long_armor->Position.x(), long_armor->Position.z()) - final_center_pos).norm();
    double r2_obs = (Eigen::Vector2d(short_armor->Position.x(), short_armor->Position.z()) - final_center_pos).norm();
    x_(6) = r1_obs;
    x_(7) = r2_obs;

    swapRadiiIfNeeded();
    clampRadii();

    is_initialized_ = true;
    last_observed_y_ = (long_armor->Position.y() + short_armor->Position.y()) / 2.0;

    last_observation_yaw_ = x_(2);
    has_last_observation_ = true;

    // 【已修正】为调试日志正确赋值
    debug_data.state_begin = Eigen::VectorXd::Zero(8);
    debug_data.state_predicted = Eigen::VectorXd::Zero(8);
    debug_data.state_final = x_;
    debugger_.log_raw_info(debug_data);
    debugger_.log_performance_summary(debug_data);
}

void Tracker::predict(double dt)
{
    x_(0) += x_(3) * dt;
    x_(1) += x_(4) * dt;
    x_(2) = normalizeAngle(x_(2) + x_(5) * dt);
}

void Tracker::update_filters(const Eigen::Vector3d& observation, double dt, DebugData& debug_data)
{
    double current_alpha_pos, current_beta_pos;
    double current_alpha_yaw;

    Eigen::Vector2d pos_innovation = observation.head<2>() - x_.head<2>();
    double yaw_innovation = normalizeAngle(observation(2) - x_(2));
    debug_data.pos_innovation = pos_innovation;
    debug_data.yaw_innovation = yaw_innovation;

    if (pos_innovation.norm() > MANEUVER_THRESHOLD_POS)
    {
        current_alpha_pos = alpha_pos_maneuver_;
        current_beta_pos = beta_pos_maneuver_;
        debug_data.decision_pos = "机动模式";
    }
    else
    {
        current_alpha_pos = alpha_pos_smooth_;
        current_beta_pos = beta_pos_smooth_;
        debug_data.decision_pos = "平滑模式";
    }

    if (std::abs(yaw_innovation) > MANEUVER_THRESHOLD_YAW)
    {
        current_alpha_yaw = alpha_yaw_maneuver_;
        debug_data.decision_yaw = "机动模式";
    }
    else
    {
        current_alpha_yaw = alpha_yaw_smooth_;
        debug_data.decision_yaw = "平滑模式";
    }

    x_.head<2>() += current_alpha_pos * pos_innovation;
    x_.segment<2>(3) += (current_beta_pos / dt) * pos_innovation;
    x_(2) = normalizeAngle(x_(2) + current_alpha_yaw * yaw_innovation);

    if (has_last_observation_)
    {
        double vyaw = normalizeAngle(observation(2) - last_observation_yaw_) / dt;
        x_(5) = 0.7 * x_(5) + 0.3 * vyaw;
        debug_data.calculated_vyaw = vyaw;
    }
    else
    {
        x_(5) = 0;
    }

    last_observation_yaw_ = observation(2);
    has_last_observation_ = true;
}

bool Tracker::process_observation(const std::vector<TrackerArmor>& armors, Eigen::Vector3d& observation, DebugData& debug_data)
{
    debug_data.observations = armors;
    if (armors.empty())
    {
        debug_data.match_success = false;
        return false;
    }

    last_observed_y_ = armors[0].Position.y();
    auto hypotheses = generate_hypotheses();
    for (const auto& h : hypotheses)
    {
        debug_data.hypotheses.push_back({ h.name, h.predicted_pos_yaw });
    }

    const TrackerArmor* long_armor = nullptr;
    const TrackerArmor* short_armor = nullptr;

    if (armors.size() >= 2)
    {
        for (const auto& armor : armors)
        {
            if (armor.Type == "long" && long_armor == nullptr) long_armor = &armor;
            if (armor.Type == "short" && short_armor == nullptr) short_armor = &armor;
        }
    }

    if (long_armor && short_armor)
    {
        debug_data.match_type = "双板匹配";
        double best_score = 1e9;
        ArmorHypothesis best_long_hypo, best_short_hypo;

        for (const auto& h_long : hypotheses)
        {
            if (h_long.id != MAIN_LONG && h_long.id != SYM_LONG) continue;
            for (const auto& h_short : hypotheses)
            {
                if (h_short.id != MAIN_SHORT && h_short.id != SYM_SHORT) continue;
                double yaw_error = std::abs(normalizeAngle(long_armor->Yaw - h_long.predicted_pos_yaw(2))) +
                    std::abs(normalizeAngle(short_armor->Yaw - h_short.predicted_pos_yaw(2)));
                double pos_error = (Eigen::Vector2d(long_armor->Position.x(), long_armor->Position.z()) - h_long.predicted_pos_yaw.head<2>()).norm() +
                    (Eigen::Vector2d(short_armor->Position.x(), short_armor->Position.z()) - h_short.predicted_pos_yaw.head<2>()).norm();
                double score = pos_error + YAW_WEIGHT * yaw_error;
                if (score < best_score)
                {
                    best_score = score;
                    best_long_hypo = h_long;
                    best_short_hypo = h_short;
                }
            }
        }

        debug_data.match_success = true;
        std::stringstream ss;
        ss << "观测 Long (Yaw: " << long_armor->Yaw * 180.0 / M_PI << "°) -> " << best_long_hypo.name << "\n"
            << "    观测 Short (Yaw: " << short_armor->Yaw * 180.0 / M_PI << "°) -> " << best_short_hypo.name;
        debug_data.match_details = ss.str();
        debug_data.match_score = best_score;

        Eigen::Vector3d center_obs1 = back_calculate_center(Eigen::Vector2d(long_armor->Position.x(), long_armor->Position.z()), long_armor->Yaw, best_long_hypo.id);
        Eigen::Vector3d center_obs2 = back_calculate_center(Eigen::Vector2d(short_armor->Position.x(), short_armor->Position.z()), short_armor->Yaw, best_short_hypo.id);
        observation = (center_obs1 + center_obs2) / 2.0;
        double s = (sin(center_obs1(2)) + sin(center_obs2(2))) / 2.0;
        double c = (cos(center_obs1(2)) + cos(center_obs2(2))) / 2.0;
        observation(2) = atan2(s, c);

        Eigen::Vector2d center_pos_obs = observation.head<2>();
        double r1_obs = (Eigen::Vector2d(long_armor->Position.x(), long_armor->Position.z()) - center_pos_obs).norm();
        double r2_obs = (Eigen::Vector2d(short_armor->Position.x(), short_armor->Position.z()) - center_pos_obs).norm();
        x_(6) = (1 - alpha_r_) * x_(6) + alpha_r_ * r1_obs;
        x_(7) = (1 - alpha_r_) * x_(7) + alpha_r_ * r2_obs;

        swapRadiiIfNeeded();
        clampRadii();
    }
    else
    {
        const auto& armor = armors[0];
        debug_data.match_type = "单板匹配";
        double best_score = 1e9;
        ArmorHypothesis best_match;

        for (const auto& hypo : hypotheses)
        {
            bool is_hypo_long = hypo.id == MAIN_LONG || hypo.id == SYM_LONG;
            if ((armor.Type == "long") != is_hypo_long) continue;
            double pos_error = (Eigen::Vector2d(armor.Position.x(), armor.Position.z()) - hypo.predicted_pos_yaw.head<2>()).norm();
            double yaw_error = std::abs(normalizeAngle(armor.Yaw - hypo.predicted_pos_yaw(2)));
            double score = pos_error + YAW_WEIGHT * yaw_error;
            if (score < best_score)
            {
                best_score = score;
                best_match = hypo;
            }
        }

        debug_data.match_success = true;
        std::stringstream ss;
        ss << "观测 " << armor.Type << " (Yaw: " << armor.Yaw * 180.0 / M_PI << "°) -> " << best_match.name;
        debug_data.match_details = ss.str();
        debug_data.match_score = best_score;

        observation = back_calculate_center(Eigen::Vector2d(armor.Position.x(), armor.Position.z()), armor.Yaw, best_match.id);
        last_seen_id_ = best_match.id;
    }

    debug_data.final_observation = observation;
    return true;
}

std::vector<Tracker::ArmorHypothesis> Tracker::generate_hypotheses() const
{
    std::vector<ArmorHypothesis> hypotheses(4);
    const double center_x = x_(0), center_z = x_(1), center_yaw = x_(2);
    const double r1 = x_(6), r2 = x_(7);

    hypotheses[MAIN_LONG] = { {center_x + r1 * cos(center_yaw), center_z - r1 * sin(center_yaw), normalizeAngle(center_yaw)}, MAIN_LONG, "主长轴" };
    hypotheses[SYM_LONG] = { {center_x - r1 * cos(center_yaw), center_z + r1 * sin(center_yaw), normalizeAngle(center_yaw - M_PI)}, SYM_LONG, "对称长轴" };
    hypotheses[MAIN_SHORT] = { {center_x + r2 * sin(center_yaw), center_z + r2 * cos(center_yaw), normalizeAngle(center_yaw + M_PI / 2.0)}, MAIN_SHORT, "主短轴" };
    hypotheses[SYM_SHORT] = { {center_x - r2 * sin(center_yaw), center_z - r2 * cos(center_yaw), normalizeAngle(center_yaw - M_PI / 2.0)}, SYM_SHORT, "对称短轴" };

    return hypotheses;
}

Eigen::Vector3d Tracker::back_calculate_center(const Eigen::Vector2d& p_armor, double yaw_armor, ArmorTypeID armor_id) const
{
    // Eigen::Vector3d center_obs;
    // double center_yaw_obs;
    // double r;
    // bool use_estimated_radii = is_initialized_;

    // switch (armor_id)
    // {
    // case MAIN_LONG:  center_yaw_obs = normalizeAngle(yaw_armor);
    //     r = use_estimated_radii ? x_(6) : 0.30; break;
    // case SYM_LONG:   center_yaw_obs = normalizeAngle(yaw_armor + M_PI);
    //     r = use_estimated_radii ? x_(6) : 0.30; break;
    // case MAIN_SHORT: center_yaw_obs = normalizeAngle(yaw_armor - M_PI / 2.0);
    //     r = use_estimated_radii ? x_(7) : 0.23; break;
    // case SYM_SHORT:  center_yaw_obs = normalizeAngle(yaw_armor + M_PI / 2.0);
    //     r = use_estimated_radii ? x_(7) : 0.23; break;
    // }

    // double center_x_obs = p_armor.x();
    // double center_z_obs = p_armor.y();

    // switch (armor_id)
    // {
    // case MAIN_LONG:  center_x_obs -= r * std::cos(center_yaw_obs);
    //     center_z_obs += r * std::sin(center_yaw_obs); break;
    // case SYM_LONG:   center_x_obs += r * std::cos(center_yaw_obs);
    //     center_z_obs -= r * std::sin(center_yaw_obs); break;
    // case MAIN_SHORT: center_x_obs -= r * std::sin(center_yaw_obs);
    //     center_z_obs -= r * std::cos(center_yaw_obs); break;
    // case SYM_SHORT:  center_x_obs += r * std::sin(center_yaw_obs);
    //     center_z_obs += r * std::cos(center_yaw_obs); break;
    // }

    // center_obs << center_x_obs, center_z_obs, center_yaw_obs;
    // return center_obs;
    Eigen::Vector3d center_obs;
    double center_yaw_obs;
    double r;
    bool use_estimated_radii = is_initialized_;

    switch (armor_id)
    {
    case MAIN_LONG:
        // center_yaw_obs = normalizeAngle(yaw_armor);
        r = use_estimated_radii ? x_(6) : 0.30;
        break;
    case SYM_LONG:
        // center_yaw_obs = normalizeAngle(yaw_armor + M_PI);
        r = use_estimated_radii ? x_(6) : 0.30;
        break;
    case MAIN_SHORT:
        // center_yaw_obs = normalizeAngle(yaw_armor - M_PI / 2.0);
        r = use_estimated_radii ? x_(7) : 0.23;
        break;
    case SYM_SHORT:
        // center_yaw_obs = normalizeAngle(yaw_armor + M_PI / 2.0);
        r = use_estimated_radii ? x_(7) : 0.23;
        break;
    }

    double center_x_obs = p_armor.x();
    double center_z_obs = p_armor.y();
    center_yaw_obs = yaw_armor;
    center_x_obs += r * std::sin(center_yaw_obs);
    center_z_obs += r * std::cos(center_yaw_obs);
    // switch (armor_id)
    // {
    // case MAIN_LONG:  center_x_obs -= r * std::cos(center_yaw_obs); center_z_obs += r * std::sin(center_yaw_obs); break;
    // case SYM_LONG:   center_x_obs += r * std::cos(center_yaw_obs); center_z_obs -= r * std::sin(center_yaw_obs); break;
    // case MAIN_SHORT: center_x_obs -= r * std::sin(center_yaw_obs); center_z_obs -= r * std::cos(center_yaw_obs); break;
    // case SYM_SHORT:  center_x_obs += r * std::sin(center_yaw_obs); center_z_obs += r * std::cos(center_yaw_obs); break;
    // }

    center_obs << center_x_obs, center_z_obs, center_yaw_obs;
    return center_obs;
}

Eigen::Vector3d Tracker::predictArmorPosition(const std::string& armor_type, double prediction_time_ms) const
{
    if (!is_initialized_) return Eigen::Vector3d::Zero();

    double dt = prediction_time_ms / 1000.0;

    Eigen::VectorXd future_status = x_;
    future_status(0) += x_(3) * dt;
    future_status(1) += x_(4) * dt;
    future_status(2) = normalizeAngle(x_(2) + x_(5) * dt);

    const double center_x = future_status(0), center_z = future_status(1), center_yaw = future_status(2);
    double armor_x, armor_z;

    if (armor_type == "long")
    {
        const double r = future_status(6);
        armor_x = center_x + r * std::cos(center_yaw);
        armor_z = center_z - r * std::sin(center_yaw);
    }
    else
    { // "short"
        const double r = future_status(7);
        armor_x = center_x + r * std::sin(center_yaw);
        armor_z = center_z + r * std::cos(center_yaw);
    }

    return Eigen::Vector3d(armor_x, last_observed_y_, armor_z);
}

double Tracker::normalizeAngle(double angle) const
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

void Tracker::clampRadii()
{
    x_(6) = std::max(MIN_RADIUS, std::min(MAX_RADIUS, x_(6)));
    x_(7) = std::max(MIN_RADIUS, std::min(MAX_RADIUS, x_(7)));
}

void Tracker::swapRadiiIfNeeded()
{
    if (x_(6) < x_(7))
    {
        std::swap(x_(6), x_(7));
    }
}

// =================================================================================================
// 4. 调试函数实现 (Debugger Function Implementations)
// =================================================================================================

void Debugger::log_raw_info(const DebugData& data)
{
    if (!raw_mode) return;

    std::cout << "\n\n\n################################################################################";
    std::cout << "\n#############################  帧 " << data.frame_id << " 开始 (原始日志) #################################";
    std::cout << "\n################################################################################";

    // 1. 帧开始状态
    std::cout << "\n[步骤 1] 帧开始状态" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    if (data.state_begin.size() > 0)
    { // Add check to prevent crash
        std::cout << "  - 状态向量: " << data.state_begin.transpose() << std::endl;
        std::cout << "  - 姿态角 Yaw (角度): " << data.state_begin(2) * 180.0 / M_PI << "°" << std::endl;
    }
    else
    {
        std::cout << "  - (状态未初始化)" << std::endl;
    }


    // 2. 预测
    std::cout << "\n[步骤 2] 状态预测" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    if (data.state_predicted.size() > 0)
    { // Add check to prevent crash
        std::cout << "  - 预测后状态向量: " << data.state_predicted.transpose() << std::endl;
        std::cout << "  - 预测后姿态角 Yaw (角度): " << data.state_predicted(2) * 180.0 / M_PI << "°" << std::endl;
    }
    else
    {
        std::cout << "  - (状态未初始化)" << std::endl;
    }


    // 3. 观测
    std::cout << "\n[步骤 3] 观测处理" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    if (data.observations.empty())
    {
        std::cout << "  - 接收到 0 个装甲板。" << std::endl;
    }
    else
    {
        std::cout << "  - 接收到 " << data.observations.size() << " 个装甲板:" << std::endl;
        for (size_t i = 0; i < data.observations.size(); ++i)
        {
            std::cout << "    - 装甲板 " << i + 1 << " (" << data.observations[i].Type << "): 位置=["
                << data.observations[i].Position.transpose() << "], 偏航角(弧度)=" << data.observations[i].Yaw << " (角度): " << data.observations[i].Yaw * 180.0 / M_PI << "°" << std::endl;
        }
    }

    // 4. 匹配
    std::cout << "\n[步骤 4] 数据关联与匹配" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    if (data.match_success)
    {
        std::cout << data.match_details << std::endl;
        std::cout << "  - 最终生成高质量观测值: 位置=[" << data.final_observation.head<2>().transpose()
            << "], 偏航角(弧度)=" << data.final_observation(2) << " (角度): " << data.final_observation(2) * 180.0 / M_PI << "°" << std::endl;
    }
    else
    {
        std::cout << "  - 匹配失败或无观测。" << std::endl;
    }

    // 5. 滤波
    std::cout << "\n[步骤 5] 滤波器更新" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    if (data.match_success)
    {
        std::cout << "  - 新息 (观测 - 预测):" << std::endl;
        std::cout << "    - 位置误差: " << data.pos_innovation.norm() << " m" << std::endl;
        std::cout << "    - 偏航角误差: " << data.yaw_innovation * 180.0 / M_PI << "° (" << data.yaw_innovation << " rad)" << std::endl;
        std::cout << "  - 决策: 位置 " << data.decision_pos << ", 偏航角 " << data.decision_yaw << std::endl;
        std::cout << "  - 角速度计算: 基于连续观测，vyaw = " << data.calculated_vyaw * 180.0 / M_PI << " °/s" << std::endl;
    }
    else
    {
        std::cout << "  - (跳过更新)" << std::endl;
    }

    // 6. 帧结束
    std::cout << "\n[步骤 6] 帧结束状态" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "  - 最终状态向量: " << data.state_final.transpose() << std::endl;
    std::cout << "  - 最终姿态角 Yaw (角度): " << data.state_final(2) * 180.0 / M_PI << "°" << std::endl;
}

void Debugger::log_performance_summary(const DebugData& data)
{
    if (!performance_mode) return;

    std::cout << "\n\n--- [性能摘要] 帧: " << data.frame_id << " ---" << std::endl;

    if (!data.match_success)
    {
        if (data.observations.empty())
        {
            std::cout << "| 状态: 跟丢 (无观测)" << std::endl;
        }
        else
        {
              // This case is now handled by the main logic, so it's less likely to be "match failure"
            std::cout << "| 状态: 跟丢 (无有效组合)" << std::endl;
        }
        if (data.state_final.size() > 0)
        {
            std::cout << "| 最终状态 (预测): "
                << "x=" << data.state_final(0) << ", z=" << data.state_final(1) << ", yaw=" << data.state_final(2) * 180.0 / M_PI << "°"
                << ", vx=" << data.state_final(3) << ", vz=" << data.state_final(4) << ", vyaw=" << data.state_final(5) * 180.0 / M_PI << "°/s"
                << std::endl;
        }
        return;
    }

    std::cout << "| 匹配决策: " << data.match_details << " (Score: " << data.match_score << ")" << std::endl;
    std::cout << "|---------------------------------|-----------------------------------|" << std::endl;
    std::cout << "|           指标 (单位)           |             数值                  |" << std::endl;
    std::cout << "|---------------------------------|-----------------------------------|" << std::endl;
    std::cout << "| 位置预测误差 (m)                | " << std::setw(25) << data.pos_innovation.norm() << " |" << std::endl;
    std::cout << "| 角度预测误差 (°)                | " << std::setw(25) << data.yaw_innovation * 180.0 / M_PI << " |" << std::endl;
    std::cout << "|---------------------------------|-----------------------------------|" << std::endl;
    std::cout << "| 瞬时角速度计算 (°/s)            | " << std::setw(25) << data.calculated_vyaw * 180.0 / M_PI << " |" << std::endl;
    std::cout << "| 平滑后角速度估计 (°/s)          | " << std::setw(25) << data.state_final(5) * 180.0 / M_PI << " |" << std::endl;
    std::cout << "|---------------------------------|-----------------------------------|" << std::endl;
    std::cout << "| 最终位置估计 (x, z)             | " << data.state_final(0) << ", " << data.state_final(1) << std::endl;
    std::cout << "| 最终姿态估计 (°)                | " << std::setw(25) << data.state_final(2) * 180.0 / M_PI << " |" << std::endl;
    std::cout << "| 最终线速度估计 (vx, vz)         | " << data.state_final(3) << ", " << data.state_final(4) << std::endl;
    std::cout << "|---------------------------------|-----------------------------------|" << std::endl;

}


// =================================================================================================
// 5. 辅助函数与主程序 (Helpers and Main Test Program)
// =================================================================================================

double getYawFromRotationMatrix(const Eigen::Matrix3d& R)
{
    return std::atan2(-R(2, 0), R(0, 0));
}


// 为了方便使用，声明 nlohmann::json 的命名空间
using json = nlohmann::json;
//======================================
// 工具
//======================================

/**
 * @brief (方案一) 将一帧数据作为一个对象添加到根JSON数组中
 * @param root_json_array 根JSON数组 (通过引用传递)
 * @param frame_number 当前的帧号
 * @param armors 当前帧检测到的装甲板向量
 */
void logFrameDataToArray(
    json& root_json_array,
    int frame_number,
    const std::vector<Armor>& armors)
{
    // 如果当前帧没有装甲板，可以选择跳过或记录一个空帧
    if (armors.empty())
    {
// return; // 如果不想记录空帧，可以取消此行注释
    }

    // 1. 创建一个代表当前帧的对象
    json frame_obj;
    frame_obj["frame"] = frame_number;

    // 2. 创建一个数组来存放当前帧的所有装甲板
    json armors_array = json::array();
    for (const auto& armor : armors)
    {
        json armor_obj;
        armor_obj["id"] = armor.id;
        armor_obj["type"] = armor.Type;
        armor_obj["position"] = { armor.Position.x(), armor.Position.y(), armor.Position.z() };
        armor_obj["rotation_matrix"] = {
            {armor.RotationMatrix(0, 0), armor.RotationMatrix(0, 1), armor.RotationMatrix(0, 2)},
            {armor.RotationMatrix(1, 0), armor.RotationMatrix(1, 1), armor.RotationMatrix(1, 2)},
            {armor.RotationMatrix(2, 0), armor.RotationMatrix(2, 1), armor.RotationMatrix(2, 2)}
        };
        armors_array.push_back(armor_obj);
    }

    // 3. 将装甲板数组赋给当前帧对象的 "armors" 键
    frame_obj["armors"] = armors_array;

    // 4. 将整个帧对象添加到根数组中
    root_json_array.push_back(frame_obj);
}

/**
 * @brief 将JSON对象保存到文件
 * @param root_json 要保存的JSON对象
 * @param filename 输出文件名
 */
void saveJsonToFile(const json& root_json, const std::string& filename)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
// 使用 dump(4) 进行美化输出，缩进为4个空格
        file << root_json.dump(4);
        file.close();
        std::cout << "JSON data successfully saved to: " << filename << std::endl;
    }
    else
    {
        std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
    }
}
//utils================================

// /**
//  * @brief 表示一个装甲版存储的历史状态
//  */
// struct YPHistory
// {
//     std::deque<double> YawHistory;
//     std::deque<cv::Point2f> PositionHistory;
// };

// =================================================================================================
// 全局常量定义 (Global Constants)
// =================================================================================================

namespace Constants
{
// ---------------- 相机参数 (Camera Parameters) ----------------
    const cv::Mat CAMERA_MATRIX = (cv::Mat_<double>(3, 3) << 2395.480340, 0.000000, 714.096961, 0.000000, 2393.500691,
                                   573.103961, 0.000000, 0.000000, 1.000000);

    const cv::Mat DISTORTION_COEFFICIENTS = (cv::Mat_<double>(1, 5) << -0.022111, 0.020275, -0.000261, 0.000434, 0.000000);

    // ---------------- 数字识别模型 (Number Recognition Model) ----------------
    const int DNN_INPUT_WIDTH = 22;
    const int DNN_INPUT_HEIGHT = 30;
    const std::vector<cv::Point2f> PERSPECTIVE_POINTS = { cv::Point2f(0.0f, 0.0f), cv::Point2f(22.0f, 0.0f),
                                                         cv::Point2f(22.0f, 30.0f), cv::Point2f(0.0f, 30.0f) };
    const std::string DNN_MODEL_PATH = "/home/changgeng/Auto_Aim_Developing/Models/2023_4_9_hj_num_1.onnx";

    const std::vector<int> AvailableID = { 1, 2, 3, 4, 5, 6, 7, 8 };

    // ---------------- 装甲板3D物理尺寸 (Armor 3D Physical Dimensions) ----------------
    // 注意：这些尺寸单位为米(m)
    // 小装甲板
    constexpr double SMALL_ARMOR_WIDTH_BOARD = 0.065;   // 装甲板宽度
    constexpr double SMALL_ARMOR_HEIGHT_BOARD = 0.0603; // 装甲板高度
    constexpr double SMALL_ARMOR_WIDTH_LIGHT = 0.065;   // 灯条区域宽度
    constexpr double SMALL_ARMOR_HEIGHT_LIGHT = 0.0265; // 灯条区域高度
    // 大装甲板
    constexpr double LARGE_ARMOR_WIDTH_BOARD = 0.1125;  // 装甲板宽度
    constexpr double LARGE_ARMOR_HEIGHT_BOARD = 0.0613; // 装甲板高度
    constexpr double LARGE_ARMOR_WIDTH_LIGHT = 0.1125;  // 灯条区域宽度
    constexpr double LARGE_ARMOR_HEIGHT_LIGHT = 0.0265; // 灯条区域高度

    const std::vector<cv::Point3f> LARGE_ARMOR_3D_POINTS = {
        cv::Point3f(-LARGE_ARMOR_WIDTH_LIGHT + 0.005, -LARGE_ARMOR_HEIGHT_LIGHT, 0.0071),
        cv::Point3f(LARGE_ARMOR_WIDTH_LIGHT - 0.005, -LARGE_ARMOR_HEIGHT_LIGHT, 0.0071),
        cv::Point3f(LARGE_ARMOR_WIDTH_LIGHT - 0.005, LARGE_ARMOR_HEIGHT_LIGHT, -0.0071),
        cv::Point3f(-LARGE_ARMOR_WIDTH_LIGHT + 0.005, LARGE_ARMOR_HEIGHT_LIGHT, -0.0071),
        cv::Point3f(-LARGE_ARMOR_WIDTH_LIGHT, -LARGE_ARMOR_HEIGHT_LIGHT, 0.0071),
        cv::Point3f(LARGE_ARMOR_WIDTH_LIGHT, -LARGE_ARMOR_HEIGHT_LIGHT, 0.0071),
        cv::Point3f(LARGE_ARMOR_WIDTH_LIGHT, LARGE_ARMOR_HEIGHT_LIGHT, -0.0071),
        cv::Point3f(-LARGE_ARMOR_WIDTH_LIGHT, LARGE_ARMOR_HEIGHT_LIGHT, -0.0071) };

    const std::vector<cv::Point3f> SMALL_ARMOR_3D_POINTS = {
        cv::Point3f(-SMALL_ARMOR_WIDTH_LIGHT + 0.005, -SMALL_ARMOR_HEIGHT_LIGHT, 0.0071),
        cv::Point3f(SMALL_ARMOR_WIDTH_LIGHT - 0.005, -SMALL_ARMOR_HEIGHT_LIGHT, 0.0071),
        cv::Point3f(SMALL_ARMOR_WIDTH_LIGHT - 0.005, LARGE_ARMOR_HEIGHT_LIGHT, -0.0071),
        cv::Point3f(-SMALL_ARMOR_WIDTH_LIGHT + 0.005, LARGE_ARMOR_HEIGHT_LIGHT, -0.0071),
        cv::Point3f(-SMALL_ARMOR_WIDTH_LIGHT, -SMALL_ARMOR_HEIGHT_LIGHT, 0.0071),
        cv::Point3f(SMALL_ARMOR_WIDTH_LIGHT, -SMALL_ARMOR_HEIGHT_LIGHT, 0.0071),
        cv::Point3f(SMALL_ARMOR_WIDTH_LIGHT, LARGE_ARMOR_HEIGHT_LIGHT, -0.0071),
        cv::Point3f(-SMALL_ARMOR_WIDTH_LIGHT, LARGE_ARMOR_HEIGHT_LIGHT, -0.0071),
        cv::Point3f(0, -SMALL_ARMOR_HEIGHT_BOARD, 0.0071),
        cv::Point3f(0, SMALL_ARMOR_HEIGHT_BOARD, -0.0071),
        cv::Point3f(0, 0, 0) };
} // namespace Constants

// =================================================================================================
// 辅助函数 (Utility Functions)
// =================================================================================================

/**
 * @brief 从给定的ROI中判断颜色 (红/蓝).
 * @param roi 包含灯条的图像区域.
 * @return 'R' 代表红色, 'B' 代表蓝色.
 */
char determineColor(const cv::Mat& roi)
{
    int redCount = 0;
    int blueCount = 0;
    for (int i = 0; i < roi.rows; ++i)
    {
        for (int j = 0; j < roi.cols; ++j)
        {
            // 在BGR格式中, 索引2是红色, 索引0是蓝色
            cv::Vec3b pixel = roi.at<cv::Vec3b>(i, j);
            redCount += pixel[2];
            blueCount += pixel[0];
        }
    }
    return (redCount > blueCount) ? 'R' : 'B';
}

// =================================================================================================
// 视觉处理管线 (Vision Processing Pipeline)
// =================================================================================================

/**
 * @brief 在图像中寻找所有可能的灯条.
 * @param frame 输入的视频帧.
 * @param debugFrame 用于绘制调试信息的可选图像.
 * @return 检测到的灯条向量.
 */
std::vector<Light> findLights(const cv::Mat& frame, cv::Mat* debugFrame = nullptr)
{
    // 1. 图像预处理
    cv::Mat grayImage, binaryImage;
    cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
    cv::threshold(grayImage, binaryImage, 90, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);

    // 2. 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector<std::pair<cv::Point2f, cv::RotatedRect>> rawLights;

    // 3. 遍历并筛选轮廓
    for (const auto& contour : contours)
    {
        if (contour.size() < 20)
            continue;

        cv::Rect boundingRect = cv::boundingRect(contour);
        if (boundingRect.height < 30 || (double)boundingRect.width / boundingRect.height > 0.8 ||
            boundingRect.area() < 30)
        {
            continue;
        }

        // 4. 使用PCA计算灯条方向
        cv::Mat pointsMat(contour);
        pointsMat = pointsMat.reshape(1);
        cv::Mat points32f;
        pointsMat.convertTo(points32f, CV_32F);

        cv::Mat covar, mean;
        cv::calcCovarMatrix(points32f, covar, mean, cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_SCALE, CV_32F);

        if (covar.rows != 2 || covar.cols != 2)
        {
            covar = covar.reshape(1, 2);
        }
        covar = (covar + covar.t()) * 0.5; // 强制对称

        cv::Vec2f eigenvalues;
        cv::Mat eigenvectors;
        if (!cv::eigen(covar, eigenvalues, eigenvectors))
        {
            std::cerr << "Eigen decomposition failed!" << std::endl;
            continue;
        }

        // 主方向向量
        cv::Point2f majorAxis(-eigenvectors.at<float>(0, 0), -eigenvectors.at<float>(0, 1));
        double angle = round(atan2(majorAxis.y, majorAxis.x) * 180.0 / CV_PI);

        if (angle < 60)
            continue;

        // 5. 根据主轴方向构建更精确的旋转矩形
        cv::Moments moments = cv::moments(contour);
        cv::Point2f centroid(moments.m10 / moments.m00, moments.m01 / moments.m00);

        std::vector<cv::Point2f> rectPoints;
        double sin_angle_sq = pow(sin(angle * CV_PI / 180.0), 2);
        double width_scale = (abs(angle - 90) > 10) ? 0.6 : 0.9;

        rectPoints = { cv::Point2f(-boundingRect.width / 2.0 * width_scale * sin_angle_sq, -boundingRect.height / 2.0),
                      cv::Point2f(boundingRect.width / 2.0 * width_scale * sin_angle_sq, -boundingRect.height / 2.0),
                      cv::Point2f(boundingRect.width / 2.0 * width_scale * sin_angle_sq, boundingRect.height / 2.0),
                      cv::Point2f(-boundingRect.width / 2.0 * width_scale * sin_angle_sq, boundingRect.height / 2.0) };

        // 旋转角点到主轴方向
        for (auto& point : rectPoints)
        {
            float x = point.y * cos(angle * CV_PI / 180.0) - point.x * sin(angle * CV_PI / 180.0);
            float y = point.x * cos(angle * CV_PI / 180.0) + point.y * sin(angle * CV_PI / 180.0);
            point = cv::Point2f(x, y) + centroid;
        }

        rawLights.push_back({ majorAxis, cv::RotatedRect(rectPoints[0], rectPoints[1], rectPoints[2]) });
    }

    // 6. 最终筛选和格式化灯条
    std::vector<Light> finalLights;
    for (auto& rawLight : rawLights)
    {
        cv::Point2f pts[4];
        rawLight.second.points(pts);

        if (rawLight.second.size.area() < 30)
            continue;
        double height = std::max(rawLight.second.size.height, rawLight.second.size.width);
        double width = std::min(rawLight.second.size.height, rawLight.second.size.width);
        if (height < 30 || width / height > 1.2)
            continue;

        cv::Point2f orientationVec;
        cv::Point2f top, bottom;
        std::vector<cv::Point2f> boundingPoints;

        // 规范化灯条角点顺序和方向
        if (cv::norm(pts[0] - pts[1]) > cv::norm(pts[1] - pts[2]))
        { // 长边是 0-1
            if (pts[0].y > pts[1].y)
            {
                orientationVec = (pts[0] + pts[3]) / 2.0f - (pts[1] + pts[2]) / 2.0f;
                top = (pts[1] + pts[2]) / 2.0f - cv::Point2f(0, width / 8.0f);
                bottom = (pts[0] + pts[3]) / 2.0f + cv::Point2f(0, width / 8.0f);
                boundingPoints = { pts[0].x < pts[3].x ? pts[1] : pts[2], pts[0].x < pts[3].x ? pts[2] : pts[1],
                                  pts[0].x < pts[3].x ? pts[3] : pts[0], pts[0].x < pts[3].x ? pts[0] : pts[3] };
            }
            else
            {
                orientationVec = (pts[1] + pts[2]) / 2.0f - (pts[0] + pts[3]) / 2.0f;
                top = (pts[0] + pts[3]) / 2.0f - cv::Point2f(0, width / 8.0f);
                bottom = (pts[1] + pts[2]) / 2.0f + cv::Point2f(0, width / 8.0f);
                boundingPoints = { pts[0].x < pts[3].x ? pts[0] : pts[3], pts[0].x < pts[3].x ? pts[3] : pts[0],
                                  pts[0].x < pts[3].x ? pts[2] : pts[1], pts[0].x < pts[3].x ? pts[1] : pts[2] };
            }
        }
        else
        { // 长边是 1-2
            if (pts[1].y > pts[2].y)
            {
                orientationVec = (pts[1] + pts[0]) / 2.0f - (pts[2] + pts[3]) / 2.0f;
                top = (pts[2] + pts[3]) / 2.0f - cv::Point2f(0, width / 8.0f);
                bottom = (pts[1] + pts[0]) / 2.0f + cv::Point2f(0, width / 8.0f);
                boundingPoints = { pts[0].x < pts[1].x ? pts[3] : pts[2], pts[0].x < pts[1].x ? pts[2] : pts[3],
                                  pts[0].x < pts[1].x ? pts[1] : pts[0], pts[0].x < pts[1].x ? pts[0] : pts[1] };
            }
            else
            {
                orientationVec = (pts[2] + pts[3]) / 2.0f - (pts[0] + pts[1]) / 2.0f;
                top = (pts[1] + pts[0]) / 2.0f - cv::Point2f(0, width / 4.0f);
                bottom = (pts[2] + pts[3]) / 2.0f + cv::Point2f(0, width / 4.0f);
                boundingPoints = { pts[0].x < pts[1].x ? pts[0] : pts[1], pts[0].x < pts[1].x ? pts[1] : pts[0],
                                  pts[0].x < pts[1].x ? pts[2] : pts[3], pts[0].x < pts[1].x ? pts[3] : pts[2] };
            }
        }

        cv::Point2f normalizedOrientation = orientationVec / cv::norm(orientationVec);
        if (normalizedOrientation.y < 0.8)
            continue; // 必须足够竖直

        Light light;
        // --- 修复：存储归一化且方向正确的向量 ---
        light.orientationVector = normalizedOrientation;
        light.rotatedRect = rawLight.second;
        light.top = top;
        light.bottom = bottom;
        light.boundingPoints = boundingPoints;

        // 裁剪ROI进行颜色判断，避免访问越界
        cv::Rect safeRoi = rawLight.second.boundingRect() & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safeRoi.area() > 0)
        {
            light.color = determineColor(frame(safeRoi));
        }
        else
        {
            light.color = 'N'; // 无法判断颜色
        }

        finalLights.push_back(light);
    }

    // 7. 按X坐标排序
    std::sort(finalLights.begin(), finalLights.end(),
              [] (const Light& a, const Light& b) { return a.rotatedRect.center.x < b.rotatedRect.center.x; });

    // 8. 移除距离过近的重复灯条
    if (!finalLights.empty())
    {
        auto last_it = std::unique(finalLights.begin(), finalLights.end(), [] (const Light& a, const Light& b)
 {
     return std::abs(a.rotatedRect.center.x - b.rotatedRect.center.x) < 30 &&
         std::abs(a.rotatedRect.center.y - b.rotatedRect.center.y) <= 50;
        });
        finalLights.erase(last_it, finalLights.end());
    }

    // --- 变更：绘制灯条序号和角度，序号从1开始 ---
    if (debugFrame)
    {
        for (size_t i = 0; i < finalLights.size(); ++i)
        {
            const auto& light = finalLights[i];
            double angle = round(atan2(light.orientationVector.y, light.orientationVector.x) * 180.0 / CV_PI);
            std::string text = std::to_string(i + 1) + ":" + std::to_string((int)angle);
            cv::putText(*debugFrame, text, light.rotatedRect.center + cv::Point2f(10, -30), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(255, 255, 0), 2);
        }
    }

    return finalLights;
}

/**
 * @brief 将检测到的灯条匹配成装甲板.
 * @param lights 已排序的灯条向量.
 * @param activeStrategy 当前激活的策略: -1:自动, 0:强制线性, 1:强制复杂, 2:强制直接.
 * @param debugFrame 用于绘制调试信息的可选图像.
 * @return 匹配到的装甲板向量.
 */
std::vector<Armor> matchArmors(const std::vector<Light>& lights, int activeStrategy, cv::Mat* debugFrame = nullptr)
{
    std::cout << "\n[INFO] matchArmors: 开始匹配. 待匹配灯条数量: " << lights.size() << ". 当前策略: " << activeStrategy
        << std::endl;
    std::vector<Armor> armors;
    std::vector<bool> matched(lights.size(), false);

    // --- 匹配参数定义 ---
    const double MAX_ANGLE_DIFF = 5.0;   // 1. 最大允许的角度差
    const double MAX_HEIGHT_RATIO = 2.0; // 2. 最大允许的高度比
    const double MIN_DIST_H_RATIO = 0.8; // 3. 最小的灯条间距/平均高度比
    const double MAX_DIST_H_RATIO = 7.0; // 4. 最大的灯条间距/平均高度比 (一个宽松的值以包含大角度)
    const double MAX_Y_DIFF_RATIO = 0.5; // 5. 最大允许的Y坐标差 / 平均灯条高度

    while (true)
    {
        int best_i = -1, best_j = -1;
        double min_score = 1e9; // 初始化为一个很大的数

        // 步骤 1: 在所有尚未匹配的灯条中，进行全局搜索，寻找得分最低 (最匹配) 的一对
        for (size_t i = 0; i < lights.size(); ++i)
        {
            if (matched[i])
                continue;
            for (size_t j = i + 1; j < lights.size(); ++j)
            {
                if (matched[j])
                    continue;

                const auto& light1 = lights[i];
                const auto& light2 = lights[j];

                // --- 几何约束过滤 ---
                // a. 角度差
                double angle1 = round(atan2(light1.orientationVector.y, light1.orientationVector.x) * 180.0 / CV_PI);
                double angle2 = round(atan2(light2.orientationVector.y, light2.orientationVector.x) * 180.0 / CV_PI);
                double angle_diff = std::abs(angle1 - angle2);
                if (angle_diff > MAX_ANGLE_DIFF)
                    continue;

                // b. 高度比
                double h1 = std::max(light1.rotatedRect.size.width, light1.rotatedRect.size.height);
                double h2 = std::max(light2.rotatedRect.size.width, light2.rotatedRect.size.height);
                if (h1 == 0 || h2 == 0)
                    continue;
                double height_ratio = std::max(h1, h2) / std::min(h1, h2);
                if (height_ratio > MAX_HEIGHT_RATIO)
                    continue;

                // c. 间距-高度比
                double center_dist = cv::norm(light1.rotatedRect.center - light2.rotatedRect.center);
                double avg_height = (h1 + h2) / 2.0;
                if (avg_height == 0)
                    continue;
                double dist_h_ratio = center_dist / avg_height;
                if (dist_h_ratio < MIN_DIST_H_RATIO || dist_h_ratio > MAX_DIST_H_RATIO)
                    continue;

                // d. Y坐标差 / 平均高度比 (确保连线足够水平)
                double y_diff = std::abs(light1.rotatedRect.center.y - light2.rotatedRect.center.y);
                if (y_diff / avg_height > MAX_Y_DIFF_RATIO)
                    continue;

                // --- 计算得分 (核心逻辑变更) ---
                // 新的逻辑：我们优先选择“矩形度”最高的一对。
                // 这通过计算灯条方向与中心连线的正交性来衡量。
                cv::Point2f centerline = light2.rotatedRect.center - light1.rotatedRect.center;
                if (cv::norm(centerline) == 0)
                    continue;
                cv::Point2f centerline_norm = centerline / cv::norm(centerline);

                double dot1 = light1.orientationVector.dot(centerline_norm);
                double dot2 = light2.orientationVector.dot(centerline_norm);

                // 得分是两个点积的绝对值之和。理想情况下，灯条与中心连线垂直，点积为0。
                double score = std::abs(dot1) + std::abs(dot2);

                if (score < min_score)
                {
                    min_score = score;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i == -1)
        {
            std::cout << "[INFO] matchArmors: 没有更多可通过几何约束的灯条对了，匹配结束." << std::endl;
            break;
        }

        std::cout << "[DEBUG] matchArmors: 找到最佳配对! 灯条 " << best_i + 1 << " 和 " << best_j + 1
            << ", 得分(正交性): " << min_score << std::endl;

        matched[best_i] = true;
        matched[best_j] = true;

        const auto& light1 = lights[best_i];
        const auto& light2 = lights[best_j];
        double yjudge_type = (light1.rotatedRect.center.y + light2.rotatedRect.center.y) / 2;
        double xjudge_type = (light1.rotatedRect.center.x + light2.rotatedRect.center.x) / 2;
        Armor armor;
        for (int i = 0;i < lights.size();i++)
        {
            if (i == best_i || i == best_j)
            {
                continue;
            }
            if (yjudge_type > lights[i].rotatedRect.center.y && std::abs(xjudge_type - lights[i].rotatedRect.center.x))
            {
                armor.Type = "long";
            }
            else if (yjudge_type < lights[i].rotatedRect.center.y && std::abs(xjudge_type - lights[i].rotatedRect.center.x))
            {
                armor.Type = "short";
            }

        }
        armor.color = light1.color;
        armor.center = (light1.rotatedRect.center + light2.rotatedRect.center) / 2.0f;
        armor.radius = cv::norm(light1.rotatedRect.center - light2.rotatedRect.center) / 2.0;

        double height1 = std::max(light1.rotatedRect.size.width, light1.rotatedRect.size.height);
        double height2 = std::max(light2.rotatedRect.size.width, light2.rotatedRect.size.height);
        double width1 = std::min(light1.rotatedRect.size.width, light1.rotatedRect.size.height);
        double width2 = std::min(light2.rotatedRect.size.width, light2.rotatedRect.size.height);

        cv::Point2f totalOrientation = light1.orientationVector / 2.0f + light2.orientationVector / 2.0f;
        double angle = atan2(abs(totalOrientation.y), abs(totalOrientation.x));
        double widthFromCenters = cv::norm(light1.rotatedRect.center - light2.rotatedRect.center);
        double widthInner = cv::norm((light2.boundingPoints[0] + light2.boundingPoints[3]) / 2.0f -
                                     (light1.boundingPoints[1] + light1.boundingPoints[2]) / 2.0f);

        cv::Point2f new_right_top, new_left_top, new_right_bottom, new_left_bottom;
        cv::Point2f new_right_top_inner, new_left_top_inner, new_right_bottom_inner, new_left_bottom_inner;

        // --- 状态机逻辑 ---
        int finalStrategy;
        if (activeStrategy != -1)
        {
            finalStrategy = activeStrategy;
        }
        else
        {
            bool complex_cond = (height1 >= height2 && width2 > 0 && width1 / width2 > 1.0 / 0.7) ||
                (height1 < height2 && width1 > 0 && width2 / width1 > 1.0 / 0.7);
            finalStrategy = complex_cond ? 1 : 0; // 自动模式下，在复杂补偿和线性估算之间选择
        }

        armor.strategy = finalStrategy;

        if (finalStrategy == 1)
        { // 策略1：复杂几何补偿
            if (height1 >= height2)
            {
                double a = height1 / 2.0;
                double b = widthFromCenters;
                double c = cv::norm(light1.top - light2.rotatedRect.center);
                if (2 * a * b == 0)
                    continue;
                double cos_val = (a * a + b * b - c * c) / (2 * a * b);
                cos_val = std::max(-1.0, std::min(1.0, cos_val));
                double the_angle = abs(acos(cos_val)) * 180.0 / CV_PI;
                double angle_rad = (180.0 - (the_angle + angle * 180.0 / CV_PI)) * CV_PI / 180.0;
                new_right_top = light1.top + cv::Point2f(widthFromCenters * abs(cos(angle_rad)),
                                                         -widthFromCenters * abs(sin(angle_rad)) * 0.9);
                new_right_bottom = light1.bottom + cv::Point2f(widthFromCenters * abs(cos(angle_rad)),
                                                               -widthFromCenters * abs(sin(angle_rad)) * 1.1);
                new_right_top_inner = light1.boundingPoints[1] + cv::Point2f(widthInner * abs(cos(angle_rad)),
                                                                             -widthInner * abs(sin(angle_rad)) * 0.9);
                new_right_bottom_inner =
                    light1.boundingPoints[2] +
                    cv::Point2f(widthInner * abs(cos(angle_rad)), -widthInner * abs(sin(angle_rad)) * 1.1);
                armor.numberROIPoints = { light1.boundingPoints[1], new_right_top_inner, new_right_bottom_inner,
                                         light1.boundingPoints[2] };
                armor.pnpPoints = { light1.boundingPoints[1], new_right_top_inner, new_right_bottom_inner,
                                   light1.boundingPoints[2], light1.top,          new_right_top,
                                   new_right_bottom,         light1.bottom };
            }
            else
            {
                double a = height2 / 2.0;
                double b = widthFromCenters;
                double c = cv::norm(light2.top - light1.rotatedRect.center);
                if (2 * a * b == 0)
                    continue;
                double cos_val = (a * a + b * b - c * c) / (2 * a * b);
                cos_val = std::max(-1.0, std::min(1.0, cos_val));
                double the_angle = abs(acos(cos_val)) * 180.0 / CV_PI;
                double angle_rad = (180.0 - (the_angle + angle * 180.0 / CV_PI)) * CV_PI / 180.0;
                new_left_top = light2.top - cv::Point2f(widthFromCenters * abs(cos(angle_rad)),
                                                        widthFromCenters * abs(sin(angle_rad)) * 0.9);
                new_left_bottom = light2.bottom - cv::Point2f(widthFromCenters * abs(cos(angle_rad)),
                                                              widthFromCenters * abs(sin(angle_rad)) * 1.1);
                new_left_top_inner = light2.boundingPoints[0] - cv::Point2f(widthInner * abs(cos(angle_rad)),
                                                                            widthInner * abs(sin(angle_rad)) * 0.9);
                new_left_bottom_inner = light2.boundingPoints[3] - cv::Point2f(widthInner * abs(cos(angle_rad)),
                                                                               widthInner * abs(sin(angle_rad)) * 1.1);
                armor.numberROIPoints = { new_left_top_inner, light2.boundingPoints[0], light2.boundingPoints[3],
                                         new_left_bottom_inner };
                armor.pnpPoints = { new_left_top_inner,
                                   light2.boundingPoints[0],
                                   light2.boundingPoints[3],
                                   new_left_bottom_inner,
                                   new_left_top,
                                   light2.top,
                                   light2.bottom,
                                   new_left_bottom };
            }
        }
        else if (finalStrategy == 0)
        { // 策略0：线性估算
            if (height1 >= height2)
            {
                double h_diff = (height1 - height2) / 4.0;
                new_right_top = light2.top - cv::Point2f(h_diff * totalOrientation.x, h_diff * totalOrientation.y);
                new_right_bottom =
                    light2.bottom + cv::Point2f(h_diff * totalOrientation.x, -h_diff * totalOrientation.y);
                new_right_top_inner =
                    light2.boundingPoints[0] - cv::Point2f(h_diff * totalOrientation.x, h_diff * totalOrientation.y);
                new_right_bottom_inner =
                    light2.boundingPoints[3] + cv::Point2f(h_diff * totalOrientation.x, -h_diff * totalOrientation.y);
                armor.numberROIPoints = { light1.boundingPoints[1], new_right_top_inner, new_right_bottom_inner,
                                         light1.boundingPoints[2] };
                armor.pnpPoints = { light1.boundingPoints[1], new_right_top_inner, new_right_bottom_inner,
                                   light1.boundingPoints[2], light1.top,          new_right_top,
                                   new_right_bottom,         light1.bottom };
            }
            else
            { // height2 > height1
                double h_diff = (height2 - height1) / 4.0;
                new_left_top = light1.top + cv::Point2f(h_diff * totalOrientation.x, -h_diff * totalOrientation.y);
                new_left_bottom = light1.bottom - cv::Point2f(h_diff * totalOrientation.x, h_diff * totalOrientation.y);
                new_left_top_inner =
                    light1.boundingPoints[1] + cv::Point2f(h_diff * totalOrientation.x, -h_diff * totalOrientation.y);
                new_left_bottom_inner =
                    light1.boundingPoints[2] - cv::Point2f(h_diff * totalOrientation.x, h_diff * totalOrientation.y);
                armor.numberROIPoints = { new_left_top_inner, light2.boundingPoints[0], light2.boundingPoints[3],
                                         new_left_bottom_inner };
                armor.pnpPoints = { new_left_top_inner,
                                   light2.boundingPoints[0],
                                   light2.boundingPoints[3],
                                   new_left_bottom_inner,
                                   new_left_top,
                                   light2.top,
                                   light2.bottom,
                                   new_left_bottom };
            }
        }
        else
        { // finalStrategy == 2, 策略2：直接连接
            armor.numberROIPoints = { light1.boundingPoints[1], light2.boundingPoints[0], light2.boundingPoints[3],
                                     light1.boundingPoints[2] };
            armor.pnpPoints = { light1.boundingPoints[1],
                               light2.boundingPoints[0],
                               light2.boundingPoints[3],
                               light1.boundingPoints[2],
                               light1.top,
                               light2.top,
                               light2.bottom,
                               light1.bottom };
        }

        cv::Mat pnpPointsMat(armor.pnpPoints);
        if (!cv::checkRange(pnpPointsMat))
        {
            std::cout << "[WARNING] matchArmors: 补偿/估算算法为灯条 " << best_i + 1 << " 和 " << best_j + 1
                << " 生成了包含NaN/inf的无效角点，此装甲板被丢弃。" << std::endl;
            continue;
        }

        armors.push_back(armor);
    }

    std::cout << "[INFO] matchArmors: 匹配完成. 共生成 " << armors.size() << " 个有效的装甲板." << std::endl;
    return armors;
}

/**
 * @brief 识别所有装甲板上的数字.
 * @param frame 原始图像帧.
 * @param armors 包含待识别装甲板的向量.
 * @param net ONNX数字识别网络.
 * @param debugFrame 用于绘制调试信息的可选图像.
 */
void recognizeNumbers(const cv::Mat& frame, std::vector<Armor>& armors, cv::dnn::Net& net,
                      cv::Mat* debugFrame = nullptr)
{
    if (armors.empty())
        return;
    std::cout << "\n[INFO] recognizeNumbers: 开始数字识别. 待识别装甲板数量: " << armors.size() << std::endl;

    std::vector<cv::Mat> numberROIs;
    std::vector<size_t> validArmorIndices;
    std::map<int, std::vector<int>> Height_note;
    std::vector<std::string> Armor_type_note;
    std::map<int, std::vector<Armor>> Armors;
    for (size_t i = 0; i < armors.size(); ++i)
    {
        try
        {
            auto& armor = armors[i];
            if (armor.numberROIPoints.size() != 4)
            {
                continue;
            }

            cv::Mat perspectiveMatrix =
                cv::getPerspectiveTransform(armor.numberROIPoints, Constants::PERSPECTIVE_POINTS);
            cv::Mat warpedROI;
            cv::warpPerspective(frame, warpedROI, perspectiveMatrix,
                                cv::Size(Constants::DNN_INPUT_WIDTH, Constants::DNN_INPUT_HEIGHT));

            if (warpedROI.empty())
            {
                std::cout << "[DEBUG] recognizeNumbers: 装甲板 " << i << " 的ROI透视变换失败，已跳过。" << std::endl;
                continue;
            }

            cv::cvtColor(warpedROI, warpedROI, cv::COLOR_BGR2GRAY);
            cv::threshold(warpedROI, warpedROI, 30, 255, cv::THRESH_BINARY);

            numberROIs.push_back(warpedROI);
            validArmorIndices.push_back(i);
        }
        catch (const cv::Exception& e)
        {
            std::cerr << "[ERROR] recognizeNumbers: 在处理装甲板 " << i << " 时捕获到OpenCV异常: " << e.what()
                << std::endl;
            continue;
        }
    }

    if (numberROIs.empty())
    {
        std::cout << "[INFO] recognizeNumbers: 没有一个装甲板能成功提取ROI，识别结束。" << std::endl;
        return;
    }

    cv::Mat blob;
    cv::dnn::blobFromImages(numberROIs, blob, 1.0f / 255.0f,
                            cv::Size(Constants::DNN_INPUT_WIDTH, Constants::DNN_INPUT_HEIGHT));

    if (blob.empty())
        return;

    net.setInput(blob);
    cv::Mat outputs = net.forward();

    int recognized_count = 0;
    for (size_t i = 0; i < validArmorIndices.size(); ++i)
    {
        size_t armorIndex = validArmorIndices[i];
        cv::Mat currentOutput = outputs.row(i);
        cv::Point maxLoc;
        cv::minMaxLoc(currentOutput, nullptr, nullptr, nullptr, &maxLoc);
        armors[armorIndex].id = maxLoc.x;
        Armors[maxLoc.x].emplace_back(armors[armorIndex]);
        Height_note[maxLoc.x].emplace_back(armors[armorIndex].center.y);
        if (armors[armorIndex].id != 0)
        {
            recognized_count++;
            std::cout << "[DEBUG] recognizeNumbers: 装甲板 " << armorIndex << " 被识别为 ID: " << armors[armorIndex].id
                << std::endl;
        }
    }

    for (auto& pair : Height_note)
    {
        double sum = std::accumulate(pair.second.begin(), pair.second.end(), 0);
        double avel = sum / pair.second.size();
        if (pair.second.size() != 1)
        {
            for (int i = 0; i < pair.second.size(); i++)
            {
                if (pair.second[i] >= avel)
                {
                    cv::Scalar type_color = cv::Scalar(255, 255, 255);
                    Armors[pair.first][i].Type = "long";
                    cv::putText(*debugFrame, Armors[pair.first][i].Type, Armors[pair.first][i].center + cv::Point2f(0, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, type_color, 2);
                }
                else
                {
                    cv::Scalar type_color = cv::Scalar(255, 0, 255);
                    Armors[pair.first][i].Type = "short";
                    cv::putText(*debugFrame, Armors[pair.first][i].Type, Armors[pair.first][i].center + cv::Point2f(0, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, type_color, 2);
                }
            }
        }
        else
        {
            if (Armors[pair.first][0].Type == "long")
            {
                cv::Scalar type_color = cv::Scalar(255, 255, 255);
                cv::putText(*debugFrame, Armors[pair.first][0].Type, Armors[pair.first][0].center + cv::Point2f(0, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, type_color, 2);
            }
            else if (Armors[pair.first][0].Type == "short")
            {
                cv::Scalar type_color = cv::Scalar(255, 0, 255);
                cv::putText(*debugFrame, Armors[pair.first][0].Type, Armors[pair.first][0].center + cv::Point2f(0, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, type_color, 2);
            }
        }
    }

    // std::vector<Armor> unrecognizedArmors;
    // // 保存未识别的装甲板（id == 0）
    // for (const auto &armor : armors)
    // {
    //     if (armor.id == 0)
    //     {
    //         unrecognizedArmors.push_back(armor);
    //     }
    // }
    // 合并 Armors 中的所有装甲板到 armors 中

    armors.clear();

    // for (const auto &pair : Armors)
    // {
    //     armors.insert(armors.end(), pair.second.begin(), pair.second.end());
    // }

    std::vector<Armor> mergedArmors;
    // 将 map 的键排序
    std::vector<int> ids;
    for (const auto& pair : Armors)
    {
        ids.push_back(pair.first);
    }
    std::sort(ids.begin(), ids.end());
    // // 按照排序后的 ID 合并装甲板
    for (int id : ids)
    {
        mergedArmors.insert(mergedArmors.end(), Armors[id].begin(), Armors[id].end());
    }
    // 更新 armors
    armors = mergedArmors;

    // armors.insert(armors.end(), unrecognizedArmors.begin(), unrecognizedArmors.end());

    std::cout << "[INFO] recognizeNumbers: 识别完成. 共 " << recognized_count << " 个装甲板被赋予了非零ID."
        << std::endl;
}


/**
 * @brief [新增稳定层] 使用带位置记忆的跟踪器来稳定化装甲板类型
 * @details 该函数接收已经过基础识别的装甲板，并利用历史信息修正其Type。
 * 这是对recognizeNumbers_Stateless的“装饰”或“增强”。
 */
void stabilizeArmorTypes(
    std::vector<Armor>& armors, // 输入：已识别ID和无状态Type的装甲板
    std::map<int, ArmorTypeTracker>& type_tracker,
    cv::Mat* debugFrame = nullptr)
{
    if (armors.empty()) return;

    // 按ID重新分组，以便应用跟踪器
    std::map<int, std::vector<int>> armor_indices_by_id;
    for (int i = 0; i < armors.size(); ++i)
    {
        // if (armors[i].id != 0)
        // {
        armor_indices_by_id[armors[i].id].push_back(i);
        // }
    }

    // 应用状态化类型分配逻辑来“修正”类型
    for (auto const& [id, indices] : armor_indices_by_id)
    {
        auto& tracker = type_tracker[id]; // 获取或创建跟踪器

        if (indices.size() == 2)
        {
            // 按y坐标获取高低装甲板的引用
            int high_idx = (armors[indices[0]].center.y < armors[indices[1]].center.y) ? indices[0] : indices[1];
            int low_idx = (armors[indices[0]].center.y < armors[indices[1]].center.y) ? indices[1] : indices[0];
            Armor& higher_armor = armors[high_idx];
            Armor& lower_armor = armors[low_idx];

            if (!tracker.confirmed)
            {
                    // 1. 确认阶段: 直接信任无状态逻辑的结果，并存入记忆
                tracker.higher_type = higher_armor.Type; // Type来自stateless函数
                tracker.lower_type = lower_armor.Type;
                tracker.last_higher_x = higher_armor.center.x;
                tracker.last_lower_x = lower_armor.center.x;
                tracker.confirmed = true;
                std::cout << "[STABILIZER-CONFIRM] ID " << id << ": Stored stateless types. Higher: " << tracker.higher_type
                    << ", Lower: " << tracker.lower_type << std::endl;
                cv::putText(*debugFrame, higher_armor.Type, higher_armor.center + cv::Point2f(0, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2);
                cv::putText(*debugFrame, lower_armor.Type, lower_armor.center + cv::Point2f(0, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2);
            }
            else
            {
                     // 2. 更新阶段: 用记忆覆盖当前Type，并更新位置
                higher_armor.Type = tracker.higher_type;
                lower_armor.Type = tracker.lower_type;
                tracker.last_higher_x = higher_armor.center.x;
                tracker.last_lower_x = lower_armor.center.x;
                // 绘制长短轴类型
                cv::putText(*debugFrame, higher_armor.Type, higher_armor.center + cv::Point2f(0, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2);
                cv::putText(*debugFrame, lower_armor.Type, lower_armor.center + cv::Point2f(0, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2);

                std::cout << "[STABILIZER-UPDATE] ID " << id << ": Overwrote with stored types and updated positions." << std::endl;
            }
        }
        else if (indices.size() == 1 && tracker.confirmed)
        {
            // 3. 智能消歧阶段:
            Armor& single_armor = armors[indices[0]];
            float dist_to_high = std::abs(single_armor.center.x - tracker.last_higher_x);
            float dist_to_low = std::abs(single_armor.center.x - tracker.last_lower_x);
            single_armor.Type = (dist_to_high < dist_to_low) ? tracker.higher_type : tracker.lower_type;
            bool update_high = (dist_to_high < dist_to_low) ? 1 : 0;
            if (update_high)
            {
                tracker.last_higher_x = single_armor.center.x;
                tracker.last_lower_x = 0;
            }
            else
            {
                tracker.last_lower_x = single_armor.center.x;
                tracker.last_higher_x = 0;
            }
            std::cout << "[STABILIZER-DISAMBIGUATE] ID " << id << ": One armor seen, corrected type to '"
                << single_armor.Type << "'." << std::endl;
            // 绘制长短轴类型
            if (single_armor.Type == "long" || single_armor.Type == "short")
            {
                cv::Scalar type_color = cv::Scalar(255, 0, 0);
                cv::putText(*debugFrame, single_armor.Type, single_armor.center + cv::Point2f(0, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, type_color, 2);
            }

        }
        // 对于其他情况 (如只有一个装甲板且无历史记录)，我们保留stateless函数赋予的类型，不做任何事。
    }
}


std::vector<Armor> selectVehicle(const std::vector<Armor>& armors)
{
    //此处还需要添加检查上一帧最终的id是否存在,如果存在,就选择那一组id的装甲板
    if (armors.empty())
    {
        return {};
    }

    std::map<int, std::vector<Armor>> idToArmors;
    for (const auto& armor : armors)
    {
        idToArmors[armor.id].push_back(armor);
    }

    std::map<int, double> idToMaxRadius;
    const auto& availableIDs = Constants::AvailableID;

    for (int id : availableIDs)
    {
        const auto& armorsWithID = idToArmors[id];
        if (armorsWithID.empty())
        {
            continue;
        }

        double maxRadius = armorsWithID[0].radius;
        for (size_t i = 1; i < armorsWithID.size(); ++i)
        {
            if (armorsWithID[i].radius > maxRadius)
            {
                maxRadius = armorsWithID[i].radius;
            }
        }
        idToMaxRadius[id] = maxRadius;
    }

    if (idToMaxRadius.empty())
    {
        return {};
    }

    int maxKey = -1;
    double maxValue = -1;
    for (const auto& pair : idToMaxRadius)
    {
        if (pair.second > maxValue)
        {
            maxValue = pair.second;
            maxKey = pair.first;
        }
    }

    return idToArmors.count(maxKey) ? idToArmors[maxKey] : std::vector<Armor> {};
}

/**
 * @brief 使用PnP算法解算所有装甲板的位姿.
 * @param armors 包含待解算装甲板的向量.
 * @param debugFrame 用于绘制调试信息的可选图像.
 */
void solvePoses(std::vector<Armor>& armors, cv::Mat* debugFrame = nullptr)
{
    if (armors.empty())
        return;
    std::cout << "\n[INFO] solvePoses: 开始位姿解算. 待处理装甲板数量: " << armors.size() << std::endl;

    std::vector<Armor> final_Armors = selectVehicle(armors);
    for (size_t i = 0; i < armors.size(); ++i)
    {
        auto& armor = armors[i];
        std::cout << "------------------------------------------" << std::endl;
        std::cout << "[DEBUG] solvePoses: 正在处理第 " << i + 1 << " 个装甲板, ID: " << armor.id
            << ", 策略: " << armor.strategy << std::endl;

        try
        {
            std::cout << "Type:" << armor.Type << std::endl;
            if (armor.pnpPoints.size() != Constants::LARGE_ARMOR_3D_POINTS.size())
            {
                std::cout << "[INFO] solvePoses: 跳过，因为PnP点数量 (" << armor.pnpPoints.size() << ") 不正确."
                    << std::endl;
                continue;
            }

            cv::Mat rotationVector, translationVector;

            // TODO: 未来可以根据armor.id来选择LARGE_ARMOR_3D_POINTS或SMALL_ARMOR_3D_POINTS
            bool success = cv::solvePnP(Constants::LARGE_ARMOR_3D_POINTS, armor.pnpPoints, Constants::CAMERA_MATRIX,
                                        Constants::DISTORTION_COEFFICIENTS, rotationVector, translationVector, false,
                                        cv::SOLVEPNP_IPPE);

            if (!success || !cv::checkRange(rotationVector) || !cv::checkRange(translationVector))
            {
                std::cout << "[WARNING] solvePoses: PnP解算失败或结果包含NaN/inf，跳过此装甲板。" << std::endl;
                continue;
            }

            std::cout << "[DEBUG] solvePoses: PnP解算成功且数值稳定! tvec: " << translationVector.t()
                << ", rvec: " << rotationVector.t() << std::endl;

            cv::Mat rotationMatrixCV;
            cv::Rodrigues(rotationVector, rotationMatrixCV);

            cv::cv2eigen(translationVector, armor.Position);
            cv::cv2eigen(rotationMatrixCV, armor.RotationMatrix);

            // armor.PositionHistory.emplace_back(Eigen::Vector2d(armor.Position.x(), armor.Position.z()));
            if (debugFrame)
            {
                std::cout << "[DEBUG] solvePoses: 正在绘制装甲板 ID " << armor.id << " 的框选和信息。" << std::endl;

                // --- 新增：根据策略选择颜色 ---
                cv::Scalar boxColor;
                if (armor.strategy == 1)
                    boxColor = cv::Scalar(0, 0, 255); // 红色: 复杂补偿
                else if (armor.strategy == 0)
                    boxColor = cv::Scalar(0, 255, 0); // 绿色: 线性估算
                else
                    boxColor = cv::Scalar(255, 0, 0); // 蓝色: 直接连接

                for (size_t q = 0; q < 8; ++q)
                {
                    cv::line(*debugFrame, armor.pnpPoints[q], armor.pnpPoints[(q + 1) % 8], boxColor, 2);
                }

                cv::Mat rvec_mat;
                cv::Rodrigues(rotationVector, rvec_mat);
                cv::Mat_<double> mtxR, mtxQ;
                cv::Vec3d eulers = cv::RQDecomp3x3(rvec_mat, mtxR, mtxQ);
                double yaw = eulers[1] + 90;
                armor.Yaw = yaw;
                if (armor.YawHistory.size() < 3)
                {
                    armor.YawHistory.emplace_back(yaw);
                }
                else
                {
                    armor.YawHistory.pop_front();
                    armor.YawHistory.emplace_back(yaw);
                }
                // --- 优化信息显示效果 ---
                std::stringstream ss_dist, ss_yaw;
                ss_dist << std::fixed << std::setprecision(2) << cv::norm(translationVector);
                ss_yaw << std::fixed << std::setprecision(0) << yaw;

                std::string id_text = "ID: " + std::to_string(armor.id);
                std::string dist_text = "Dist: " + ss_dist.str() + "m";
                std::string yaw_text = "Yaw: " + ss_yaw.str() + "deg";

                int font_face = cv::FONT_HERSHEY_SIMPLEX;
                double font_scale = 0.6;
                int thickness = 1;
                int baseline = 0;

                cv::Size id_size = cv::getTextSize(id_text, font_face, font_scale, thickness, &baseline);
                cv::Size dist_size = cv::getTextSize(dist_text, font_face, font_scale, thickness, &baseline);
                cv::Size yaw_size = cv::getTextSize(yaw_text, font_face, font_scale, thickness, &baseline);

                int box_width = std::max({ id_size.width, dist_size.width, yaw_size.width }) + 20;
                int line_height = id_size.height + 8;
                int box_height = line_height * 3 + 10;

                float minY = 1e9;
                for (const auto& p : armor.pnpPoints)
                {
                    minY = std::min(minY, p.y);
                }
                cv::Point box_tl(armor.center.x - box_width / 2, minY - box_height - 10);

                if (box_tl.x < 0)
                    box_tl.x = 0;
                if (box_tl.y < 0)
                    box_tl.y = 0;
                if (box_tl.x + box_width > debugFrame->cols)
                    box_tl.x = debugFrame->cols - box_width;
                if (box_tl.y + box_height > debugFrame->rows)
                    box_tl.y = debugFrame->rows - box_height;

                cv::Mat roi = (*debugFrame)(cv::Rect(box_tl, cv::Size(box_width, box_height)));
                cv::Mat color_bg(roi.size(), CV_8UC3, cv::Scalar(0, 0, 0));
                double alpha = 0.5;
                cv::addWeighted(color_bg, alpha, roi, 1.0 - alpha, 0.0, roi);

                cv::rectangle(*debugFrame, cv::Rect(box_tl, cv::Size(box_width, box_height)), cv::Scalar(100, 100, 100),
                              1);
                cv::line(*debugFrame, cv::Point(box_tl.x, box_tl.y + 5), cv::Point(box_tl.x + box_width, box_tl.y + 5),
                         boxColor, 5);

                cv::putText(*debugFrame, id_text, cv::Point(box_tl.x + 10, box_tl.y + line_height), font_face,
                            font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
                cv::putText(*debugFrame, dist_text, cv::Point(box_tl.x + 10, box_tl.y + line_height * 2), font_face,
                            font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
                cv::putText(*debugFrame, yaw_text, cv::Point(box_tl.x + 10, box_tl.y + line_height * 3), font_face,
                            font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
            }
        }
        catch (const cv::Exception& e)
        {
            std::cerr << "[ERROR] solvePoses: 在处理装甲板 " << i << " (ID: " << armor.id
                << ") 时捕获到OpenCV异常: " << e.what() << std::endl;
            continue;
        }

    }
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "[INFO] solvePoses: 位姿解算流程结束。" << std::endl;

}

// =================================================================================================
// 5. 新增：追踪器结果可视化函数 (New: Tracker Result Visualization)
// =================================================================================================
void drawTrackerResults(cv::Mat& image, const Tracker& tracker)
{
    if (!tracker.isInitialized()) return;

    Eigen::VectorXd state = tracker.getState();
    double px = state(0), pz = state(1), yaw = state(2);
    double r1 = state(6), r2 = state(7);
    double y = tracker.getLastObservedY(); // 使用追踪器内部记录的Y值

    // 1. 绘制车辆中心
    std::vector<cv::Point3f> center_axis_3d = {
        cv::Point3f(px, y, pz), // Center
        cv::Point3f(px + 0.2, y, pz), // X-axis
        cv::Point3f(px, y - 0.2, pz), // Y-axis
        cv::Point3f(px, y, pz + 0.2)  // Z-axis
    };
    std::vector<cv::Point2f> center_axis_2d;
    cv::projectPoints(center_axis_3d, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), Constants::CAMERA_MATRIX, Constants::DISTORTION_COEFFICIENTS, center_axis_2d);

    cv::circle(image, center_axis_2d[0], 8, cv::Scalar(255, 255, 0), -1); // 青色中心点
    cv::line(image, center_axis_2d[0], center_axis_2d[1], cv::Scalar(0, 0, 255), 3); // X-Red
    cv::line(image, center_axis_2d[0], center_axis_2d[2], cv::Scalar(0, 255, 0), 3); // Y-Green
    cv::line(image, center_axis_2d[0], center_axis_2d[3], cv::Scalar(255, 0, 0), 3); // Z-Blue

    // 2. 绘制预测的装甲板位置
    std::vector<cv::Point3f> predicted_armors_3d;
    predicted_armors_3d.push_back(cv::Point3f(px + r1 * cos(yaw), y, pz - r1 * sin(yaw))); // Main Long
    predicted_armors_3d.push_back(cv::Point3f(px + r2 * sin(yaw), y, pz + r2 * cos(yaw))); // Main Short
    predicted_armors_3d.push_back(cv::Point3f(px - r1 * cos(yaw), y, pz + r1 * sin(yaw))); // Sym Long
    predicted_armors_3d.push_back(cv::Point3f(px - r2 * sin(yaw), y, pz - r2 * cos(yaw))); // Sym Short

    std::vector<cv::Point2f> predicted_armors_2d;
    cv::projectPoints(predicted_armors_3d, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), Constants::CAMERA_MATRIX, Constants::DISTORTION_COEFFICIENTS, predicted_armors_2d);

    cv::circle(image, predicted_armors_2d[0], 6, cv::Scalar(255, 255, 255), -1); // Main Long (White)
    cv::circle(image, predicted_armors_2d[1], 6, cv::Scalar(0, 255, 255), -1);   // Main Short (Yellow)
    cv::circle(image, predicted_armors_2d[2], 6, cv::Scalar(255, 0, 255), -1);   // Sym Long (Magenta)
    cv::circle(image, predicted_armors_2d[3], 6, cv::Scalar(255, 255, 0), -1);   // Sym Short (Cyan)

    cv::putText(image, "Tracker Center", center_axis_2d[0] + cv::Point2f(10, -10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
}


// =================================================================================================
// 主函数 (Main Function)
// =================================================================================================

int main()
{
    // 1. 初始化
    cv::dnn::Net numberRecognitionNet;
    try
    {
        numberRecognitionNet = cv::dnn::readNetFromONNX(Constants::DNN_MODEL_PATH);
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "Error: Could not read the ONNX model. Check the path: " << Constants::DNN_MODEL_PATH << std::endl;
        std::cerr << "OpenCV exception: " << e.what() << std::endl;
        return -1;
    }

    const std::string videoPath = "/home/changgeng/Auto_Aim_Developing/videos/A1.mp4"; // 可修改为其他视频路径
    cv::VideoCapture capture(videoPath);
    if (!capture.isOpened())
    {
        std::cerr << "Error: Could not open video file: " << videoPath << std::endl;
        return -1;
    }

    // --- 新增：状态机变量 ---
    int forceMode = -1;    // -1: 自动模式, 0: 强制线性, 1: 强制复杂, 2: 强制直接
    bool paused = true;    // 默认暂停启动，便于观察第一帧
    int playbackSpeed = 1; // 播放速度, 1x, 2x, 4x, 8x
    const std::vector<double> speedLevels = { 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0 };
    int speedIndex = 3; // 默认1.0x速度

    Tracker tracker(false, true); // 第一个参数: 性能摘要, 第二个参数: 原始日志
    cv::Mat frame;
    int frame_count = 0;
    double fps = capture.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 100.0; // 假设一个默认帧率
    double dt = 1.0 / fps;


    // --- 修复：在循环开始前无条件读取第一帧 ---
    if (!capture.read(frame))
    {
        std::cerr << "Error: Could not read the first frame from video." << std::endl;
        return -1;
    }
    frame_count = 1;

    // --- 核心：声明用于存储所有车辆类型状态的 map ---
    std::map<int, ArmorTypeTracker> vehicle_type_map;

    int64_t lastFrameTime = cv::getTickCount();
    json all_frames_data;
    while (true)
    {
        std::cout << "\n\n======================= 处理第 " << frame_count << " 帧 =======================" << std::endl;

        cv::Mat debugFrame = frame.clone();

        // 2. 视觉处理管线
        std::vector<Light> lights = findLights(frame, &debugFrame);
        std::vector<Armor> armors = matchArmors(lights, forceMode, &debugFrame);
        recognizeNumbers(frame, armors, numberRecognitionNet, &debugFrame);
        // stabilizeArmorTypes(armors, vehicle_type_map,&debugFrame);
        solvePoses(armors, &debugFrame);
        // logFrameDataToArray(all_frames_data, frame_count, armors);
        // std::cout << "Logged data for frame " << frame_count << std::endl;
        // ==================================================================
        // --- 追踪器集成点 (Tracker Integration Point) ---
        // ==================================================================

        // 2a. 数据适配：将主流程的Armor转换为追踪器所需的格式
        std::vector<TrackerArmor> tracker_armors;
        for (const auto& armor : armors)
        {
            if (armor.Position.norm() > 0)
            { // 只处理成功解算的装甲板
                TrackerArmor ta;
                ta.Type = armor.Type;
                ta.Position = armor.Position;
                ta.RotationMatrix = armor.RotationMatrix;
                ta.Yaw = getYawFromRotationMatrix(armor.RotationMatrix);
                tracker_armors.push_back(ta);
            }
        }

        // 2b. 调用追踪器更新
        tracker.update(tracker_armors, dt, frame_count);

        // 2c. 可视化追踪结果
        drawTrackerResults(debugFrame, tracker);
        // --- 变更：使用英文显示，避免乱码问题 ---
        std::string mode_text;
        if (forceMode == -1)
            mode_text = "Mode: Auto";
        else if (forceMode == 0)
            mode_text = "Mode: Force Linear (Green)";
        else if (forceMode == 1)
            mode_text = "Mode: Force Complex (Red)";
        else
            mode_text = "Mode: Force Direct (Blue)";

        // std::string speed_text = "Speed: " + std::to_string(playbackSpeed) + "x";
        std::string speed_text = "Speed: " + std::to_string(speedLevels[speedIndex]) + "x";

        cv::putText(debugFrame, mode_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255),
                    2);
        cv::putText(debugFrame, speed_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255),
                    2);

        std::string help_text_1 = "d: Cycle Mode";
        cv::putText(debugFrame, help_text_1, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255),
                    2);

        std::string help_text_2 = "+/-: Change Speed";
        cv::putText(debugFrame, help_text_2, cv::Point(10, 130), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255),
                    2);

        std::string help_text_3 = "p: Pause/Continue";
        cv::putText(debugFrame, help_text_3, cv::Point(10, 160), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255),
                    2);

        std::string help_text_4 = "n: Next Frame (when paused)";
        cv::putText(debugFrame, help_text_4, cv::Point(10, 190), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255),
                    2);

        std::string help_text_5 = "q: Quit";
        cv::putText(debugFrame, help_text_5, cv::Point(10, 220), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255),
                    2);

        // 3. 显示结果
        cv::namedWindow("Debug View", cv::WINDOW_KEEPRATIO);
        cv::imshow("Debug View", debugFrame);

        // 计算延时
        double speedFactor = speedLevels[speedIndex];
        int baseDelay = static_cast<int>(1000.0 / fps);
        int actualDelay = (speedFactor >= 1.0) ? 1 : static_cast<int>(baseDelay / speedFactor);

        // 4. 用户输入与帧控制
        char key = (char)cv::waitKey(paused ? 0 : 1); // 暂停时无限等待按键，否则等待1ms

        if (key == 'q')
        {
            break;
        }
        else if (key == 'd')
        {
            forceMode = (forceMode + 2) % 4 - 1; // 循环 -1, 0, 1, 2
            std::cout << "\n\n>>>>>> 状态切换: 当前策略模式 " << forceMode << " <<<<<<\n\n" << std::endl;
        }
        else if (key == 'p')
        {
            paused = !paused;
        }
        else if (key == '+' || key == '=')
        {
            speedIndex = std::min(speedIndex + 1, (int)speedLevels.size() - 1);
            std::cout << "\n>>>>>> Speed changed to: " << speedLevels[speedIndex] << "x <<<<<<\n" << std::endl;
        }
        else if (key == '-')
        {
            speedIndex = std::max(speedIndex - 1, 0);
            std::cout << "\n>>>>>> Speed changed to: " << speedLevels[speedIndex] << "x <<<<<<\n" << std::endl;
        }

        // 获取下一帧
        if (!paused || (paused && key == 'n'))
        {
            if (!paused)
            {
                double speedFactor = speedLevels[speedIndex];
                int64_t now = cv::getTickCount();
                static int64_t lastFrameTime = now;
                double timeDiff = (now - lastFrameTime) / cv::getTickFrequency();

                if (speedFactor >= 1.0)
                {
                    // 快进模式: 跳过帧
                    int framesToSkip = static_cast<int>(speedFactor) - 1;
                    for (int i = 0; i < framesToSkip; ++i)
                    {
                        capture.read(frame);
                    }
                    lastFrameTime = now;
                }
                else
                {
                    // 慢放模式: 控制帧率
                    double targetFrameTime = 1.0 / (fps * speedFactor);
                    if (timeDiff < targetFrameTime)
                    {
                        // 还没到显示下一帧的时间
                        continue; // 保持当前帧不变
                    }
                    lastFrameTime = now;
                }
            }

            // 读取下一帧
            if (!capture.read(frame))
            {
                capture.set(cv::CAP_PROP_POS_FRAMES, 0);
                if (!capture.read(frame))
                {
                    std::cerr << "Error: Failed to restart video" << std::endl;
                    break;
                }
                frame_count = 1;
            }
            else
            {
                frame_count += (speedFactor >= 1.0) ? static_cast<int>(speedFactor) : 1;
            }
        }
    }

    // 5. 释放资源
    capture.release();
    cv::destroyAllWindows();
    // saveJsonToFile(all_frames_data, "../Data/A1.json");
    return 0;
}

