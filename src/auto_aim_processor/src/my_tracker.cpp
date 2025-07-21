#include "auto_aim_processor/my_tracker.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace auto_aim_processor
{

// =================================================================================================
// 4. 调试函数实现 (Debugger Function Implementations)
// =================================================================================================

Debugger::Debugger(bool perf_on, bool raw_on)
: performance_mode(perf_on), raw_mode(raw_on)
{
}

void Debugger::log_raw_info(const DebugData & data)
{
  if (!raw_mode) {return;}

  std::cout << "\n\n\n################################################################################";
  std::cout << "\n#############################  帧 " << data.frame_id << " 开始 (原始日志) #################################";
  std::cout << "\n################################################################################";

  // 1. 帧开始状态
  std::cout << "\n[步骤 1] 帧开始状态" << std::endl;
  std::cout << "--------------------------------------------------------------------------------" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  if (data.state_begin.size() > 0) {  // Add check to prevent crash
    std::cout << "  - 状态向量: " << data.state_begin.transpose() << std::endl;
    std::cout << "  - 姿态角 Yaw (角度): " << data.state_begin(2) * 180.0 / M_PI << "°" << std::endl;
  } else {
    std::cout << "  - (状态未初始化)" << std::endl;
  }


  // 2. 预测
  std::cout << "\n[步骤 2] 状态预测" << std::endl;
  std::cout << "--------------------------------------------------------------------------------" << std::endl;
  if (data.state_predicted.size() > 0) {  // Add check to prevent crash
    std::cout << "  - 预测后状态向量: " << data.state_predicted.transpose() << std::endl;
    std::cout << "  - 预测后姿态角 Yaw (角度): " << data.state_predicted(2) * 180.0 / M_PI << "°" << std::endl;
  } else {
    std::cout << "  - (状态未初始化)" << std::endl;
  }


  // 3. 观测
  std::cout << "\n[步骤 3] 观测处理" << std::endl;
  std::cout << "--------------------------------------------------------------------------------" << std::endl;
  if (data.observations.empty()) {
    std::cout << "  - 接收到 0 个装甲板。" << std::endl;
  } else {
    std::cout << "  - 接收到 " << data.observations.size() << " 个装甲板:" << std::endl;
    for (size_t i = 0; i < data.observations.size(); ++i) {
      std::cout << "    - 装甲板 " << i + 1 << " (" << data.observations[i].Type << "): 位置=["
                << data.observations[i].Position.transpose() << "], 偏航角(弧度)=" << data.observations[i].Yaw <<
        " (角度): " << data.observations[i].Yaw * 180.0 / M_PI << "°" << std::endl;
    }
  }

  // 4. 匹配
  std::cout << "\n[步骤 4] 数据关联与匹配" << std::endl;
  std::cout << "--------------------------------------------------------------------------------" << std::endl;
  if (data.match_success) {
    std::cout << data.match_details << std::endl;
    std::cout << "  - 最终生成高质量观测值: 位置=[" << data.final_observation.head<2>().transpose() <<
      "], 偏航角(弧度)=" << data.final_observation(2) << " (角度): " << data.final_observation(2) * 180.0 / M_PI <<
      "°" << std::endl;
  } else {
    std::cout << "  - 匹配失败或无观测。" << std::endl;
  }

  // 5. 滤波
  std::cout << "\n[步骤 5] 滤波器更新" << std::endl;
  std::cout << "--------------------------------------------------------------------------------" << std::endl;
  if (data.match_success) {
    std::cout << "  - 新息 (观测 - 预测):" << std::endl;
    std::cout << "    - 位置误差: " << data.pos_innovation.norm() << " m" << std::endl;
    std::cout << "    - 偏航角误差: " << data.yaw_innovation * 180.0 / M_PI << "° (" << data.yaw_innovation <<
      " rad)" << std::endl;
    std::cout << "  - 决策: 位置 " << data.decision_pos << ", 偏航角 " << data.decision_yaw << std::endl;
    std::cout << "  - 角速度计算: 基于连续观测，vyaw = " << data.calculated_vyaw * 180.0 / M_PI << " °/s" << std::endl;
  } else {
    std::cout << "  - (跳过更新)" << std::endl;
  }

  // 6. 帧结束
  std::cout << "\n[步骤 6] 帧结束状态" << std::endl;
  std::cout << "--------------------------------------------------------------------------------" << std::endl;
  std::cout << "  - 最终状态向量: " << data.state_final.transpose() << std::endl;
  std::cout << "  - 最终姿态角 Yaw (角度): " << data.state_final(2) * 180.0 / M_PI << "°" << std::endl;
}

void Debugger::log_performance_summary(const DebugData & data)
{
  if (!performance_mode) {return;}

  std::cout << "\n\n--- [性能摘要] 帧: " << data.frame_id << " ---" << std::endl;

  if (!data.match_success) {
    if (data.observations.empty()) {
      std::cout << "| 状态: 跟丢 (无观测)" << std::endl;
    } else {
      // This case is now handled by the main logic, so it's less likely to be "match failure"
      std::cout << "| 状态: 跟丢 (无有效组合)" << std::endl;
    }
    if (data.state_final.size() > 0) {
      std::cout << "| 最终状态 (预测): " <<
        "x=" << data.state_final(0) << ", z=" << data.state_final(1) << ", yaw=" << data.state_final(2) * 180.0 /
        M_PI << "°" <<
        ", vx=" << data.state_final(3) << ", vz=" << data.state_final(4) << ", vyaw=" << data.state_final(5) *
        180.0 / M_PI << "°/s" <<
        std::endl;
    }
    return;
  }

  std::cout << "| 匹配决策: " << data.match_details << " (Score: " << data.match_score << ")" << std::endl;
  std::cout << "|---------------------------------|-----------------------------------|" << std::endl;
  std::cout << "|           指标 (单位)           |             数值                  |" << std::endl;
  std::cout << "|---------------------------------|-----------------------------------|" << std::endl;
  std::cout << "| 位置预测误差 (m)                | " << std::setw(25) << data.pos_innovation.norm() << " |" <<
    std::endl;
  std::cout << "| 角度预测误差 (°)                | " << std::setw(25) << data.yaw_innovation * 180.0 / M_PI <<
    " |" << std::endl;
  std::cout << "|---------------------------------|-----------------------------------|" << std::endl;
  std::cout << "| 瞬时角速度计算 (°/s)            | " << std::setw(25) << data.calculated_vyaw * 180.0 / M_PI <<
    " |" << std::endl;
  std::cout << "| 平滑后角速度估计 (°/s)          | " << std::setw(25) << data.state_final(5) * 180.0 / M_PI <<
    " |" << std::endl;
  std::cout << "|---------------------------------|-----------------------------------|" << std::endl;
  std::cout << "| 最终位置估计 (x, z)             | " << data.state_final(0) << ", " << data.state_final(1) <<
    std::endl;
  std::cout << "| 最终姿态估计 (°)                | " << std::setw(25) << data.state_final(2) * 180.0 / M_PI <<
    " |" << std::endl;
  std::cout << "| 最终线速度估计 (vx, vz)         | " << data.state_final(3) << ", " << data.state_final(4) <<
    std::endl;
  std::cout << "|---------------------------------|-----------------------------------|" << std::endl;

}

Tracker::Tracker(bool performance_debug, bool raw_debug)
: is_initialized_(false),
  x_(Eigen::VectorXd::Zero(8)),
  last_seen_id_(MAIN_LONG),
  frames_since_last_update_(0),
  last_observed_y_(0.0),
  has_last_observation_(false),
  last_observation_yaw_(0.0),
  debugger_(performance_debug, raw_debug)  // 初始化调试器
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

void Tracker::update(const std::vector<TrackerArmor> & armors, double dt, int frame_id)
{
  if (dt <= 0) {return;}

  DebugData debug_data;  // 创建数据托盘
  debug_data.frame_id = frame_id;

  if (!is_initialized_) {
    if (armors.size() >= 2) {
      initialize(armors, debug_data);
    }
    return;
  }

  debug_data.state_begin = x_;

  predict(dt);
  debug_data.state_predicted = x_;

  Eigen::Vector3d observation;
  if (process_observation(armors, observation, debug_data)) {
    update_filters(observation, dt, debug_data);
    frames_since_last_update_ = 0;
  } else {
    frames_since_last_update_++;
    has_last_observation_ = false;
  }

  debug_data.state_final = x_;
  debugger_.log_raw_info(debug_data);
  debugger_.log_performance_summary(debug_data);
}

void Tracker::initialize(const std::vector<TrackerArmor> & armors, DebugData & debug_data)
{
  const TrackerArmor * long_armor = nullptr;
  const TrackerArmor * short_armor = nullptr;

  for (const auto & armor : armors) {
    if (armor.Type == "long") {long_armor = &armor;} else if (armor.Type == "short") {short_armor = &armor;}
  }

  if (!long_armor || !short_armor) {
    return;
  }

  double center_yaw_est = normalizeAngle(long_armor->Yaw);
  Eigen::Vector3d center1 = back_calculate_center(
    Eigen::Vector2d(
      long_armor->Position.x(),
      long_armor->Position.z()), center_yaw_est, MAIN_LONG);
  double expected_short_yaw = normalizeAngle(center_yaw_est + M_PI / 2.0);
  Eigen::Vector3d center2 = back_calculate_center(
    Eigen::Vector2d(
      short_armor->Position.x(),
      short_armor->Position.z()), expected_short_yaw, MAIN_SHORT);
  Eigen::Vector2d final_center_pos = (center1.head<2>() + center2.head<2>()) / 2.0;

  x_.setZero();
  x_(0) = final_center_pos(0);
  x_(1) = final_center_pos(1);
  x_(2) = center_yaw_est;

  double r1_obs = (Eigen::Vector2d(long_armor->Position.x(), long_armor->Position.z()) -
    final_center_pos).norm();
  double r2_obs = (Eigen::Vector2d(short_armor->Position.x(), short_armor->Position.z()) -
    final_center_pos).norm();
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

void Tracker::update_filters(const Eigen::Vector3d & observation, double dt, DebugData & debug_data)
{
  double current_alpha_pos, current_beta_pos;
  double current_alpha_yaw;

  Eigen::Vector2d pos_innovation = observation.head<2>() - x_.head<2>();
  double yaw_innovation = normalizeAngle(observation(2) - x_(2));
  debug_data.pos_innovation = pos_innovation;
  debug_data.yaw_innovation = yaw_innovation;

  if (pos_innovation.norm() > MANEUVER_THRESHOLD_POS) {
    current_alpha_pos = alpha_pos_maneuver_;
    current_beta_pos = beta_pos_maneuver_;
    debug_data.decision_pos = "机动模式";
  } else {
    current_alpha_pos = alpha_pos_smooth_;
    current_beta_pos = beta_pos_smooth_;
    debug_data.decision_pos = "平滑模式";
  }

  if (std::abs(yaw_innovation) > MANEUVER_THRESHOLD_YAW) {
    current_alpha_yaw = alpha_yaw_maneuver_;
    debug_data.decision_yaw = "机动模式";
  } else {
    current_alpha_yaw = alpha_yaw_smooth_;
    debug_data.decision_yaw = "平滑模式";
  }

  x_.head<2>() += current_alpha_pos * pos_innovation;
  x_.segment<2>(3) += (current_beta_pos / dt) * pos_innovation;
  x_(2) = normalizeAngle(x_(2) + current_alpha_yaw * yaw_innovation);

  if (has_last_observation_) {
    double vyaw = normalizeAngle(observation(2) - last_observation_yaw_) / dt;
    x_(5) = 0.7 * x_(5) + 0.3 * vyaw;
    debug_data.calculated_vyaw = vyaw;
  } else {
    x_(5) = 0;
  }

  last_observation_yaw_ = observation(2);
  has_last_observation_ = true;
}

bool Tracker::process_observation(
  const std::vector<TrackerArmor> & armors,
  Eigen::Vector3d & observation, DebugData & debug_data)
{
  debug_data.observations = armors;
  if (armors.empty()) {
    debug_data.match_success = false;
    return false;
  }

  last_observed_y_ = armors[0].Position.y();
  auto hypotheses = generate_hypotheses();
  for (const auto & h : hypotheses) {
    debug_data.hypotheses.push_back({h.name, h.predicted_pos_yaw});
  }

  const TrackerArmor * long_armor = nullptr;
  const TrackerArmor * short_armor = nullptr;

  if (armors.size() >= 2) {
    for (const auto & armor : armors) {
      if (armor.Type == "long" && long_armor == nullptr) {long_armor = &armor;}
      if (armor.Type == "short" && short_armor == nullptr) {short_armor = &armor;}
    }
  }

  if (long_armor && short_armor) {
    debug_data.match_type = "双板匹配";
    double best_score = 1e9;
    ArmorHypothesis best_long_hypo, best_short_hypo;

    for (const auto & h_long : hypotheses) {
      if (h_long.id != MAIN_LONG && h_long.id != SYM_LONG) {continue;}
      for (const auto & h_short : hypotheses) {
        if (h_short.id != MAIN_SHORT && h_short.id != SYM_SHORT) {continue;}
        double yaw_error = std::abs(normalizeAngle(long_armor->Yaw - h_long.predicted_pos_yaw(2))) +
          std::abs(normalizeAngle(short_armor->Yaw - h_short.predicted_pos_yaw(2)));
        double pos_error = (Eigen::Vector2d(long_armor->Position.x(), long_armor->Position.z()) -
          h_long.predicted_pos_yaw.head<2>()).norm() +
          (Eigen::Vector2d(short_armor->Position.x(), short_armor->Position.z()) -
          h_short.predicted_pos_yaw.head<2>()).norm();
        double score = pos_error + YAW_WEIGHT * yaw_error;
        if (score < best_score) {
          best_score = score;
          best_long_hypo = h_long;
          best_short_hypo = h_short;
        }
      }
    }

    debug_data.match_success = true;
    std::stringstream ss;
    ss << "观测 Long (Yaw: " << long_armor->Yaw * 180.0 / M_PI << "°) -> " << best_long_hypo.name <<
      "\n" << "    观测 Short (Yaw: " << short_armor->Yaw * 180.0 / M_PI << "°) -> " << best_short_hypo.name;
    debug_data.match_details = ss.str();
    debug_data.match_score = best_score;

    Eigen::Vector3d center_obs1 = back_calculate_center(
      Eigen::Vector2d(
        long_armor->Position.x(),
        long_armor->Position.z()), long_armor->Yaw, best_long_hypo.id);
    Eigen::Vector3d center_obs2 = back_calculate_center(
      Eigen::Vector2d(
        short_armor->Position.x(),
        short_armor->Position.z()), short_armor->Yaw, best_short_hypo.id);
    observation = (center_obs1 + center_obs2) / 2.0;
    double s = (sin(center_obs1(2)) + sin(center_obs2(2))) / 2.0;
    double c = (cos(center_obs1(2)) + cos(center_obs2(2))) / 2.0;
    observation(2) = atan2(s, c);

    Eigen::Vector2d center_pos_obs = observation.head<2>();
    double r1_obs = (Eigen::Vector2d(long_armor->Position.x(), long_armor->Position.z()) -
      center_pos_obs).norm();
    double r2_obs = (Eigen::Vector2d(short_armor->Position.x(), short_armor->Position.z()) -
      center_pos_obs).norm();
    x_(6) = (1 - alpha_r_) * x_(6) + alpha_r_ * r1_obs;
    x_(7) = (1 - alpha_r_) * x_(7) + alpha_r_ * r2_obs;

    swapRadiiIfNeeded();
    clampRadii();
  } else {
    const auto & armor = armors[0];
    debug_data.match_type = "单板匹配";
    double best_score = 1e9;
    ArmorHypothesis best_match;

    for (const auto & hypo : hypotheses) {
      bool is_hypo_long = hypo.id == MAIN_LONG || hypo.id == SYM_LONG;
      if ((armor.Type == "long") != is_hypo_long) {continue;}
      double pos_error = (Eigen::Vector2d(armor.Position.x(), armor.Position.z()) -
        hypo.predicted_pos_yaw.head<2>()).norm();
      double yaw_error = std::abs(normalizeAngle(armor.Yaw - hypo.predicted_pos_yaw(2)));
      double score = pos_error + YAW_WEIGHT * yaw_error;
      if (score < best_score) {
        best_score = score;
        best_match = hypo;
      }
    }

    debug_data.match_success = true;
    std::stringstream ss;
    ss << "观测 " << armor.Type << " (Yaw: " << armor.Yaw * 180.0 / M_PI << "°) -> " <<
      best_match.name;
    debug_data.match_details = ss.str();
    debug_data.match_score = best_score;

    observation = back_calculate_center(
      Eigen::Vector2d(
        armor.Position.x(),
        armor.Position.z()), armor.Yaw, best_match.id);
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

  hypotheses[MAIN_LONG] = { {center_x + r1 * cos(center_yaw), center_z - r1 * sin(center_yaw),
    normalizeAngle(center_yaw)}, MAIN_LONG, "主长轴"};
  hypotheses[SYM_LONG] = { {center_x - r1 * cos(center_yaw), center_z + r1 * sin(center_yaw),
    normalizeAngle(center_yaw - M_PI)}, SYM_LONG, "对称长轴"};
  hypotheses[MAIN_SHORT] = { {center_x + r2 * sin(center_yaw), center_z + r2 * cos(center_yaw),
    normalizeAngle(center_yaw + M_PI / 2.0)}, MAIN_SHORT, "主短轴"};
  hypotheses[SYM_SHORT] = { {center_x - r2 * sin(center_yaw), center_z - r2 * cos(center_yaw),
    normalizeAngle(center_yaw - M_PI / 2.0)}, SYM_SHORT, "对称短轴"};

  return hypotheses;
}

Eigen::Vector3d Tracker::back_calculate_center(
  const Eigen::Vector2d & p_armor, double yaw_armor,
  ArmorTypeID armor_id) const
{
  Eigen::Vector3d center_obs;
  double center_yaw_obs;
  double r;
  bool use_estimated_radii = is_initialized_;

  switch (armor_id) {
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

  center_obs << center_x_obs, center_z_obs, center_yaw_obs;
  return center_obs;
}

Eigen::Vector3d Tracker::predictArmorPosition(
  const std::string & armor_type,
  double prediction_time_ms) const
{
  if (!is_initialized_) {return Eigen::Vector3d::Zero();}

  double dt = prediction_time_ms / 1000.0;

  Eigen::VectorXd future_status = x_;
  future_status(0) += x_(3) * dt;
  future_status(1) += x_(4) * dt;
  future_status(2) = normalizeAngle(x_(2) + x_(5) * dt);

  const double center_x = future_status(0), center_z = future_status(1), center_yaw =
    future_status(2);
  double armor_x, armor_z;

  if (armor_type == "long") {
    const double r = future_status(6);
    armor_x = center_x + r * std::cos(center_yaw);
    armor_z = center_z - r * std::sin(center_yaw);
  } else {  // "short"
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
  if (x_(6) < x_(7)) {
    std::swap(x_(6), x_(7));
  }
}

}  // namespace auto_aim_processor
