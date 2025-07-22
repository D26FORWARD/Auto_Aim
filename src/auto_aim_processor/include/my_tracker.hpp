#ifndef AUTO_AIM_PROCESSOR__MY_TRACKER_HPP_
#define AUTO_AIM_PROCESSOR__MY_TRACKER_HPP_

#include "common_types.hpp"

namespace auto_aim_processor
{

// 调试器类，用于集中处理所有日志打印
  class Debugger
  {
  public:
    Debugger(bool perf_on, bool raw_on);

    // 函数 a: 原始信息记录
    void log_raw_info(const DebugData& data);
    // 函数 b: 性能分析摘要
    void log_performance_summary(const DebugData& data);

  private:
    bool performance_mode;
    bool raw_mode;
  };

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
    bool process_observation(
      const std::vector<TrackerArmor>& armors, Eigen::Vector3d& observation,
      DebugData& debug_data);
    std::vector<ArmorHypothesis> generate_hypotheses() const;
    Eigen::Vector3d back_calculate_center(
      const Eigen::Vector2d& p_armor, double yaw_armor,
      ArmorTypeID armor_id) const;
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

}  // namespace auto_aim_processor

#endif  // AUTO_AIM_PROCESSOR__MY_TRACKER_HPP_
