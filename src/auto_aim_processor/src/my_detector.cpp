#include "my_detector.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

namespace auto_aim_processor
{

namespace
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
}

MyDetector::MyDetector(const std::string& model_path)
{
  try {
    number_recognition_net_ = cv::dnn::readNetFromONNX(model_path);
  } catch (const cv::Exception & e) {
    std::cerr << "Error: Could not read the ONNX model. Check the path: " << model_path << std::endl;
    std::cerr << "OpenCV exception: " << e.what() << std::endl;
  }
}

char MyDetector::determineColor(const cv::Mat & roi)
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

std::vector<Light> MyDetector::findLights(const cv::Mat & frame, cv::Mat * debugFrame)
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

std::vector<Armor> MyDetector::matchArmors(const std::vector<Light>& lights, int activeStrategy, cv::Mat* debugFrame)
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

void MyDetector::recognizeNumbers(const cv::Mat & frame, std::vector<Armor>& armors, cv::Mat* debugFrame)
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
                cv::getPerspectiveTransform(armor.numberROIPoints, PERSPECTIVE_POINTS);
            cv::Mat warpedROI;
            cv::warpPerspective(frame, warpedROI, perspectiveMatrix,
                                cv::Size(DNN_INPUT_WIDTH, DNN_INPUT_HEIGHT));

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
                            cv::Size(DNN_INPUT_WIDTH, DNN_INPUT_HEIGHT));

    if (blob.empty())
        return;

    number_recognition_net_.setInput(blob);
    cv::Mat outputs = number_recognition_net_.forward();

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

void MyDetector::stabilizeArmorTypes(
    std::vector<Armor>& armors, // 输入：已识别ID和无状态Type的装甲板
    std::map<int, ArmorTypeTracker>& type_tracker,
    cv::Mat* debugFrame)
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

std::vector<Armor> MyDetector::selectVehicle(const std::vector<Armor>& armors)
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
    const auto& availableIDs = AvailableID;

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

void MyDetector::solvePoses(std::vector<Armor>& armors, cv::Mat* debugFrame)
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
            if (armor.pnpPoints.size() != LARGE_ARMOR_3D_POINTS.size())
            {
                std::cout << "[INFO] solvePoses: 跳过，因为PnP点数量 (" << armor.pnpPoints.size() << ") 不正确."
                    << std::endl;
                continue;
            }

            cv::Mat rotationVector, translationVector;

            // TODO: 未来可以根据armor.id来选择LARGE_ARMOR_3D_POINTS或SMALL_ARMOR_3D_POINTS
            bool success = cv::solvePnP(LARGE_ARMOR_3D_POINTS, armor.pnpPoints, CAMERA_MATRIX,
                                        DISTORTION_COEFFICIENTS, rotationVector, translationVector, false,
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

} // namespace auto_aim_processor
