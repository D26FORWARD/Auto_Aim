import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 配置文件路径
    hik_camera_params = os.path.join(
        get_package_share_directory('hik_camera'), 'config', 'camera_params.yaml')
    hik_camera_info_url = 'package://hik_camera/config/camera_info.yaml'

    default_params = os.path.join(
        get_package_share_directory('auto_aim_bringup'), 'config', 'default.yaml')

    processor_config_path = os.path.join(
        get_package_share_directory('auto_aim_processor'),
        'config',
        'processor_params.yaml'
    )

    return LaunchDescription([
        # 摄像头参数声明
        DeclareLaunchArgument(name='camera_info_url', default_value=hik_camera_info_url),
        DeclareLaunchArgument(name='use_sensor_data_qos', default_value='false'),

        # 摄像头节点
        Node(
            package='hik_camera',
            executable='hik_camera_node',
            output='screen',
            emulate_tty=True,
            parameters=[hik_camera_params, {
                'camera_info_url': LaunchConfiguration('camera_info_url'),
                'use_sensor_data_qos': LaunchConfiguration('use_sensor_data_qos'),
            }]
        ),

        # 合并后的处理节点
        Node(
            package='auto_aim_processor',
            executable='combined_processor_node',
            name='combined_processor',
            output='screen',
            emulate_tty=True,
            parameters=[default_params, processor_config_path],
            ros_arguments=['--log-level', 'combined_processor:=DEBUG'],
            remappings=[
                ('/image_raw', '/hik_camera/image'),
                ('/tracker/target', '/fire_control/target')
            ]
        ),

        # 火控节点
        # Node(
        #     package='fire_control',
        #     executable='fire_control_node',
        #     output='screen',
        #     emulate_tty=True,
        # ),

        # TF 发布节点（camera → odom）
        Node(
            package='armor_detector',
            executable='quaternion_to_tf_node',
            # output='screen',
            emulate_tty=True,
        ),

        # 串口通信节点（Vision 发布）
        Node(
            package='serical_device_ros2',
            executable='vision_pub_node',
            # output='screen',
            emulate_tty=True,
        ),

        Node(
            package='serical_device_ros2',
            executable='robot_ctrl_main',
            # output='screen',
            emulate_tty=True,
        ),

    ])
