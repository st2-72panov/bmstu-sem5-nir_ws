# launch/uav_launch.py
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node

def generate_launch_description():
    world_name = "uav_world"
    sdf_file = "/home/viktor/BMSTU/nir_ws/src/uav_aruco/uav_world.sdf"

    # 1. Запуск Gazebo (на основе gazebo.html "Launch Gazebo from ROS 2")
    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', sdf_file],
        output='screen'
    )

    # 2. Мост для топиков (на основе ros2.html Example 5)
    bridge_topics = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={sdf_file.replace(".sdf", ".yaml")}'], # Путь к вашему yaml
        parameters=[{'config_file': '/home/viktor/BMSTU/nir_ws/src/uav_aruco/config/bridge.yaml'}], # В реальном проекте используйте полный путь или find_package
        output='screen'
    )
    # Примечание: Для корректной работы в launch файле лучше указать полный путь к yaml или использовать RosGzBridge action.
    # Ниже упрощенный вариант запуска с аргументами CLI для надежности согласно документации.
    bridge_topics_cmd = ExecuteProcess(
        cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
             '--ros-args', '-p', 'config_file:=/home/viktor/BMSTU/nir_ws/src/uav_aruco/config/bridge.yaml'],
        output='screen'
    )

    # 3. Мост для сервиса управления миром (на основе ros2.html Example 4)
    bridge_service = ExecuteProcess(
        cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
             f'/world/{world_name}/control@ros_gz_interfaces/srv/ControlWorld'],
        output='screen'
    )

    # 4. Запуск симуляции (unpause) (на основе ros2.html Example 4)
    unpause_sim = ExecuteProcess(
        cmd=['ros2', 'service', 'call',
             f'/world/{world_name}/control',
             'ros_gz_interfaces/srv/ControlWorld',
             "{world_control: {pause: false}}"],
        output='screen'
    )

    # # 5. Нода команд (полет)
    # command_node = Node(
    #     package='uav_aruco',
    #     executable='command_publisher',
    #     output='screen'
    # )

    # 6. Нода камеры (сохранение)
    camera_node = Node(
        package='uav_aruco',
        executable='image_saver',
        parameters=[{'save_directory': '/tmp/uav_images'}],
        output='screen'
    )

    # Запуск сервиса после старта моста сервиса
    delay_unpause = RegisterEventHandler(
        OnProcessStart(
            target_action=bridge_service,
            on_start=[unpause_sim]
        )
    )

    return LaunchDescription([
        gz_sim,
        bridge_topics_cmd,
        bridge_service,
        delay_unpause,
        # command_node,
        camera_node
    ])