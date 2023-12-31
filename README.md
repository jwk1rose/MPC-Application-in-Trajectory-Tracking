# MPC Application in Trajectory Tracking 🤖

### 简介 📖

本项目基于 ROS navigation + MPC 实现。navigation完成感知，全局规划，定位等任务，MPC订阅全局路径并进行跟随。项目使用的MPC是基于麦轮底盘运动学模型的线性MPC（三积分模型），对麦轮的速度，加速度，加加速度进行了约束。仿真环境使用的是 GAZEBO，地图为科大讯飞17届智能车线下赛地图，仿真车模使用的是16届科大讯飞智能车竞赛线上赛车模。MPC基于高飞老师运动规划课程的作业修改。

### 实验结果 👀️

[B站视频链接:](https://www.bilibili.com/video/BV1Th411T7NP/?spm_id_from=333.999.0.0&vd_source=0b5eb0b012de36c1500b604ce41d87c1)

### 不足 😕

严格来说，MPC跟随的是 path 而不是 Trajectory 。因为MPC直接跟随的navigation输出的全局路径，这个路径只有位置信息，也就是说这个路径是无运动学约束的！想要达到更好的效果应该对其进行运动学插值，使其具有速度，加速度，加加速度信息。

### NOTE 🔑

项目中只包含MPC部分，需要自己修改 部分关话题名称进行功能移植
