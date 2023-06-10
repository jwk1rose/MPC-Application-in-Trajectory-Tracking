#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <arc_spline/arc_spline.hpp>

#include <Eigen/Geometry>
#include <mpc_car/mpc_car.hpp>
using namespace std;
namespace mpc_car
{
class Nodelet : public nodelet::Nodelet
{
private:
  std::shared_ptr<MpcCar> mpcPtr_;
  ros::Timer plan_timer_;                //定时器 求解QP
  ros::Subscriber state_sub_, IMU_sub_;  //订阅state
  ros::Publisher cmd_pub_;               //发布速度
  ros::Subscriber ref_sub_;              //发布三次样条曲线
  VectorX state_;                        //小车X0
  bool state_init = false;               // state 初始化成功
  bool path_init = false;                //获得参考路径
  bool path_update = false;              //路径是否更新
  double delay_ = 0.0;
  double ax_ = 0;
  double ay_ = 0;

  bool if_path_update(const nav_msgs::Path& path, double s0)
  {
    double s = s0;
    double dx, x, ref_x = 0;
    double dy, y, ref_y = 0;
    double error = 0;
    for (int i = 1; i < min(int(path.poses.size()), 400); i++)  //检查路径相似度
    {
      if (s > mpcPtr_->s_.arcL())
      {
        if (error > 50)
          return true;
        return false;
      }

      ref_x = mpcPtr_->s_(s, 0)[0];
      ref_y = mpcPtr_->s_(s, 0)[1];
      x = path.poses[i - 1].pose.position.x;
      y = path.poses[i - 1].pose.position.y;
      error += sqrt((ref_x - x) * (ref_x - x) + (ref_y - y) * (ref_y - y));
      dx = path.poses[i].pose.position.x - x;
      dy = path.poses[i].pose.position.y - y;
      s += sqrt(dx * dx + dy * dy);
      if (error > 50)
        return true;
    }

    return false;
  }
  void plan_timer_callback(const ros::TimerEvent& event)
  {
    if (state_init && path_init)  //获得参考轨迹和x0
    {
      if (path_update)
      {
        mpcPtr_->s_.reset();
        mpcPtr_->s_.setWayPoints(mpcPtr_->track_points_x, mpcPtr_->track_points_y);  //根据离散点生成样条曲线
        cout << "======================= PATH UPDATE =====================" << endl;
        path_update = false;
      }

      // std::cout << "start solve QP" << std::endl;
      ros::Time t1 = ros::Time::now();
      // std::cout << "start solve QP STATE=" << state_.transpose() << std::endl;

      auto ret = mpcPtr_->solveQP(state_);
      ros::Time t2 = ros::Time::now();
      double solve_time = (t2 - t1).toSec();
      std::cout << "solve qp costs: " << 1e3 * solve_time << "ms" << std::endl;
      // TODO
      geometry_msgs::Twist cmd_vel;
      // VectorX x;
      // VectorU u;
      // for (int i = 0; i < mpcPtr_->N_; i++)
      // {
      //   // x.setZero();
      //   // u.setZero();
      //   // x=mpcPtr_->
      //   // mpcPtr_->getPredictXU(i, x, u);
      //   x = mpcPtr_->predictState_[i];
      //   u = mpcPtr_->predictInput_[i];
      //   std::cout << "i=" << i << "  "
      //             << "u: " << u.transpose() << std::endl;
      //   std::cout << "i=" << i << "  "
      //             << "x: " << x.transpose() << std::endl;
      //   std::cout << std::endl;
      // }

      cmd_vel.linear.x = mpcPtr_->predictState_[0][1];
      cmd_vel.linear.y = mpcPtr_->predictState_[0][4];
      cmd_vel.angular.z = mpcPtr_->predictState_[0][7];

      cmd_pub_.publish(cmd_vel);
      mpcPtr_->visualization();
    }
    return;
  }
  void state_call_back(const nav_msgs::Odometry::ConstPtr& msg)
  {
    double vx = 0;
    double vy = 0;
    double w = 0;
    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;

    double ax = 0;
    double ay = 0;
    double alp = 0;
    if (state_init)
    {
      ax = mpcPtr_->predictState_[1][2];
      ay = mpcPtr_->predictState_[1][5];
      alp = mpcPtr_->predictState_[1][8];
    }
    vx = msg->twist.twist.linear.x;
    vy = msg->twist.twist.linear.y;
    w = msg->twist.twist.angular.z;
    Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                         msg->pose.pose.orientation.z);
    Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
    state_init = true;
    state_ << x, vx, ax, y, vy, ay, euler.z(), w, alp;
  }
  void ref_call_back(const nav_msgs::Path::ConstPtr& msg)
  {
    nav_msgs::Path path;
    path = *msg;

    if (msg->poses.size() > 0)  //路径不为空
    {
      if (!path_init)  //未初始化过
      {
        cout << "========================= PATH INIT =======================" << endl;
        path_init = true;
        path_update = true;
      }
      else  //已经初始化过
      {
        double s0 = mpcPtr_->sp_;
        if (!path_update)
          path_update = if_path_update(path, s0);
        if (!path_update)
          return;
      }
      mpcPtr_->track_points_x.clear();
      mpcPtr_->track_points_y.clear();
      for (int i = 0; i < msg->poses.size(); i++)
      {
        mpcPtr_->track_points_x.push_back(msg->poses[i].pose.position.x);
        mpcPtr_->track_points_y.push_back(msg->poses[i].pose.position.y);
      }
    }
  }
  void imu_call_back(const sensor_msgs::Imu::ConstPtr& msg)
  {
    ax_ = msg->linear_acceleration.y;
    ay_ = msg->linear_acceleration.x;
    ax_ = min(max(ax_, -0.5), 0.5);
    ay_ = min(max(ay_, -0.5), 0.5);
  }

public:
  void onInit(void)
  {
    ros::NodeHandle nh(getMTPrivateNodeHandle());
    mpcPtr_ = std::make_shared<MpcCar>(nh);
    double dt = 0;
    nh.getParam("dt", dt);
    nh.getParam("delay", delay_);

    plan_timer_ = nh.createTimer(ros::Duration(dt), &Nodelet::plan_timer_callback, this);
    state_sub_ = nh.subscribe<nav_msgs::Odometry>("/state", 1, &Nodelet::state_call_back, this);
    ref_sub_ = nh.subscribe<nav_msgs::Path>("/move_base/GlobalPlanner/plan", 1, &Nodelet::ref_call_back, this);
    // IMU_sub_ = nh.subscribe<sensor_msgs::Imu>("/imu", 1, &Nodelet::imu_call_back, this);
    cmd_pub_ = nh.advertise<geometry_msgs::Twist>("/my_cmd_vel", 1);
  }
};
}  // namespace mpc_car

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(mpc_car::Nodelet, nodelet::Nodelet);