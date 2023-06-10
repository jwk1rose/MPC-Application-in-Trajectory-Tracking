#pragma once
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <arc_spline/arc_spline.hpp>
#include <deque>
#include <iosqp/iosqp.hpp>
#include "tf2_ros/buffer.h"

namespace mpc_car
{
static constexpr int n = 9;
static constexpr int m = 3;
typedef Eigen::Matrix<double, n, n> MatrixA;
typedef Eigen::Matrix<double, n, m> MatrixB;
typedef Eigen::Matrix<double, n, 1> VectorX;
typedef Eigen::Vector3d VectorU;

class MpcCar
{
private:
  ros::NodeHandle nh_;
  ros::Publisher ref_pub_, traj_pub_, traj_delay_pub_, ref_point_pub_;

  double dt_;
  double rho_;
  double rhoN_;
  double ks_;
  double km_;
  double kl_;
  double v_max_, a_max_, w_max_, alf_max_, j_max_;  //约束量范围
  double delay_;                                    //执行器延迟

  double desired_v_;  //参考点在参考轨迹上的移动速度

  osqp::IOSQP qpSolver_;  // qp求解器

  std::vector<double> s0_;  //未来T个周期内预测状态

  std::deque<VectorU> historyInput_;  //历史输入 长度<=history_length_
  int history_length_;
  VectorX x0_observe_;  // x0
  /*
  // x_{n+1} = Ad * x_{n} + Bd * u_{n}

          / px_{n+1} \
          | vx_{n+1} |
          | ax_{n+1} |
          | py_{n+1} |
  x_{n+1}:| vy_{n+1} |
          | ay_{n+1} |
          | psi_{n+1}|
          | w_{n+1}  |
          \ alp_{n+1}/
  //     /  1   t   0.5t^2                                  \
  //     |  0   1     t                                     |
  //     |  0   0     1                                     |
  //     |                 1   t   0.5t^2                   |
  // Ad_ |                 0   1     t                      |
  //     |                 0   0     1                      |
  //     |                                  1   t   0.5t^2  |
  //     |                                  0   1     t     |
  //     \                                  0   0     1     /
  //
  //     /  0   0   0  \
  //     |  0   0   0  |
  //     |  t   0   0  |
  //     |  0   0   0  |
  // Bd_ |  0   0   0  |
  //     |  0   t   0  |
  //     |  0   0   0  |
  //     |  0   0   0  |
  //     \  0   0   t  /
  */
  MatrixA Ad_;
  MatrixB Bd_;

  /**
   * osqp interface:
   * minimize     0.5 x^T P_ x + q_^T x
   * subject to   l_ <= A_ x <= u_
   **/
  Eigen::SparseMatrix<double> P_, q_, A_, l_, u_;  //二次项系数 一次项系数 约束系数 约束下界 约束上界

  /* *
   *               /  x1  \
   *               |  x2  |
   *  lx_ <=  Cx_  |  x3  |  <= ux_
   *               | ...  |
   *               \  xN  /
   * */

  Eigen::SparseMatrix<double> Cx_, lx_, ux_;  // p, v constrains 状态量约束
  /* *
   *               /  u0  \
   *               |  u1  |
   *  lu_ <=  Cu_  |  u2  |  <= uu_
   *               | ...  |
   *               \ uN-1 /
   * */
  Eigen::SparseMatrix<double> Cu_, lu_, uu_;  // a delta ks constrains 输入量约束
  Eigen::SparseMatrix<double> Qx_;            //以x作为优化对象的二次型系数

  //获取参考点信息
  void calLinPoint(const double& s0, double& phi)
  {
    Eigen::Vector2d dxy = s_(s0, 1);   // s0处速度
    Eigen::Vector2d ddxy = s_(s0, 2);  // s0处加速度
    double dx = dxy.x();
    double dy = dxy.y();
    phi = atan2(dy, dx);
    // desired_v_ = sqrt(dx * dx + dy * dy);
    // std::cout << "dx=" << dx << "dy=" << dy << "desired_v_=" << desired_v_ << std::endl;
  }

  // state的导数
  inline VectorX diff(const VectorX& state, const VectorU& input) const
  {
    VectorX ds;
    double px = state(0);
    double vx = state(1);
    double ax = state(2);
    double py = state(3);
    double vy = state(4);
    double ay = state(5);
    double psi = state(6);
    double w = state(7);
    double alp = state(8);
    double jx = input(0);
    double jy = input(1);
    double jz = input(2);
    ds(0) = vx;
    ds(1) = ax;
    ds(2) = jx;
    ds(3) = vy;
    ds(4) = ay;
    ds(5) = jy;
    ds(6) = w;
    ds(7) = alp;
    ds(8) = jz;
    return ds;
  }

  inline void step(VectorX& state, const VectorU& input, const double dt) const
  {
    // Runge–Kutta 近似求解常微分方程
    VectorX k1 = diff(state, input);
    VectorX k2 = diff(state + k1 * dt / 2, input);
    VectorX k3 = diff(state + k2 * dt / 2, input);
    VectorX k4 = diff(state + k3 * dt, input);
    state = state + (k1 + k2 * 2 + k3 * 2 + k4) * dt / 6;
  }

  VectorX compensateDelay(const VectorX& x0)
  {
    VectorX x0_delay = x0;
    double dt = 0.001;
    for (double t = delay_; t > 0; t -= dt)
    {
      int i = std::ceil(t / dt_);
      VectorU input = historyInput_[history_length_ - i];
      step(x0_delay, input, dt);
    }
    return x0_delay;
  }

public:
  arc_spline::ArcSpline s_;  //三次样条曲线
  double sp_;
  std::vector<double> track_points_x, track_points_y;
  int N_;
  std::vector<VectorX> predictState_;  //未来T个周期内预测状态
  std::vector<VectorU> predictInput_;  //未来T-1个周期的预测输入
  MpcCar(ros::NodeHandle& nh) : nh_(nh)
  {
    sp_ = 0;
    // load map
    nh.getParam("desired_v", desired_v_);
    // load parameters
    nh.getParam("dt", dt_);
    nh.getParam("rho", rho_);
    nh.getParam("N", N_);
    nh.getParam("rhoN", rhoN_);
    nh.getParam("v_max", v_max_);
    nh.getParam("a_max", a_max_);
    nh.getParam("w_max", w_max_);
    nh.getParam("alf_max", alf_max_);
    nh.getParam("j_max", j_max_);
    nh.getParam("delay", delay_);
    nh.getParam("ks", ks_);
    nh.getParam("km", km_);
    nh.getParam("kl", kl_);

    history_length_ = std::ceil(delay_ / dt_);

    ref_pub_ = nh.advertise<nav_msgs::Path>("reference_path", 1);
    traj_pub_ = nh.advertise<nav_msgs::Path>("traj", 1);
    traj_delay_pub_ = nh.advertise<nav_msgs::Path>("traj_delay", 1);
    ref_point_pub_ = nh.advertise<nav_msgs::Path>("ref_point", 1);

    // set initial value of Ad, Bd
    Ad_.setZero();
    Bd_.setZero();
    Ad_.block(6, 6, 3, 3) << 1, dt_, 0.5 * dt_ * dt_, 0, 1, dt_, 0, 0, 1;
    Ad_.block(0, 0, 3, 3) = Ad_.block(3, 3, 3, 3) = Ad_.block(6, 6, 3, 3);
    Bd_.coeffRef(2, 0) = Bd_.coeffRef(5, 1) = Bd_.coeffRef(8, 2) = dt_;
    Bd_.coeffRef(1, 0) = Bd_.coeffRef(4, 1) = Bd_.coeffRef(7, 2) = dt_ * dt_ * 0.5;
    Bd_.coeffRef(0, 0) = Bd_.coeffRef(3, 1) = Bd_.coeffRef(6, 2) = dt_ * dt_ * dt_ / 6;

    std::cout << "Ad_=" << std::endl << Ad_ << std::endl;
    std::cout << "Bd_=" << std::endl << Bd_ << std::endl;

    // set size of sparse matrices
    P_.resize(m * N_, m * N_);
    q_.resize(m * N_, 1);
    Qx_.resize(n * N_, n * N_);
    // stage cost
    Qx_.setZero();
    // TODO: set special boundary cost
    for (int i = 0; i < N_; ++i)
    {
      Qx_.coeffRef(i * n, i * n) = 1;
      Qx_.coeffRef(i * n + 3, i * n + 3) = 1;
      Qx_.coeffRef(i * n + 6, i * n + 6) = rho_;  //不对速度优化
    }
    std::cout << "Qx_=" << std::endl << Qx_ << std::endl;

    int n_cons = 9;                  //所有的约束数量
    A_.resize(n_cons * N_, m * N_);  //所有的约束系数
    l_.resize(n_cons * N_, 1);
    u_.resize(n_cons * N_, 1);

    int x_cons = 6;
    Cx_.resize(x_cons * N_, n * N_);  //状态量约束系数矩阵 仅有一个v
    lx_.resize(x_cons * N_, 1);
    ux_.resize(x_cons * N_, 1);
    // TODO: set input constains
    int u_cons = 3;
    Cu_.resize(u_cons * N_, m * N_);  //输入量约束系数矩阵
    lu_.resize(u_cons * N_, 1);
    uu_.resize(u_cons * N_, 1);
    Cu_.setIdentity();
    // set lower and upper boundaries
    for (int i = 0; i < N_; ++i)
    {
      Cx_.coeffRef(i * x_cons, i * n + 1) = 1;
      Cx_.coeffRef(i * x_cons + 1, i * n + 2) = 1;
      Cx_.coeffRef(i * x_cons + 2, i * n + 4) = 1;
      Cx_.coeffRef(i * x_cons + 3, i * n + 5) = 1;
      Cx_.coeffRef(i * x_cons + 4, i * n + 7) = 1;
      Cx_.coeffRef(i * x_cons + 5, i * n + 8) = 1;
      // set stage constraints of inputs
      lx_.coeffRef(i * x_cons + 0, 0) = -v_max_;
      lx_.coeffRef(i * x_cons + 1, 0) = -a_max_;
      lx_.coeffRef(i * x_cons + 2, 0) = -v_max_;
      lx_.coeffRef(i * x_cons + 3, 0) = -a_max_;
      lx_.coeffRef(i * x_cons + 4, 0) = -v_max_;
      lx_.coeffRef(i * x_cons + 5, 0) = -a_max_;
      ux_.coeffRef(i * x_cons + 0, 0) = v_max_;
      ux_.coeffRef(i * x_cons + 1, 0) = a_max_;
      ux_.coeffRef(i * x_cons + 2, 0) = v_max_;
      ux_.coeffRef(i * x_cons + 3, 0) = a_max_;
      ux_.coeffRef(i * x_cons + 4, 0) = v_max_;
      ux_.coeffRef(i * x_cons + 5, 0) = a_max_;
      lu_.coeffRef(i * u_cons + 0, 0) = -j_max_;
      lu_.coeffRef(i * u_cons + 1, 0) = -j_max_;
      lu_.coeffRef(i * u_cons + 2, 0) = -j_max_;
      uu_.coeffRef(i * u_cons + 0, 0) = j_max_;
      uu_.coeffRef(i * u_cons + 1, 0) = j_max_;
      uu_.coeffRef(i * u_cons + 2, 0) = j_max_;
    }
    std::cout << "Cx_=" << std::endl << Cx_ << std::endl;
    std::cout << "Cu_=" << std::endl << Cu_ << std::endl;
    std::cout << "lx_=" << std::endl << lx_ << std::endl;
    std::cout << "ux_=" << std::endl << ux_ << std::endl;
    std::cout << "lu_=" << std::endl << lu_ << std::endl;
    std::cout << "uu_=" << std::endl << uu_ << std::endl;

    // set predict mats size
    predictState_.resize(N_);
    predictInput_.resize(N_);
    for (int i = 0; i < N_; ++i)
    {
      predictInput_[i].setZero();
    }
    for (int i = 0; i < history_length_; ++i)
    {
      historyInput_.emplace_back(0, 0, 0);
    }
  }

  int solveQP(const VectorX& x0_observe)
  {
    s0_.clear();
    x0_observe_ = x0_observe;                        //初始化x0
    historyInput_.pop_front();                       //将存储的最早的历史的输入弹出
    historyInput_.push_back(predictInput_.front());  //将上次的预测输入压入
    VectorX x0 = compensateDelay(x0_observe_);       //考虑执行延迟
    // set BB, AA
    Eigen::MatrixXd BB, AA;  // resize and zero
    BB.setZero(n * N_, m * N_);
    AA.setZero(n * N_, n);
    Eigen::Vector2d temp;
    temp(0) = x0(0);
    temp(1) = x0(3);
    double s0 = s_.findS(temp);  //返回当前点距离曲线最近的一点在整条曲线上的位置(长度) 这个点作为初始参考点
    double phi;
    double last_phi = x0(6);  //车体朝向
    Eigen::SparseMatrix<double> qx;
    qx.resize(n * N_, 1);
    double target_v = desired_v_;
    for (int i = 0; i < N_; ++i)
    {
      calLinPoint(s0, phi);  //获得参考点处的 phi v delta 作为线性化的参考点
      if (phi - last_phi > M_PI)
      {
        phi -= 2 * M_PI;
      }
      else if (phi - last_phi < -M_PI)
      {
        phi += 2 * M_PI;
      }
      last_phi = phi;
      // calculate big state-space matrices
      /* *                BB                       AA
       * x1    /       B0     0    ... 0 \    /  A0      \
       * x2    |      A1B0    B1   ... 0 |    | A1A0     |
       * x3  = |    A2A1B0   A2B1  ... 0 |U + | ...      |x0
       * ...   |     ...      ...  ... 0 |    | ...      |
       * xN    \AN*A(N-1)B0   ...  ... BN/    \ AN*..A0  /
       *
       *     X = BB * U + AA * x0 + GG*gg
       * */
      if (i == 0)
      {
        BB.block(0, 0, n, m) = Bd_;  // B
        AA.block(0, 0, n, n) = Ad_;  // A
      }
      else
      {
        // set BB AA gg
        AA.block(i * n, 0, n, n) = Ad_ * AA.block((i - 1) * n, 0, n, n);
        BB.block(i * n, i * m, n, m) = Bd_;
        for (int j = 0; j < i; j++)
          BB.block(i * n, j * m, n, m) = Ad_ * BB.block((i - 1) * n, j * m, n, m);
      }
      Eigen::Vector2d xy = s_(s0);  // reference (x_r, y_r)

      // cost function should be represented as follows:
      /* *
       *           /  x1  \T       /  x1  \         /  x1  \
       *           |  x2  |        |  x2  |         |  x2  |
       *  J =  0.5 |  x3  |   Qx_  |  x3  | + qx^T  |  x3  | + const.
       *           | ...  |        | ...  |         | ...  |
       *           \  xN  /        \  xN  /         \  xN  /
       * */

      qx.coeffRef(i * n, 0) = -xy[0];
      qx.coeffRef(i * n + 3, 0) = -xy[1];
      qx.coeffRef(i * n + 6, 0) = -rho_ * phi;
      // desired_v_ = sqrt(predictState_.front()[1] * predictState_.front()[1] +
      //                   predictState_.front()[4] * predictState_.front()[4]);
      double k = s_(s0, 2).norm();

      if (k > 0.1 && k < 0.3)
        target_v = kl_ * desired_v_;
      else if (k >= 0.3 && k < 1.0)
        target_v = km_ * desired_v_;
      else if (k >= 1.0)
        target_v = ks_ * desired_v_;
      else
        target_v = desired_v_;
      s0 += dt_ * target_v;
      s0 = s0 < s_.arcL() ? s0 : s_.arcL();
      s0_.push_back(s0);
      if (i == 0)
        sp_ = s0;
    }
    Eigen::SparseMatrix<double> BB_sparse = BB.sparseView();
    Eigen::SparseMatrix<double> AA_sparse = AA.sparseView();
    Eigen::SparseMatrix<double> x0_sparse = x0.sparseView();

    // state constrants propogate to input constraints using "X = BB * U + AA * x0 + gg"
    /* *
     *               /  x1  \                              /  u0  \
     *               |  x2  |                              |  u1  |
     *  lx_ <=  Cx_  |  x3  |  <= ux_    ==>    lx <=  Cx  |  u2  |  <= ux
     *               | ...  |                              | ...  |
     *               \  xN  /                              \ uN-1 /
     * */
    Eigen::SparseMatrix<double> Cx = Cx_ * BB_sparse;
    Eigen::SparseMatrix<double> lx = lx_ - Cx_ * AA_sparse * x0_sparse;
    Eigen::SparseMatrix<double> ux = ux_ - Cx_ * AA_sparse * x0_sparse;

    /* *      / Cx  \       / lx  \       / ux  \
     *   A_ = \ Cu_ /, l_ = \ lu_ /, u_ = \ uu_ /
     * */

    Eigen::SparseMatrix<double> A_T = A_.transpose();
    A_T.middleCols(0, Cx.rows()) = Cx.transpose();
    A_T.middleCols(Cx.rows(), Cu_.rows()) = Cu_.transpose();
    A_ = A_T.transpose();
    for (int i = 0; i < lx.rows(); ++i)
    {
      l_.coeffRef(i, 0) = lx.coeff(i, 0);
      u_.coeffRef(i, 0) = ux.coeff(i, 0);
    }
    for (int i = 0; i < lu_.rows(); ++i)
    {
      l_.coeffRef(i + lx.rows(), 0) = lu_.coeff(i, 0);
      u_.coeffRef(i + lx.rows(), 0) = uu_.coeff(i, 0);
    }
    Eigen::SparseMatrix<double> BBT_sparse = BB_sparse.transpose();
    P_ = BBT_sparse * Qx_ * BB_sparse;
    q_ = BBT_sparse * Qx_.transpose() * (AA_sparse * x0_sparse) + BBT_sparse * qx;
    // osqp
    Eigen::VectorXd q_d = q_.toDense();
    Eigen::VectorXd l_d = l_.toDense();
    Eigen::VectorXd u_d = u_.toDense();
    qpSolver_.setMats(P_, q_d, A_, l_d, u_d);
    qpSolver_.solve();
    int ret = qpSolver_.getStatus();
    if (ret != 1)
    {
      ROS_ERROR("fail to solve QP!");
      return ret;
    }
    Eigen::VectorXd sol = qpSolver_.getPrimalSol();
    Eigen::MatrixXd solMat = Eigen::Map<const Eigen::MatrixXd>(sol.data(), m, N_);
    Eigen::VectorXd solState = BB * sol + AA * x0;
    Eigen::MatrixXd predictMat = Eigen::Map<const Eigen::MatrixXd>(solState.data(), n, N_);

    for (int i = 0; i < N_; ++i)
    {
      predictInput_[i] = solMat.col(i);
      predictState_[i] = predictMat.col(i);
    }
    return ret;
  }

  void getPredictXU(double t, VectorX& state, VectorU& input)
  {  //获取t对应的预测区间，获取预测区间的状态量和输入量，再对此区间开始到t积分
    if (t <= dt_)
    {
      state = predictState_.front();  //这边如下积分一次会更好
      input = predictInput_.front();
      return;
    }
    int horizon = std::floor(t / dt_);
    double dt = t - horizon * dt_;
    state = predictState_[horizon - 1];
    input = predictInput_[horizon - 1];
    state(0) = state(0) + dt * state(1) + 0.5 * dt * dt * state(2) + dt * dt * dt * input(0) / 6;
    state(1) = state(1) + dt * state(2) + 0.5 * dt * dt * input(0);
    state(2) = state(2) + dt * input(0);

    state(3) = state(3) + dt * state(4) + 0.5 * dt * dt * state(5) + dt * dt * dt * input(1) / 6;
    state(4) = state(4) + dt * state(5) + 0.5 * dt * dt * input(1);
    state(5) = state(5) + dt * input(1);

    state(6) = state(6) + dt * state(7) + 0.5 * dt * dt * state(8) + dt * dt * dt * input(2) / 6;
    state(7) = state(7) + dt * state(8) + 0.5 * dt * dt * input(2);
    state(8) = state(8) + dt * input(2);
  }

  // visualization
  void visualization()
  {
    nav_msgs::Path msg;
    msg.header.frame_id = "map";
    msg.header.stamp = ros::Time::now();
    geometry_msgs::PoseStamped p;
    for (double s = 0; s < s_.arcL(); s += 0.01)  //对样条曲线以0.01为步长 取点离散化 作为path发布
    {
      p.pose.position.x = s_(s).x();
      p.pose.position.y = s_(s).y();
      p.pose.position.z = 0.0;
      msg.poses.push_back(p);
    }
    ref_pub_.publish(msg);
    msg.poses.clear();
    for (int i = 0; i < N_; ++i)  //将预测的车体位置可视化
    {
      p.pose.position.x = predictState_[i](0);
      p.pose.position.y = predictState_[i](3);
      p.pose.position.z = 0.0;
      msg.poses.push_back(p);
    }
    traj_pub_.publish(msg);
    msg.poses.clear();
    VectorX x0_delay = x0_observe_;
    double dt = 0.001;
    for (double t = delay_; t > 0; t -= dt)
    {
      int i = std::ceil(t / dt_);
      VectorU input = historyInput_[history_length_ - i];
      step(x0_delay, input, dt);
      p.pose.position.x = x0_delay(0);
      p.pose.position.y = x0_delay(3);
      p.pose.position.z = 0.0;
      msg.poses.push_back(p);
    }
    traj_delay_pub_.publish(msg);
    msg.poses.clear();
    for (int i = 0; i < N_; ++i)
    {
      double s = s0_[i];
      p.pose.position.x = s_(s).x();
      p.pose.position.y = s_(s).y();
      p.pose.position.z = 0.0;
      msg.poses.push_back(p);
    }
    ref_point_pub_.publish(msg);
  }
};

}  // namespace mpc_car