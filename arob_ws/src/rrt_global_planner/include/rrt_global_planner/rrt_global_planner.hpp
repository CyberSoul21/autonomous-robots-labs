#ifndef RRT_PLANNER_CPP
#define RRT_PLANNER_CPP

#include <string>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "nav2_core/global_planner.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav2_util/robot_utils.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"

#include "rclcpp_lifecycle/lifecycle_publisher.hpp"
#include "visualization_msgs/msg/marker.hpp"

#include "rrt_global_planner/TreeNode.hpp"

namespace rrt_global_planner
{

class RRTPlanner : public nav2_core::GlobalPlanner
{
public:
  RRTPlanner() = default;
  ~RRTPlanner() = default;

  // plugin configure
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  // plugin cleanup
  void cleanup() override;

  // plugin activate
  void activate() override;

  // plugin deactivate
  void deactivate() override;

  // This method creates path for given start and goal pose.
  nav_msgs::msg::Path createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) override;

private:
  // TF buffer
  std::shared_ptr<tf2_ros::Buffer> tf_;

  // node ptr
  nav2_util::LifecycleNode::SharedPtr node_;

  // Global Costmap
  nav2_costmap_2d::Costmap2D * costmap_;

  // Global frame of the costmap
  std::string global_frame_, name_;

  double max_samples_;
  double dist_th_;
  double max_dist_;
  double resolution_;

  rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;


  // Functions to compute the plan
  bool obstacleFree(const unsigned int x0, const unsigned int y0, const unsigned int x1, const unsigned int y1);
  bool computeRRT(const std::vector<int> start, const std::vector<int> goal, 
                            std::vector<std::vector<int>>& sol);
 void getPlan(const std::vector<std::vector<int>> sol, nav_msgs::msg::Path& plan);

  //Functions for visualization in RViz
  void visualizeSolTree(const std::vector<int>& start, const std::vector<int>& goal,
                                    std::vector<std::vector<int>>& sol);
  void fillMarkerMsg(const std::string frame_id, const std::string ns, const int id,
                    const int type, visualization_msgs::msg::Marker& msg);

};

}  // namespace rrt_global_planner

#endif  
