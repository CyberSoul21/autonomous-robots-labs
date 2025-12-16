
#include <cmath>
#include <string>
#include <memory>
#include "nav2_util/node_utils.hpp"
#include "nav2_costmap_2d/cost_values.hpp"

#include "rrt_global_planner/rrt_global_planner.hpp"

# define random (rand() / ((double)RAND_MAX + 1.0))

namespace rrt_global_planner
{

double distance(const unsigned int x0, const unsigned int y0, const unsigned int x1, const unsigned int y1){
    return std::sqrt((int)(x1-x0)*(int)(x1-x0) + (int)(y1-y0)*(int)(y1-y0));
}



void RRTPlanner::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent.lock(); 
  name_ = name;
  tf_ = tf;
  costmap_ = costmap_ros->getCostmap();
  global_frame_ = costmap_ros->getGlobalFrameID();

  // Parameter initialization
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".max_samples", rclcpp::ParameterValue(
      30000.0));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".dist_th", rclcpp::ParameterValue(
      0.2));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".max_dist", rclcpp::ParameterValue(
      1.0));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".resolution", rclcpp::ParameterValue(
      0.05));    

  node_->get_parameter(name_ + ".max_samples", max_samples_);
  node_->get_parameter(name_ + ".dist_th", dist_th_);
  node_->get_parameter(name_ + ".max_dist", max_dist_);
  node_->get_parameter(name_ + ".resolution", resolution_);

  marker_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>(
    "rrt_markers", rclcpp::SystemDefaultsQoS());
}

void RRTPlanner::cleanup()
{
  RCLCPP_INFO(
    node_->get_logger(), "CleaningUp plugin %s of type RRTPlanner",
    name_.c_str());
}

void RRTPlanner::activate()
{
  RCLCPP_INFO(
    node_->get_logger(), "Activating plugin %s of type RRTPlanner",
    name_.c_str());
    marker_pub_->on_activate();

}

void RRTPlanner::deactivate()
{
  RCLCPP_INFO(
    node_->get_logger(), "Deactivating plugin %s of type RRTPlanner",
    name_.c_str());
    marker_pub_->on_deactivate();

}

nav_msgs::msg::Path RRTPlanner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  nav_msgs::msg::Path global_path;

  std::cout << "RRTPlanner::createPlan" << std::endl;
    

  // Checking if the goal and start state is in the global frame
  if (start.header.frame_id != global_frame_) {
    RCLCPP_ERROR(
      node_->get_logger(), "Planner will only except start position from %s frame",
      global_frame_.c_str());
    return global_path;
  }

  if (goal.header.frame_id != global_frame_) {
    RCLCPP_INFO(
      node_->get_logger(), "Planner will only except goal position from %s frame",
      global_frame_.c_str());
    return global_path;
  }

  global_path.poses.clear();

  // Get start and goal poses in map coordinates
  unsigned int goal_mx, goal_my, start_mx, start_my;

  if (!costmap_->worldToMap(goal.pose.position.x, goal.pose.position.y, goal_mx, goal_my)){
    RCLCPP_INFO(
      node_->get_logger(), "Goal position is out of map bounds.");
    return global_path;
  }   
    
  costmap_->worldToMap(start.pose.position.x, start.pose.position.y, start_mx, start_my);

  global_path.header.stamp = node_->now();
  global_path.header.frame_id = global_frame_;

  std::vector<int> point_start{(int)start_mx,(int)start_my};
  std::vector<int> point_goal{(int)goal_mx,(int)goal_my};    
  std::vector<std::vector<int>> solRRT;
  bool computed = computeRRT(point_start, point_goal, solRRT);
  //computed = true;
  if (computed){   
    getPlan(solRRT, global_path);
    // Add goal to plan
    // global_path.poses.push_back(goal);
  }else{
      RCLCPP_INFO(
    node_->get_logger(), "No plan computed");
  }

  return global_path;
}

bool RRTPlanner::computeRRT(const std::vector<int> start, const std::vector<int> goal, 
                            std::vector<std::vector<int>>& sol) {

  // Initialize
  srand(time(NULL));     
  
  // Initialize the tree with the starting point in map coordinates
  TreeNode *itr_node = new TreeNode(start);
  TreeNode *goal_node;
  bool finished = false;

  double dis_max_grid = max_dist_ / resolution_;
  double dis_thres_grid = dist_th_ / resolution_;
  int samples = 0;

  // First check
  if (obstacleFree(start[0], start[1], goal[0], goal[1])) {
    sol = { goal, start };
    delete itr_node;
    return true;
  }

  while(!finished && samples < max_samples_) {
    samples += 1;

    // sample
    int x_rand = (int)((double)rand() / RAND_MAX * costmap_->getSizeInCellsX());
    int y_rand = (int)((double)rand() / RAND_MAX * costmap_->getSizeInCellsY());
    std::vector<int> rand_point = {x_rand, y_rand};

    //nearest node
    TreeNode *rand_node = new TreeNode(rand_point);
    TreeNode *near_node = rand_node->neast(itr_node);
    std::vector<int> near_point = near_node->getNode();
    
    delete rand_node;

    //steer
    std::vector<int> new_point;
    double dist = distance(near_point[0], near_point[1], rand_point[0], rand_point[1]);

    if (dist <= dis_max_grid){
      new_point = rand_point;
    } else{
      double delta_x = rand_point[0] - near_point[0];
      double delta_y = rand_point[1] - near_point[1];

      delta_x *= dis_max_grid / dist;
      delta_y *= dis_max_grid / dist;
      new_point = {(int)round(near_point[0] + delta_x), (int)round(near_point[1] + delta_y)};
    }

    // In case steer has gone out of the map
    if (new_point[0] < 0 || new_point[0] >= (int)costmap_->getSizeInCellsX() ||
        new_point[1] < 0 || new_point[1] >= (int)costmap_->getSizeInCellsY()) {
        continue; 
    }
    
    //obstacle free
    if(obstacleFree(near_point[0], near_point[1], new_point[0], new_point[1])) {
      TreeNode *new_node = new TreeNode(new_point);
      near_node->appendChild(new_node);

      // If we are close to the goal
      if (obstacleFree(new_point[0], new_point[1], goal[0], goal[1])) {
        double distance_goal_c = distance(new_point[0], new_point[1], goal[0], goal[1]);
        
        if (distance_goal_c <= dis_thres_grid) {
          goal_node = new TreeNode(goal);
          new_node->appendChild(goal_node);
          finished = true;
        }
      }
    }
  }

  if (finished) {
    sol = goal_node->returnSolution();
    // Visualization in RViz
    visualizeSolTree(start, goal, sol); 
  } else {
    RCLCPP_INFO(node_->get_logger(), "No solution after %d samples", samples);
  }

  
  delete itr_node;
  return finished;
}

bool RRTPlanner::obstacleFree(const unsigned int x0, const unsigned int y0, 
                            const unsigned int x1, const unsigned int y1){
    //Bresenham algorithm to check if the line between points (x0,y0) - (x1,y1) is free of collision

    int dx = x1 - x0;
    int dy = y1 - y0;

    int incr_x = (dx > 0) ? 1.0 : -1.0;
    int incr_y = (dy > 0) ? 1.0 : -1.0;

    unsigned int da, db, incr_x_2, incr_y_2;
    if (abs(dx) >= abs(dy)){
        da = abs(dx); db = abs(dy);
        incr_x_2 = incr_x; incr_y_2 = 0;
    }else{
        da = abs(dy); db = abs(dx);
        incr_x_2 = 0; incr_y_2 = incr_y;
    }

    int p = 2*db - da;
    unsigned int a = x0; 
    unsigned int b = y0;
    unsigned int end = da;
    for (unsigned int i=0; i<end; i++){
        if (costmap_->getCost(a, b) >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE){  //  != nav2_costmap_2d::FREE_SPACE to include cells with inflated cost
            return false;
        }else{
            if (p >= 0){
                a += incr_x;
                b += incr_y;
                p -= 2*da;
            }else{
                a += incr_x_2;
                b += incr_y_2;
            }
            p += 2*db;
        }
    }

    return true;
}

void RRTPlanner::getPlan(const std::vector<std::vector<int>> sol, nav_msgs::msg::Path& plan){

    for (auto it = sol.rbegin(); it != sol.rend(); it++){
      std::vector<int> point = (*it);
      geometry_msgs::msg::PoseStamped pose;
 
      costmap_->mapToWorld((unsigned int)point[0], (unsigned int)point[1], 
                            pose.pose.position.x, pose.pose.position.y);
      pose.pose.orientation.x = 0.0;
      pose.pose.orientation.y = 0.0;
      pose.pose.orientation.z = 0.0;
      pose.pose.orientation.w = 1.0;
      pose.header.stamp = node_->now();
      pose.header.frame_id = global_frame_;
      plan.poses.push_back(pose);                            
    }
}

void RRTPlanner::visualizeSolTree(const std::vector<int>& start, const std::vector<int>& goal,
                                    std::vector<std::vector<int>>& sol){
    // std::cout << "Visualize tree solution: " << sol.size() << std::endl;

    visualization_msgs::msg::Marker solTree;
    geometry_msgs::msg::Point marker_point;
    solTree.color.g = 0.0; solTree.color.r = 0.0; solTree.color.b = 1.0; solTree.color.a = 1.0;
    solTree.scale.x = 0.1;
    fillMarkerMsg(global_frame_, "tree", 3, visualization_msgs::msg::Marker::LINE_LIST, solTree);
    
    //include start point to include line between start and first point in the tree
    if (sol.size() != 0){
        costmap_->mapToWorld(start[0], start[1], marker_point.x, marker_point.y);
        solTree.points.push_back(marker_point);
    }
    for (int i = sol.size()-1; i>=0; i--){  // inverse order
        if (i == sol.size() - 1){
            costmap_->mapToWorld(sol[i][0], sol[i][1], marker_point.x, marker_point.y);
            solTree.points.push_back(marker_point);
        }else{
            // include each pair of points to draw a line
            costmap_->mapToWorld(sol[i+1][0], sol[i+1][1], marker_point.x, marker_point.y);
            solTree.points.push_back(marker_point);
            costmap_->mapToWorld(sol[i][0], sol[i][1], marker_point.x, marker_point.y);
            solTree.points.push_back(marker_point);
        }
    }
    // include line between last point in the tree and the goal
    if (sol.size() != 0){
        int i = sol.size()-1;
        costmap_->mapToWorld(sol[0][0], sol[0][1], marker_point.x, marker_point.y);
        solTree.points.push_back(marker_point);
        costmap_->mapToWorld(goal[0], goal[1], marker_point.x, marker_point.y);
        solTree.points.push_back(marker_point);
    }

    marker_pub_->publish(solTree);

}

void RRTPlanner::fillMarkerMsg(const std::string frame_id, const std::string ns, const int id,
                    const int type, visualization_msgs::msg::Marker& msg){
    
    msg.header.frame_id = frame_id;
    msg.header.stamp = node_->now();
    msg.ns = ns;
    msg.action = visualization_msgs::msg::Marker::ADD;
    msg.pose.orientation.w = 1.0;
    msg.id = id;
    msg.type = type;
}

}  // namespace rrt_global_planner

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(rrt_global_planner::RRTPlanner, nav2_core::GlobalPlanner)
