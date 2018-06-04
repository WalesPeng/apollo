/******************************************************************************
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

// @brief: lane detection on a single image frame

#ifndef MODULES_PERCEPTION_OBSTACLE_CAMERA_CC_LANE_POST_PROCESSOR_FRAME_H_
#define MODULES_PERCEPTION_OBSTACLE_CAMERA_CC_LANE_POST_PROCESSOR_FRAME_H_

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "opencv2/opencv.hpp"

#include "modules/common/log.h"
#include "modules/perception/obstacle/camera/lane_post_process/cc_lane_post_processor/group.h"
#include "modules/perception/obstacle/camera/lane_post_process/common/connected_component.h"
#include "modules/perception/obstacle/camera/lane_post_process/common/projector.h"
#include "modules/perception/obstacle/camera/lane_post_process/common/type.h"
#include "modules/perception/obstacle/camera/lane_post_process/common/util.h"

namespace apollo {
namespace perception {

struct LaneFrameOptions {
  // used for initialization
  SpaceType space_type = SpaceType::IMAGE;  // space type
  cv::Rect image_roi;
  bool use_cc = true;
  int min_cc_pixel_num = 10;  // minimum number of pixels of CC    PMH ：连通域最小像素个数
  int min_cc_size = 5;        // minimum size of CC                连通域最小尺寸

  bool use_non_mask = false;  // indicating whether use non_mask or not      是否使用 non_mask？      

  // used for greedy search association method                    贪婪搜索关联方法相关参数
  // maximum number of markers used for matching for each CC
  int max_cc_marker_match_num = 1;                  //用于匹配每个CC的最大 markers 数

  // used for marker association
  // minimum longitudinal offset used for
  // search upper markers
  ScalarType min_y_search_offset = 0.0;             // PMH：y方向最小纵向搜索范围

  AssociationParam assoc_param;
  GroupParam group_param;
  int orientation_estimation_skip_marker_num = 1;

  // for determining lane object label
  // predefined label interval distance
  ScalarType lane_interval_distance = 4.0;       	// PMH：车道间隔距离 = 4m

  // for fitting curve
  // minimum size of lane instance in meter to
  // be prefiltered
  ScalarType min_instance_size_prefiltered = 0.5;          // PMH：拟合得到曲线的最小长度大于0.5米

  // maximum size of instance to fit
  // a straight line
  ScalarType max_size_to_fit_straight_line = 4.0;         // PMH：拟合直线的最大长度
};

class LaneFrame {
 public:
  bool Init(const std::vector<ConnectedComponentPtr>& input_cc,
            const std::shared_ptr<NonMask>& non_mask,
            const LaneFrameOptions& options,
            const double scale,
            const int start_y_pos);

  bool Init(const std::vector<ConnectedComponentPtr>& input_cc,
            const std::shared_ptr<NonMask>& non_mask,
            const std::shared_ptr<Projector<ScalarType>>& projector,
            const LaneFrameOptions& options,
            const double scale,
            const int start_y_pos);

  void SetTransformer(const std::shared_ptr<Projector<ScalarType>>& projector) {
    projector_ = projector;												// PMH：
    is_projector_init_ = true;
  }

  bool Process(LaneInstancesPtr instances);

  int MarkerNum() const { return static_cast<int>(markers_.size()); }

  bool IsValidMarker(int i) { return i >= 0 && i < MarkerNum(); }

  const Marker* marker(int i) { return &(markers_.at(i)); }

  int GraphNum() const { return static_cast<int>(graphs_.size()); }

  const Graph* graph(int i) { return &(graphs_.at(i)); }

  Bbox bbox(int i) const { return boxes_.at(i); }

  bool FitPolyCurve(const int& graph_id, const ScalarType& graph_siz,             // PMH：根据graph计算多项式曲线系数和横向距离。
                    PolyModel* poly_coef, ScalarType* lateral_distance) const;

 protected:
  ScalarType ComputeMarkerPairDistance(const Marker& ref, const Marker& tar);          // PMH：计算Marker对的距离

  std::vector<int> ComputeMarkerEdges(                           // PMH：计算Marker边缘
      std::vector<std::unordered_map<int, ScalarType>>* edges);

  bool GreedyGroupConnectAssociation();

  int AddGroupIntoGraph(const Group& group, Graph* graph,
                        std::unordered_set<int>* hash_marker_idx);

  int AddGroupIntoGraph(const Group& group, const int& start_marker_ascend_id,
                        const int& end_marker_descend_id, Graph* graph,
                        std::unordered_set<int>* hash_marker_idx);

  void ComputeBbox();

 private:
  // options
  LaneFrameOptions opts_;

  // markers
  std::vector<Marker> markers_;
  int max_cc_num_ = 0;

  // CC index for each marker
  std::vector<int> cc_idx_;
  // marker indices for each CC
  std::unordered_map<int, std::vector<int>> cc_marker_lut_;

  std::shared_ptr<const Projector<ScalarType>> projector_;
  bool is_projector_init_ = false;

  // lane marker clusters
  std::vector<Graph> graphs_;       // 相邻marker_id号构成marker二元组
  // tight bounding boxes of lane clusters
  std::vector<Bbox> boxes_;      // 每个graph的水平包围盒。
  double scale_;
  double start_y_pos_;
};

}  // namespace perception
}  // namespace apollo

#endif  // MODULES_PERCEPTION_OBSTACLE_CAMERA_CC_LANE_POST_PROCESSOR_FRAME_H_
