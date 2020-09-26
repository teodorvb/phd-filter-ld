#include "flimp_tools.hpp"
#include <track-select-msmm>
#include <track-select>

namespace flimp_tools {

  float start_from_zero(const MSMMDataModel::Track& extrapolated, MSMMProject::Project& project) {
    std::vector<MSMMDataModel::ImType> extr_int;

    for (MSMMDataModel::FrameId fid = extrapolated.first_frameid(); fid < extrapolated.first_non_interpolated_frameid(); fid++)
      extr_int.push_back(extrapolated[fid].counts.at(project.get_channel_config().getId(0)));

    if (extr_int.size() > 0) {
      CppNumUtil::Statistics<float> extr_int_stat(extr_int);
      return extr_int_stat.mean()/extr_int_stat.std_error_in_mean();
    }


    return 0;
  }

  float photobleach_to_zero(const MSMMDataModel::Track& extrapolated, MSMMProject::Project& project) {
    std::vector<MSMMDataModel::ImType> extr_int;

    for (MSMMDataModel::FrameId fid = extrapolated.last_non_interpolated_frameid() + 1; fid < extrapolated.last_frameid(); fid++)
      extr_int.push_back(extrapolated[fid].counts.at(project.get_channel_config().getId(0)));

    if (extr_int.size() > 0) {
      CppNumUtil::Statistics<float> extr_int_stat(extr_int);
      return extr_int_stat.mean()/extr_int_stat.std_error_in_mean();
    }

    return 0;
  }

  typedef std::vector<std::pair<MSMMDataModel::Level, MSMMDataModel::FrameRange> > ConnectedLevels;
  class ExtractError : public std::runtime_error {
  public:
    explicit ExtractError()
      : runtime_error("not specified") {}
    explicit ExtractError(const std::string& str)
      : runtime_error(str) {}
    explicit ExtractError(const char* str)
      : runtime_error(str) {}
  };


  track_select::msmm::CustomRange
  trackIntersect(unsigned int first_frame, unsigned int last_frame, unsigned int step,
		 const std::multiset<track_select::msmm::CustomRange>& gd_ranges) {

    track_select::msmm::CustomRange range(first_frame, last_frame);

    for (const auto& gdr : gd_ranges) {
      track_select::msmm::CustomRange intersection = gdr.intersect(range);
      if (intersection.length() > 20 && intersection.contains(step))
	return intersection;
    }

    throw ExtractError("GDR intersection not found");
  }


  ConnectedLevels extractLevelSegments(const std::vector<MSMMDataModel::Level>& levels) {
    ConnectedLevels range_levels;

    for (std::vector<MSMMDataModel::Level>::const_iterator level_it = levels.begin();
         level_it != levels.end();
         level_it++)
      for (MSMMDataModel::FrameRanges::const_iterator range_it = level_it->get_ranges().begin();
           range_it != level_it->get_ranges().end();
           range_it++)
        range_levels.push_back(std::make_pair(*level_it, *range_it));

    return range_levels;
  }

  ConnectedLevels::iterator extractTwoSegments(ConnectedLevels::iterator track_end, ConnectedLevels::iterator end) {
    std::set<MSMMDataModel::ImType> num_levels;

    for (;track_end != end; track_end++) {
      num_levels.insert(track_end->first.mean_counts.begin()->second);
      if (num_levels.size() > 2)  break;
    }

    if (num_levels.size() < 2) // track needs to have at least two levels
      throw ExtractError("track less than 2 levels in ok ranges");

    track_end--;
    return track_end;
  } // extractTwoSegments

  std::tuple <Eigen::VectorXf,Eigen::VectorXf,Eigen::VectorXf, Eigen::VectorXf,Eigen::VectorXf,Eigen::VectorXf>
  extractData(const MSMMDataModel::Track& track,
	      const std::vector<track_select::msmm::CustomRange>::iterator lvl1_begin, const std::vector<track_select::msmm::CustomRange>::iterator lvl1_end,
	      const std::vector<track_select::msmm::CustomRange>::iterator lvl2_begin, const std::vector<track_select::msmm::CustomRange>::iterator lvl2_end) {
    unsigned int size1 = 0;
    unsigned int size2 = 0;

    for (auto it = lvl1_begin; it != lvl1_end; it++) size1 += it->length();
    for (auto it = lvl2_begin; it != lvl2_end; it++) size2 += it->length();

    unsigned int index;


    Eigen::VectorXf I1(size1);
    Eigen::VectorXf x1(size1);
    Eigen::VectorXf y1(size1);

    index = 0;
    for (auto it = lvl1_begin; it != lvl1_end; it++) {
      for (unsigned int i = it->low(); i <= it->high(); i++) {
	I1(index) = track[i].counts.begin()->second; // There is only one channel
	x1(index) = track[i].getX();
	y1(index) = track[i].getY();
	index++;
      }	
    }

    Eigen::VectorXf I2(size2);
    Eigen::VectorXf x2(size2);
    Eigen::VectorXf y2(size2);

    index = 0;
    for (auto it = lvl2_begin; it != lvl2_end; it++) {
      for (unsigned int i = it->low(); i <= it->high(); i++) {
	I2(index) = track[i].counts.begin()->second; // There is only one channel
	x2(index) = track[i].getX();
	y2(index) = track[i].getY();
	index++;
      }	
    }

    return std::make_tuple(I1, x1, y1, I2, x2, y2);
  }

  void splitLevels(const ConnectedLevels::iterator begin, const ConnectedLevels::iterator end,
		   ConnectedLevels::iterator& lvl1_begin, ConnectedLevels::iterator& lvl1_end,
		   ConnectedLevels::iterator& lvl2_begin, ConnectedLevels::iterator& lvl2_end) {
    float low_intensity = begin->first.mean_counts.begin()->second;
    float high_intensity = begin->first.mean_counts.begin()->second;

    lvl1_begin = begin;
    lvl2_end = end-1;

    for (auto it = begin; it != end; it++)
      if ( (high_intensity = it->first.mean_counts.begin()->second) != low_intensity) {
	lvl1_end = it-1;
	lvl2_begin = it;
	break;
      }

    if (high_intensity < low_intensity) throw ExtractError("Track Structure Error");
  }

  std::tuple <Eigen::VectorXf,Eigen::VectorXf,Eigen::VectorXf, Eigen::VectorXf,Eigen::VectorXf,Eigen::VectorXf>
  extract_last_two_levels(const MSMMDataModel::Track& track,
			  const std::multiset<track_select::msmm::CustomRange>& gd_ranges) {

    ConnectedLevels range_levels = extractLevelSegments(track.get_levels());
    std::sort(range_levels.begin(), range_levels.end(),
              [](const std::pair<MSMMDataModel::Level, MSMMDataModel::FrameRange>& l,
                 const std::pair<MSMMDataModel::Level, MSMMDataModel::FrameRange>& r) -> bool {
                return r.second < l.second;
              }); // Sort in descending order (Latest First)

    ConnectedLevels::iterator track_end = extractTwoSegments(range_levels.begin(), range_levels.end()); 

    ConnectedLevels::iterator lvl1_begin, lvl1_end, lvl2_begin, lvl2_end;
    splitLevels(range_levels.begin(), track_end+1, lvl1_begin, lvl1_end, lvl2_begin, lvl2_end);

    std::vector<track_select::msmm::CustomRange> lvl1_ranges;
    std::vector<track_select::msmm::CustomRange> lvl2_ranges;

    track_select::msmm::CustomRange range = trackIntersect(lvl2_end->second.min(), lvl1_begin->second.max(), lvl1_end->second.min(), gd_ranges);

    for (auto it = lvl1_begin; it != lvl1_end+1; it++) {
      track_select::msmm::CustomRange i = range.intersect(track_select::msmm::CustomRange(it->second.min(), it->second.max()));
      if (i.length() > 3) lvl1_ranges.push_back(i);
    }

    for (auto it = lvl2_begin; it != lvl2_end+1; it++) {
      track_select::msmm::CustomRange i = range.intersect(track_select::msmm::CustomRange(it->second.min(), it->second.max()));
      if (i.length() > 3) lvl2_ranges.push_back(i);
    }


    return extractData(track,
		       lvl1_ranges.begin(), lvl1_ranges.end(),
		       lvl2_ranges.begin(), lvl2_ranges.end());			    			    
  } // extract_last_two_levels


  std::tuple <Eigen::VectorXf,Eigen::VectorXf,Eigen::VectorXf, Eigen::VectorXf,Eigen::VectorXf,Eigen::VectorXf>
  extract_first_two_levels(const MSMMDataModel::Track& track,
			   const std::multiset<track_select::msmm::CustomRange>& gd_ranges) {

    ConnectedLevels range_levels = extractLevelSegments(track.get_levels());

    std::sort(range_levels.begin(), range_levels.end(),
              [](const std::pair<MSMMDataModel::Level, MSMMDataModel::FrameRange>& l,
                 const std::pair<MSMMDataModel::Level, MSMMDataModel::FrameRange>& r) -> bool {
                return !(r.second < l.second);
              });  // Sort in ascending order (Earliest First)

    
    ConnectedLevels::iterator track_end = extractTwoSegments(range_levels.begin(), range_levels.end());

    ConnectedLevels::iterator lvl1_begin, lvl1_end, lvl2_begin, lvl2_end;
    splitLevels(range_levels.begin(), track_end+1, lvl1_begin, lvl1_end, lvl2_begin, lvl2_end);

    std::vector<track_select::msmm::CustomRange> lvl1_ranges;
    std::vector<track_select::msmm::CustomRange> lvl2_ranges;

    track_select::msmm::CustomRange range;

    range = trackIntersect(lvl1_begin->second.min(), lvl2_end->second.max(), lvl1_end->second.max(), gd_ranges);

    for (auto it = lvl1_begin; it != lvl1_end+1; it++) {
      track_select::msmm::CustomRange i = range.intersect(track_select::msmm::CustomRange(it->second.min(), it->second.max()));
      if (i.length() > 3) lvl1_ranges.push_back(i);
    }

    for (auto it = lvl2_begin; it != lvl2_end+1; it++) {
      track_select::msmm::CustomRange i = range.intersect(track_select::msmm::CustomRange(it->second.min(), it->second.max()));
      if (i.length() > 3) lvl2_ranges.push_back(i);
    }

    return extractData(track,
		       lvl1_ranges.begin(), lvl1_ranges.end(),
		       lvl2_ranges.begin(), lvl2_ranges.end()); 
  } //  extract_first_two_levels



  float semP12(const Eigen::VectorXf& x1, const Eigen::VectorXf& x2, const Eigen::VectorXf& y1, const Eigen::VectorXf& y2) {
    float xdiff = x1.mean() - x2.mean();
    float ydiff = y1.mean() - y2.mean();
    float diff = std::sqrt(std::pow(xdiff, 2) + std::pow(ydiff, 2));

    return std::sqrt(std::pow(xdiff, 2)*(std::pow(eigen_sem(x1), 2) + std::pow(eigen_sem(x2), 2)) +
		     std::pow(ydiff, 2)*(std::pow(eigen_sem(y1), 2) + std::pow(eigen_sem(y2), 2)))/diff;
  }

  float posStd(const Eigen::VectorXf& x, const Eigen::VectorXf& y) {
    Eigen::VectorXf ones = Eigen::VectorXf::Ones(x.size());
    return std::sqrt(((x - ones*x.mean()).cwiseAbs2() + (y - ones*y.mean()).cwiseAbs2()).sum()/(x.size() - 1));
  }

  float posSem(const Eigen::VectorXf& x, const Eigen::VectorXf& y) {
    return posStd(x, y)/std::sqrt(x.size() - 1);
  }


  bool checkCiSep_net_decode(const track_select::Vector& vec) {
    return vec(0) > vec(1);
  }

  bool checkCISep(const std::tuple<Eigen::VectorXf,Eigen::VectorXf,Eigen::VectorXf, Eigen::VectorXf,Eigen::VectorXf,Eigen::VectorXf>& track_data) {
    const Eigen::VectorXf& I1 = std::get<0>(track_data);
    const Eigen::VectorXf& x1 = std::get<1>(track_data);
    const Eigen::VectorXf& y1 = std::get<2>(track_data);
    const Eigen::VectorXf& I2 = std::get<3>(track_data);
    const Eigen::VectorXf& x2 = std::get<4>(track_data);
    const Eigen::VectorXf& y2 = std::get<5>(track_data);
    
    if (I2.size() == 0 or I1.size() == 0)
        return false;

    if (std::abs(I2.mean()/I1.mean() - 2.0) > filter_level_ratio_threshold) // stepratio_lvl12
      return false;


    track_select::Vector predictors(16);

    predictors(0) = semP12(x1, x2, y1, y2); // sem_p12
    predictors(1) = posSem(x1, y1);  // sem_p1
    predictors(2) = posSem(x2, y2);  // sem_p2
    predictors(3) = posStd(x1, y1);  // std_p1
    predictors(4) = posStd(x2, y2);  // std_p2
    predictors(5) = std::sqrt(std::pow(x1.mean() - x2.mean(), 2) + std::pow(y1.mean() - y2.mean(), 2)); // dr12
    predictors(6) = (eigen_std(I1) + eigen_std(I2))/std::abs(I1.mean() - I2.mean()); // clustering
    predictors(7) = eigen_sem(I1);  // sem_i1
    predictors(8) = eigen_sem(I2);  // sem_i2
    predictors(9) = eigen_std(I1);  // std_i1
    predictors(10) = eigen_std(I2);  // std_i2
    predictors(11) = I1.size();  // len1
    predictors(12) = I2.size();  // len2
    predictors(13) = I1.mean();  // mean_i1
    predictors(14) = I2.mean();  // mean_i2
    predictors(15) = predictors(7)/predictors(13) + predictors(8)/predictors(14); // noise_sig

    boost::filesystem::path INSTALLED_RESOURCES_DIR(MSMM_INSTALLED_RESOURCES_DIR);
    if (getenv("MSMM_INSTALLED_RESOURCES_DIR")) {
        INSTALLED_RESOURCES_DIR=getenv("MSMM_INSTALLED_RESOURCES_DIR");
    }
    static track_select::nn::FeedForwardNetwork
        net = track_select::nn::FeedForwardNetwork::readHDF5((INSTALLED_RESOURCES_DIR/"flimp"/"sepBoundary.h5").string());

    if (predictors(5) > filter_separation_threshold)
      return false; // dr12

    return checkCiSep_net_decode(net(predictors));
  }

  void checkTrack(MSMMDataModel::Track& track,
                  MSMMProject::Project& project,
                  unsigned int min_length,
                  unsigned int max_frame_id,
		  const std::multiset<track_select::msmm::CustomRange>& gd_ranges) {


    if (track.size(true) < min_length) throw ExtractError("track is too short");
    if (track.last_non_interpolated_frameid() - track.first_non_interpolated_frameid() < 4) throw ExtractError("Track has no non-interpolated points");

    const MSMMDataModel::ImFrameStack& frames = project.stacks().proc_frames();
    const MSMMDataModel::ChannelConfig& cc = project.get_channel_config();

    track.generate_levels_check_bg(cc, frames);

    
    bool last_two_ok = false;
    bool first_two_ok = false;

    try {
      last_two_ok = (track.last_frameid() < max_frame_id - 3) && checkCISep(extract_last_two_levels(track, gd_ranges));
    } catch (std::runtime_error e) {
      last_two_ok = false;
    }

    try {
      first_two_ok = track.first_frameid() > 3 && checkCISep(extract_first_two_levels(track, gd_ranges));
    } catch (std::runtime_error e) {
      first_two_ok = false;
    }

    MSMMDataModel::Track extrapolated = track;
    bool keepgoing = true;
    MSMMTracking::ExtrapolateTrack(extrapolated,
                                   project.stacks().proc_frames(),
                                   project.get_options().spot_fwhm,
                                   project.get_options().maskw,
                                   project.get_options().nbar,
                                   NULL,
                                   keepgoing,
                                   10,
                                   project.should_i_separate_channels());

    bool zero_after = std::abs(photobleach_to_zero(extrapolated, project)) < 3;
    bool zero_before = std::abs(start_from_zero(extrapolated, project)) < 3;

    if (!((first_two_ok && zero_before) || (last_two_ok && zero_after))) throw ExtractError("score error");
  }


  void calc_flimp_score(MSMMProject::Project& project,
                        const std::multiset<track_select::msmm::CustomRange>& gd_ranges,
                        unsigned int min_length) {

    unsigned int max_frame_id = project.get_FrameId_range().second; // this is super slow - 0.01 sec.

    for (auto track_it = project.get_tracks().begin();
         track_it!= project.get_tracks().end();
         track_it++) {

      try {
        checkTrack(track_it->second, project, min_length, max_frame_id, gd_ranges);
        track_it->second.set_score("flimp_filter_score", 1);
      } catch (ExtractError e) {
        track_it->second.set_score("flimp_filter_score", -1);
      }

    } // end track loop
  }

} // flimp_tools
