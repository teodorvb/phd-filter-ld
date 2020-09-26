#include "flimp_tools.hpp"
#include "kolmogorov_smirnov.hpp"
#include <track-select>

using namespace MSMMDataModel;
using namespace flimp_tools;
namespace level_detect {
  typedef std::vector<std::pair<int,int> > LevelSegments;
  std::ostream& operator<<(std::ostream& out, const LevelSegments& sgm) {
    for (auto& s : sgm)
      out << "(" << s.first << " " << s.second << ") ";

    return out;
  }

  track_select::Vector eigen_whiten(track_select::Vector vec) {
    return (vec - track_select::Vector::Ones(vec.size())*vec.mean())/eigen_std(vec);
  }


  template<class T>
  bool has_gradient(T vec) {
    unsigned int d_size = vec.size();
    track_select::Matrix data(d_size, 2);
    data.col(0) = track_select::Vector::LinSpaced(d_size, 1, d_size);
    data.col(1) = track_select::Vector(vec);

    track_select::Matrix w = data.rowwise() - data.colwise().mean(); // center the data

    track_select::Matrix covar = w.transpose() * w;

    track_select::real slope = covar(1, 2)/covar(1, 1);
    track_select::real bias = data.col(1).mean() - slope * data.col(0).mean();

    track_select::real std_err =
      std::sqrt((data.col(1) - track_select::Vector((data.col(0)* slope).array() + bias)).squaredNorm()/
		(track_select::Vector(data.col(0).array() - data.col(0).mean()).squaredNorm() * (d_size - 1)) );

    return std::abs(slope)/std_err > 2;
  } // has_gradient



  float kstest_wrapper(const track_select::Vector &data1, const track_select::Vector &data2) {
    std::deque<float> data_1(data1.data(), data1.data()+data1.size());
    std::deque<float> data_2(data2.data(), data2.data()+data2.size());

    return CppNumUtil::kstestmodifiesdata(data_1.begin(), data_1.end(), data_2.begin(), data_2.end());
  }
  float kstest_wrapper(const track_select::Vector &data1) {
    std::deque<float> data_1(data1.data(), data1.data()+data1.size());
    return CppNumUtil::kstestmodifiesdata(data_1.begin(), data_1.end(), CppNumUtil::std_normal_cdf<float>);
  }


  LevelSegments detect(const track_select::Vector& seq) {
    float tr = 0.3;
    LevelSegments segments;
    size_t N = seq.size();


    size_t win_N = size_t(std::floor(float(N)/win_size));

    if (win_N == 1) {
      segments.push_back(std::make_pair(0, seq.size()-1));
      return segments;
    }

    track_select::Vector wins(win_N);

    for (unsigned int i = 0; i < win_N; i++)
      wins(i) = eigen_std(seq.segment(i*win_size, win_size));

    
    bool level_on = false;
    unsigned int ls = 0;

    for (unsigned int i = 1; i < win_N; i++) {
      if (std::abs(wins.segment(ls, i - ls + 1).mean()/eigen_std(seq.segment(ls*win_size, (i -ls + 1)*win_size)) - 1) < tr &&
          std::abs(wins.segment(i-1, 2).mean()/eigen_std(seq.segment((i-1)*win_size,  2 *win_size)) - 1) < tr &&
          kstest_wrapper(eigen_whiten(seq.segment(ls*win_size, (i - ls+1)*win_size))) > 0.01) {

        level_on = true;
      } else if (level_on) {
        segments.push_back(std::make_pair(ls*win_size, (i) * win_size -1));
        level_on = 0;
        ls = i;
      } else {
        ls++;
      } // endif

    } // endfor

    if (level_on) {
      segments.push_back(std::make_pair(ls*win_size, win_N*win_size -1));
    }


    return segments;
  }


  LevelSegments merge(const LevelSegments& sgm, const track_select::Vector& seq) {
    int sgm_N = sgm.size();
    if (!sgm_N) return sgm;

    int merge_N = sgm_N - 1;
    track_select::Vector merge = track_select::Vector::Zero(merge_N);
    for (int i = 0; i < merge_N; i++)
      merge(i) = ((sgm[i+1].first - sgm[i].second) < 2) &&
        (kstest_wrapper(seq.segment(sgm[i].first, sgm[i].second - sgm[i].first + 1),
                        seq.segment(sgm[i+1].first, sgm[i+1].second - sgm[i+1].first + 1)) > 0.05);

    LevelSegments merged_segments(sgm_N - merge.sum());
    int merged_index = 0;

    int sgm_index = 0;
    bool merging = false;
    int merge_start = -1;
    for (int i = 0; i < merge_N; i++) {
      if (merge(i) && merging) {
        sgm_index++;

      } else if (merge(i) && !merging) {
        merging = true;
        merge_start = sgm_index;
        sgm_index++;

      } else if (!merge(i) && merging) {
        merging = false;
        merged_segments[merged_index] = std::make_pair(sgm[merge_start].first, sgm[sgm_index].second);
        sgm_index++;
        merged_index++;

      } else if (!merge(i) && !merging) {
        merged_segments[merged_index] = sgm[sgm_index];
        merged_index++;
        sgm_index++;
      } // endif

    } // endfor
    if (merging) {
      merged_segments[merged_index] = std::make_pair(sgm[merge_start].first, sgm[sgm_N-1].second);
    } else {
      merged_segments[merged_index] = sgm[sgm_N -1];
    }

    return merged_segments;
  } // segment_merge

  LevelSegments extend(const LevelSegments& sgm, const track_select::Vector& seq) {

    LevelSegments extended_segments;
    int seq_N = seq.size();
    int sgm_N = sgm.size();

    for (int i = 0; i < sgm_N; i++) {
      track_select::Vector segment = seq.segment(sgm[i].first, sgm[i].second - sgm[i].first + 1);
      float m = eigen_median(segment);
      float s = eigen_std(segment)/segment.size(); //
      std::pair<int, int> extended_sgm = sgm[i];

      if (sgm[i].second < seq_N-1) { // if can extend on the right try to extend
        int low = (i < sgm_N - 1) ?
          std::min(std::min(sgm[i].second + win_size -1, seq_N - 1), sgm[i+1].first - 1) :
          std::min(sgm[i].second + win_size -1, seq_N - 1);

        for (int j = sgm[i].second + 1; j < low + 1; j++)
          if (std::abs(seq(j) - m) < 2*s)
            extended_sgm.second++;

      } // endif

      if (sgm[i].first > 0) { // if can extend on the left try to extend
        int high = (i > 0) ?
          std::max(std::max(sgm[i].first - win_size +1, 0), extended_segments[i-1].second + 1) :
          std::max(sgm[i].first - win_size + 1, 0);

        for (int j = sgm[i].first - 1; j >= high; j--)
          if (std::abs(seq(j) - m) < 1.5*s)
            extended_sgm.first--;
      } // endif

      extended_segments.push_back(extended_sgm);
    } // endfor

    return extended_segments;
  } // segment_extend


  LevelSegments split(const LevelSegments& sgm, const track_select::Vector& seq) {
    LevelSegments split_segments;
    int sgm_N = sgm.size();

    for (int i = 0; i < sgm_N; i++) {
      track_select::Vector segment = seq.segment(sgm[i].first, sgm[i].second - sgm[i].first + 1);

      float m = eigen_median(segment);
      float s = eigen_std(segment);

      int sgm_start = 0;
      bool sgm_on = false;

      for (int j = 0; j < segment.size(); j++) {
        bool outlier = std::abs(segment(j) - m) > 2*s;
        if (outlier && sgm_on) {
          split_segments.push_back(std::make_pair(sgm[i].first + sgm_start, sgm[i].first + j -1));
          sgm_on = false;

        } else if (!outlier && !sgm_on) {
          sgm_start = j;
          sgm_on = true;
        } //endif
      } // endfor
      if (sgm_on) {
        split_segments.push_back(std::make_pair(sgm[i].first + sgm_start, sgm[i].second));
      } //endif

    } // endfor

    return split_segments;
  } // segment_split


  LevelSegments remove(const LevelSegments& sgm, const track_select::Vector& seq) {
    LevelSegments deleted_levels;

    for (unsigned int i = 0; i < sgm.size(); i++) {
      track_select::Vector segment = seq.segment(sgm[i].first, sgm[i].second - sgm[i].first + 1);
      if (segment.size() > 3 && !has_gradient(segment))// && kstest_wrapper(eigen_whiten(segment)) > 0.01)
        deleted_levels.push_back(sgm[i]);

    } // endfor

    return deleted_levels;
  }

  bool detect_background_net_decode(const track_select::Vector& vec) {
    return vec(0) < vec(1);
  }


  //  typedef std::vector<std::pair<int,int> > LevelSegments;
  LevelSegments detect_background(const LevelSegments& sgm, const std::vector<track_select::Vector>& bg) {
    boost::filesystem::path INSTALLED_RESOURCES_DIR(MSMM_INSTALLED_RESOURCES_DIR);
    if (getenv("MSMM_INSTALLED_RESOURCES_DIR")) {
        INSTALLED_RESOURCES_DIR=getenv("MSMM_INSTALLED_RESOURCES_DIR");
    }
    static track_select::nn::FeedForwardNetwork net
        = track_select::nn::FeedForwardNetwork::readHDF5((INSTALLED_RESOURCES_DIR/"flimp"/"background-net.h5").string());

    LevelSegments new_segments;

    for (int segment_counter = 0; segment_counter < sgm.size(); segment_counter++) {

      int begin = sgm[segment_counter].first;
      int end = sgm[segment_counter].second;

      bool segment_open = true;
      int i = begin;
      int used_at_all = 0;
      bool used;

      for (; i <= end; i++) {
        used = detect_background_net_decode(net(bg[i]));

        used_at_all = used_at_all + used;

        if (used && segment_open) {
        } else if (used && ! segment_open) {
          begin = i;
          segment_open = true;

        } else if (! used && segment_open) {
          if (begin < i-1)
            new_segments.push_back(std::make_pair(begin, i-1));
          segment_open = false;

        } else if ( ! used && ! segment_open) {
        }

      } // endfor

      if (segment_open) {
        new_segments.push_back(std::make_pair(begin, end));
      }
    } // endwhile

    return new_segments;
  }


  std::vector<LevelSegments> create_levels(const LevelSegments& sgm, const track_select::Vector& seq) {
    //  typedef std::vector<std::pair<int,int> > LevelSegments;
    LevelSegments unmerged_sgm = sgm;
    std::vector<LevelSegments> groups;

    while (unmerged_sgm.size() > 0) {
      LevelSegments merged_sgm;
      std::vector<int> to_copy;

      merged_sgm.push_back(unmerged_sgm[0]);

      track_select::Vector level = seq.segment(merged_sgm.back().first, merged_sgm.back().second - merged_sgm.back().first + 1);

      for (int i = 1; i < unmerged_sgm.size(); i++) {
        std::pair<int, int>& sgm_range = unmerged_sgm[i];
        track_select::Vector sgm = seq.segment(sgm_range.first, sgm_range.second - sgm_range.first + 1);

        if (kstest_wrapper(level, sgm) > 0.05) {
          merged_sgm.push_back(sgm_range);

          track_select::Vector new_lvl(level.size() + sgm.size());
          new_lvl << level, sgm;
          level = new_lvl;

        } else {
          to_copy.push_back(i);
        }// endif

      } // endfor

      LevelSegments new_unmerged_sgm;
      for (auto i : to_copy)
        new_unmerged_sgm.push_back(unmerged_sgm[i]);
      unmerged_sgm = new_unmerged_sgm;

      groups.push_back(merged_sgm);
    } //endwhile

    return groups;
  } // create_levels


  track_select::Vector diff(const track_select::Vector& reg) {
    return reg.segment(1, reg.size()-1) - reg.segment(0, reg.size()-1);
  } // diff

  std::vector<std::pair<int, int> > construct_ranges(const std::vector<int>& steps, int begin, int end) {
    std::vector<std::pair<int, int> > ranges;

    int range_begin = begin;
    for (int i = 0; i < steps.size(); i++) {
      if (steps[i] - range_begin +1 >= 3)
	ranges.push_back(std::make_pair(range_begin, steps[i])); // exclude the steps
      range_begin = steps[i]+1;
    }

    if (end - begin + 1 >= 3)
      ranges.push_back(std::make_pair(range_begin, end));

    return ranges;
  } // construct_ranges


  /* *************** Find Segments Step STD *************** */
  /* Splits a track into segments by calculating the mean intensity step. Steps
   * which are more than 3 standard deviations away from the mean*/

  std::vector<int> find_step_std(const track_select::Vector& diff, int begin) {
    float m = diff.cwiseAbs().mean();
    float s = std::sqrt((diff.array() - m).cwiseAbs2().sum()/diff.size() - 1);

    std::vector<int> res;

    for (int i = 0; i < diff.size(); i++)
      if (std::abs(diff[i] - m) > 3*s) res.push_back(begin + i);

    return res;
  } // find_steps

  std::vector<std::pair<int, int> > find_segments_step_std(const track_select::Vector& reg, int begin, int end) {
    if ((end - begin) < 5)
      return {};


    std::vector<std::pair<int, int>> ranges = construct_ranges(find_step_std(diff(reg.segment(begin, end - begin + 1)), begin), begin, end);
    if (ranges.size() == 0)
      return {};
    if (ranges.size() == 1)
      return ranges;


    std::vector<std::pair<int, int> > res;
    for (auto& range : ranges)
      for (auto& new_range : find_segments_step_std(reg, range.first, range.second))
        res.push_back(new_range);

    return res;
  }

  std::vector<std::pair<int, int> > find_segments_step_std(const track_select::Vector& ref) {
    return find_segments_step_std(ref, 0, ref.size() -1);
  }

 
  /* ***************************** */


  /* *************** Find Segments Intensity STD *************** */
  /* Finds steps by comparing a step size to the sum of the error
   * in measuring the intensity (generated by Quincy)
   */

  std::vector<int> find_steps_int_std(const track_select::Vector& reg,
				      const track_select::Vector& sig_reg) {
    std::vector<int> res;
    for (int i = 0; i < reg.size()-1; i++)
      if (CppNumUtil::prob_diff_is_real(reg(i), sig_reg(i), reg(i+1), sig_reg(i+1)) > 0.99)
	res.push_back(i);

    return res;
  }

  std::vector<std::pair<int, int>> find_segments_merge(std::vector<std::pair<int, int>> ranges,
						       const track_select::Vector& reg) {

    bool merged = false;

    /* Merge similar segments */
    do {
      merged = false;

      for (int i =0; i < ranges.size()-1; i++) {
	track_select::Vector seg1 = reg.segment(ranges[i].first, ranges[i].second - ranges[i].first + 1);
	track_select::Vector seg2 = reg.segment(ranges[i+1].first, ranges[i+1].second - ranges[i+1].first + 1);
	if (seg1.size() > 2 && seg2.size() > 2 &&
	    ranges[i].second +1 == ranges[i+1].first &&
	    //	    CppNumUtil::prob_diff_is_real(seg1.mean(), eigen_std(seg1),
	    //				  seg2.mean(), eigen_std(seg2)) <= 0.99) {
	    std::abs(seg1.mean() - seg2.mean()) < (eigen_std(seg1) + eigen_std(seg2)) ) {
	  merged = true;
	  std::vector<std::pair<int, int> > new_ranges;
	  for (int j = 0; j < i; j++)
	    new_ranges.push_back(ranges[j]);
	  new_ranges.push_back(std::make_pair(ranges[i].first, ranges[i+1].second));
	  for (int j = i+2; j < ranges.size(); j++)
	    new_ranges.push_back(ranges[j]);

	  ranges = new_ranges;
	  break;
	} // endif
      } // endfor
    } while(merged);

    return ranges;
  }

  std::vector<std::pair<int, int> > find_segments_int_std(const track_select::Vector& reg,
							  const track_select::Vector& sig_reg) {

    std::vector<std::pair<int, int>> ranges = 
      find_segments_merge(construct_ranges(find_steps_int_std(reg, sig_reg), 0, reg.size() -1), reg);
    
    return ranges;
  }

  /* ***************************** */
 
  /* ******** Find segments with neural network ********** */

  bool find_steps_is_step(const track_select::Vector in) {

    static boost::filesystem::path INSTALLED_RESOURCES_DIR(MSMM_INSTALLED_RESOURCES_DIR);
    if (getenv("MSMM_INSTALLED_RESOURCES_DIR")) {
      INSTALLED_RESOURCES_DIR=getenv("MSMM_INSTALLED_RESOURCES_DIR");
    }
    static track_select::nn::FeedForwardNetwork net
      = track_select::nn::FeedForwardNetwork::readHDF5((INSTALLED_RESOURCES_DIR/"flimp"/"level-detection-net.h5").string());
    
    track_select::Vector vec = track_select::nn::softmax(net(in));
    return vec(0)/vec(1) > 0.01;
  }

  std::vector<int> find_steps_nn(const track_select::Vector& reg,
				 const track_select::Vector& sig_reg) {
    std::vector<int> steps;

    for (int i = 5; i < reg.size() - 5; i++) 
      if (find_steps_is_step(reg.segment(i - 5, 11)))
	steps.push_back(i);

    return steps;
  }

  std::vector<std::pair<int, int>> find_segments_nn(const track_select::Vector& reg,
						    const track_select::Vector& sig_reg) {
    std::vector<std::pair<int, int>> ranges = 
      construct_ranges(find_steps_nn(reg, sig_reg), 0, reg.size() -1);

    return ranges;
  }
  /* ***************************** */

  /* Splits track into subtracks (a proxy function) */

  std::vector<std::pair<int, int> > find_segments(const track_select::Vector& reg,
						  const track_select::Vector& sig_reg,
						  StepDetection detection_type) {
    switch (detection_type) {
    case StepSTD:
      return find_segments_step_std(reg);
    case NNet:
      return find_segments_nn(reg, sig_reg);
    }

    return find_segments_step_std(reg);
  }


  void split_into_subregions(const track_select::Vector& reg,
			     const track_select::Vector& sig_reg,
			     int seq_begin,
                             std::vector<track_select::Vector>& new_sequences,
                             std::vector<int>& new_sequences_begin,
			     StepDetection detection_type) {
    std::vector<std::pair<int, int> > ranges = find_segments(reg, sig_reg, detection_type);

    for (auto& range : ranges) {
      new_sequences.push_back(reg.segment(range.first, range.second - range.first +1));
      new_sequences_begin.push_back(seq_begin + range.first);
    }

  } // split_into_subregions


  void split_into_subregions(const track_select::Vector& reg,
			     const track_select::Vector& sig_reg,
			     int seq_begin,
                             const std::vector<track_select::Vector>& bg,
                             std::vector<track_select::Vector>& new_sequences,
                             std::vector<int>& new_sequences_begin,
                             std::vector<std::vector<track_select::Vector>>& new_bg,
			     StepDetection detection_type) {

    std::vector<std::pair<int, int> > ranges = find_segments(reg, sig_reg, detection_type);

    for (auto& range : ranges) {
      std::vector<track_select::Vector> b(range.second - range.first + 1);
      new_sequences.push_back(reg.segment(range.first, range.second - range.first +1));
      new_sequences_begin.push_back(seq_begin + range.first);
      std::copy(bg.begin() + range.first, bg.begin() + range.second + 1, b.begin());
      new_bg.push_back(b);
    }

  } // split_into_subregions



  void split_into_subregions(const std::vector<track_select::Vector>& sequences,
			     const std::vector<track_select::Vector>& sig_sequences,
                             const std::vector<int> sequences_begin,
                             std::vector<track_select::Vector>& new_sequences,
                             std::vector<int>& new_sequences_begin,
			     StepDetection detection_type) {
    if (sequences_begin.size() != sequences.size())
      throw std::runtime_error("identify_levels: sequences and sequences_begin not the same length");

    for (unsigned int seq_index = 0; seq_index < sequences.size(); seq_index++)
      split_into_subregions(sequences[seq_index], sig_sequences[seq_index],
			    sequences_begin[seq_index],
                            new_sequences, new_sequences_begin,
			    detection_type);
  }

  void split_into_subregions(const std::vector<track_select::Vector>& sequences,
			     const std::vector<track_select::Vector>& sig_sequences,
                             const std::vector<int> sequences_begin,
                             const std::vector<std::vector<track_select::Vector>>& bg,
                             std::vector<track_select::Vector>& new_sequences,
                             std::vector<int>& new_sequences_begin,
                             std::vector<std::vector<track_select::Vector>>& new_bg,
			     StepDetection detection_type) {

    if (sequences_begin.size() != sequences.size())
      throw std::runtime_error("identify_levels: sequences and sequences_begin not the same length");

    for (unsigned int seq_index = 0; seq_index < sequences.size(); seq_index++)
      split_into_subregions(sequences[seq_index], sig_sequences[seq_index],
			    sequences_begin[seq_index], bg[seq_index],
                            new_sequences, new_sequences_begin, new_bg, detection_type);
  }


  void identify_levels(const std::vector<FrameId>& frameid,
                       const std::vector<ImType>& counts,
		       const std::vector<ImType>& sigcounts,
		       StepDetection detection_type,
                       const std::vector<bool>& interp,
                       std::vector<Level>& levels)
  {

    if (frameid.size() < 5 || counts.size() < 5 || interp.size() < 5)  // less than 5 frames cannot be detected as level
      return;

    std::vector<track_select::Vector> new_sequences;
    std::vector<int> new_sequences_begin;

    {
      std::vector<track_select::Vector> sequences;
      std::vector<track_select::Vector> sig_sequences;
      std::vector<int> sequences_begin;

      FrameId begin = 0;
      FrameId end = frameid.size() - 1;
      for (int i = 1; i < frameid.size(); i++)
        if (frameid[i-1] +1 < frameid[i]) {
          track_select::Vector seq(frameid[i-1] + 1 - frameid[begin]);
          track_select::Vector sig_seq(frameid[i-1] + 1 - frameid[begin]);

          std::copy( counts.begin() + begin, counts.begin() + i - 1, seq.data());
          std::copy( sigcounts.begin() + begin, sigcounts.begin() + i - 1, sig_seq.data());

          sequences.push_back(seq);
	  sig_sequences.push_back(sig_seq);
          sequences_begin.push_back(begin);

          begin = i;
        } // endif


      if (begin < end) {
        track_select::Vector seq(frameid[end] + 1 - frameid[begin]);
        track_select::Vector sig_seq(frameid[end] + 1 - frameid[begin]);

        std::copy( counts.begin() + begin, counts.begin() + end - 1, seq.data());
        std::copy( sigcounts.begin() + begin, sigcounts.begin() + end - 1, sig_seq.data());

        sequences.push_back(seq);
        sig_sequences.push_back(sig_seq);
        sequences_begin.push_back(begin);

        begin = end;
      }

      split_into_subregions(sequences, sig_sequences,
                            sequences_begin, new_sequences, new_sequences_begin,
			    detection_type);
    }

    LevelSegments segments;
    //
    // Should iterate here over each level to store
    for (int seq_it = 0;  seq_it < new_sequences.size(); seq_it++) {
      // Create FrameRanges object
      track_select::Vector& seq = new_sequences[seq_it];

      LevelSegments res;

      if (detection_type == StepSTD) {
	res = remove(merge(extend(detect(seq), seq), seq), seq);
      } else if (detection_type == NNet) {
	LevelSegments seg;
	seg.push_back(std::make_pair(0, seq.size()-1));
	res = remove(seg, seq);
      } else
	throw std::runtime_error("Unknown level detection type");


      for (auto& sgm : res) {
        std::pair<int, int> shifted_sgm = sgm;

        shifted_sgm.first = frameid[shifted_sgm.first + new_sequences_begin[seq_it]];
        shifted_sgm.second = frameid[shifted_sgm.second + new_sequences_begin[seq_it]];

        segments.push_back(shifted_sgm);
      }

    } // endfor

    track_select::Vector seq = track_select::Vector::Zero(frameid.back() + 1);
    for (unsigned int i = 0; i < new_sequences.size(); i++)
      seq.segment(frameid[new_sequences_begin[i]], new_sequences[i].size()) = new_sequences[i];

    for (auto& raw_lvl : create_levels(segments, seq)) {
      FrameRanges new_level_frameranges;
      for (auto& sgm : raw_lvl)
        new_level_frameranges.push_back(FrameRange(sgm.first, sgm.second));

      levels.push_back(Level(new_level_frameranges));
    } // endfor

  } // identify_levels






  void identify_levels(const std::vector<FrameId>& frameid,
                       const std::vector<ImType>& counts,
		       const std::vector<ImType>& sigcounts,
		       StepDetection detection_type,
		       const std::vector<bool>& interp,
                       const std::vector<track_select::Vector>& bg,
                       std::vector<Level>& levels)
  {
    if (frameid.size() < 5 || counts.size() < 5 || interp.size() < 5)  // less than 5 frames cannot be detected as level
      return;

    std::vector<track_select::Vector> new_sequences;
    std::vector<int> new_sequences_begin;
    std::vector<std::vector<track_select::Vector>> new_background;

    {
      std::vector<track_select::Vector> sequences;
      std::vector<track_select::Vector> sig_sequences;
      std::vector<int> sequences_begin;
      std::vector<std::vector<track_select::Vector>> background;

      FrameId begin = 0;
      FrameId end = frameid.size() - 1;

      for (int i = 1; i < frameid.size(); i++)
        if (frameid[i-1] +1 < frameid[i]) {
          int c_size = frameid[i-1] + 1 - frameid[begin];
          track_select::Vector seq(c_size);
          track_select::Vector sig_seq(c_size);
          std::vector<track_select::Vector> b(c_size);

          std::copy( counts.begin() + begin, counts.begin() + begin + c_size, seq.data());
	  std::copy( sigcounts.begin() + begin, sigcounts.begin() + begin + c_size, sig_seq.data());
          std::copy( bg.begin() + begin, bg.begin() + begin + c_size, b.begin());

          sequences.push_back(seq);
	  sig_sequences.push_back(sig_seq);
          background.push_back(b);

          sequences_begin.push_back(begin);
          begin = i;
        } // endif

      if (begin < end) {
        int c_size = frameid[end] + 1 - frameid[begin];
        track_select::Vector seq = track_select::Vector::Zero(c_size);
        track_select::Vector sig_seq = track_select::Vector::Zero(c_size);
        std::vector<track_select::Vector> b(c_size);

        std::copy( counts.begin() + begin, counts.begin() + begin + c_size, seq.data());
        std::copy( sigcounts.begin() + begin, sigcounts.begin() + begin + c_size, sig_seq.data());
        std::copy( bg.begin() + begin, bg.begin() + begin + c_size, b.begin());

        sequences.push_back(seq);
        sig_sequences.push_back(sig_seq);
        sequences_begin.push_back(begin);
        background.push_back(b);
        begin = end;
      }

      split_into_subregions(sequences, sig_sequences, sequences_begin, background,
			    new_sequences, new_sequences_begin, new_background,
			    detection_type);
    }

    LevelSegments segments;
    //
    // Should iterate here over each level to store
    for (int seq_it = 0;  seq_it < new_sequences.size(); seq_it++) {
      // Create FrameRanges object
      track_select::Vector& seq = new_sequences[seq_it];
      std::vector<track_select::Vector>& background = new_background[seq_it];

      LevelSegments res;

      if (detection_type == StepSTD) {
	res = remove(merge(detect_background(extend(detect(seq), seq), background), seq), seq);
      } else if (detection_type == NNet) {
	LevelSegments seg;
	seg.push_back(std::make_pair(0, seq.size()-1));
	res = remove(detect_background(seg, background), seq);
      } else
	throw std::runtime_error("Unknown level detection type");
      
      for (auto& sgm : res) {
        std::pair<int, int> shifted_sgm = sgm;
        shifted_sgm.first = frameid[shifted_sgm.first + new_sequences_begin[seq_it]];
        shifted_sgm.second = frameid[shifted_sgm.second + new_sequences_begin[seq_it]];

        segments.push_back(shifted_sgm);
      }

    } // endfor

    track_select::Vector seq = track_select::Vector::Zero(frameid.back() + 1);

    for (unsigned int i = 0; i < new_sequences.size(); i++)
      seq.segment(frameid[new_sequences_begin[i]], new_sequences[i].size()) = new_sequences[i];

    for (auto& raw_lvl : create_levels(segments, seq)) {
      FrameRanges new_level_frameranges;
      for (auto& sgm : raw_lvl) {
        new_level_frameranges.push_back(FrameRange(sgm.first, sgm.second));
      }

      levels.push_back(Level(new_level_frameranges));
    } // endfor

  } // identify_levels


  typedef std::vector<std::pair<FrameRange, std::vector<short> > > fRanges;

  fRanges levels2ranges(const std::vector<Level>& levels) {
    fRanges frame_ranges;
    for (unsigned int i = 0; i < levels.size(); i++)
      for (unsigned int j = 0; j < levels[i].get_ranges().size(); j++) {
        std::vector<short> lvl(1);
        lvl[0] = i;
        frame_ranges.push_back(std::make_pair(levels[i].get_ranges()[j], lvl));
      }
    return frame_ranges;
  }

  std::vector<Level> ranges2levels(const fRanges& ranges) {
    std::map<std::string, FrameRanges> lvl;

    for (unsigned int i = 0; i < ranges.size(); i++) {
      std::stringstream ss;
      for (unsigned int j = 0; j < ranges[i].second.size(); j++)
        ss << ranges[i].second[j] << " ";
      lvl[ss.str()].push_back(ranges[i].first);
    }

    std::vector<Level> levels;
    for (std::map<std::string, FrameRanges>::iterator it = lvl.begin(); it != lvl.end(); it++)
      levels.push_back(Level(it->second));

    return levels;
  }
  typedef std::pair<FrameRange, std::vector<short> > fRange;

  fRange intersect(const fRange& left, const fRange& right) {
    std::vector<short> lvl(left.second.size() + right.second.size());
    unsigned int index = 0;

    for (unsigned int i = 0; i < left.second.size(); i++)
      lvl[index++] = left.second[i];
    for (unsigned int i = 0; i < right.second.size(); i++)
      lvl[index++] = right.second[i];



    return std::make_pair(FrameRange(std::max(left.first.min(), right.first.min()),
                                     std::min(left.first.max(), right.first.max())),
                          lvl);
  }


  bool overlap(const fRange& left, const fRange& right) {
    return (left.first.min() >= right.first.min() && left.first.min() < right.first.max()) ||
      (right.first.min() >= left.first.min() && right.first.min() < left.first.max());
  }

  fRanges intersect(const fRanges& left, const fRanges& right) {
    fRanges intersection;
    for (unsigned int i = 0; i < left.size(); i++)
      for (unsigned int j = 0; j < right.size(); j++)
        if (overlap(left[i], right[j]))
          intersection.push_back(intersect(left[i], right[j]));

    return intersection;
  }

  /* Finds levels which are common in all channels */
  std::vector<Level> intersect_channel_levels(const std::vector<std::vector<Level> >& ch_levels) {
    if (ch_levels.size() == 1)
      return ch_levels[0];
    if (ch_levels.size() == 0)
      return std::vector<Level>();

    std::vector< fRanges > frame_ranges;
    for (unsigned int lvl = 0; lvl < ch_levels.size(); lvl++)
      frame_ranges.push_back(levels2ranges(ch_levels[lvl]));

    fRanges intersection = frame_ranges[0];

    for (unsigned int i = 1; i < frame_ranges.size(); i++)
      intersection = intersect(frame_ranges[i], intersection);

    return ranges2levels(intersection);
  } // intersect_channel_levels

} // level_detect
