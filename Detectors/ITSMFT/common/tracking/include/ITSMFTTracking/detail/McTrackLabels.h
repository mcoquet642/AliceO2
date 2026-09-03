// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file McTrackLabels.h
/// \brief MC label assignment for reconstructed CA tracks (mft-time-aware parity)
///

#ifndef ALICEO2_ITSMFT_TRACKING_DETAIL_MCTRACKLABELS_H_
#define ALICEO2_ITSMFT_TRACKING_DETAIL_MCTRACKLABELS_H_

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include <gsl/span>

#include "ITSMFTTracking/GenericTrack.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2::itsmft::tracking::detail
{

inline bool isMftTopology(int nLayers) noexcept
{
  return nLayers == MFTNLayers;
}

/// Reference mft-time-aware occurrence counting: for each attached cluster, bump
/// every matching label entry (multiple labels on one cluster can increment
/// several entries in the same pass).
inline std::vector<std::pair<o2::MCCompLabel, std::size_t>> collectMftMcLabelOccurrences(
  const GenericTrack& track,
  gsl::span<const TrackClusterReference> references,
  const TimeFrame& frame)
{
  std::vector<std::pair<o2::MCCompLabel, std::size_t>> occurrences;
  for (uint32_t index = track.firstClusterRef; index < track.clusterRefEnd; ++index) {
    const auto& reference = references[index];
    const auto labels = frame.getLabels(reference.layer, reference.clusterId);
    bool found{false};
    for (auto& occurrence : occurrences) {
      for (const auto& label : labels) {
        if (label == occurrence.first) {
          ++occurrence.second;
          found = true;
        }
      }
    }
    if (!found) {
      for (const auto& label : labels) {
        occurrences.emplace_back(label, 1);
      }
    }
  }
  return occurrences;
}

/// MFT: dominant label must cover at least TrueTrackMCThreshold of clusters.
inline o2::MCCompLabel assignMftTrackMcLabel(std::size_t nClusters,
                                               gsl::span<const std::pair<o2::MCCompLabel, std::size_t>> occurrences,
                                               float trueTrackThreshold)
{
  o2::MCCompLabel winner;
  if (occurrences.empty()) {
    winner.setFakeFlag();
    return winner;
  }
  const auto best = std::max_element(occurrences.begin(), occurrences.end(),
                                     [](const auto& left, const auto& right) {
                                       return left.second < right.second;
                                     });
  winner = best->first;
  if (nClusters == 0 || static_cast<float>(best->second) / static_cast<float>(nClusters) < trueTrackThreshold) {
    winner.setFakeFlag();
  }
  return winner;
}

struct ItsMcLabelCandidate {
  o2::MCCompLabel representative;
  std::size_t count{0};
  std::size_t lastSeenCluster{0};
};

/// ITS: count each MC label at most once per attached cluster.
inline std::vector<ItsMcLabelCandidate> collectItsMcLabelCandidates(
  const GenericTrack& track,
  gsl::span<const TrackClusterReference> references,
  const TimeFrame& frame)
{
  std::vector<ItsMcLabelCandidate> candidates;
  std::size_t attachedClusters = 0;
  for (uint32_t index = track.firstClusterRef; index < track.clusterRefEnd; ++index) {
    const auto& reference = references[index];
    ++attachedClusters;
    for (const auto& label : frame.getLabels(reference.layer, reference.clusterId)) {
      const auto candidate = std::find_if(candidates.begin(), candidates.end(), [&label](const auto& current) {
        return label == current.representative;
      });
      if (candidate == candidates.end()) {
        candidates.push_back({label, 1, attachedClusters});
      } else if (candidate->lastSeenCluster != attachedClusters) {
        ++candidate->count;
        candidate->lastSeenCluster = attachedClusters;
      }
    }
  }
  return candidates;
}

/// ITS: every attached cluster must carry the winning MC label.
inline o2::MCCompLabel assignItsTrackMcLabel(std::size_t attachedClusters,
                                               gsl::span<const ItsMcLabelCandidate> candidates)
{
  o2::MCCompLabel winner;
  if (candidates.empty()) {
    winner.setFakeFlag();
    return winner;
  }
  const auto best = std::max_element(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
    return left.count < right.count;
  });
  winner = best->representative;
  if (best->count != attachedClusters) {
    winner.setFakeFlag();
  }
  return winner;
}

} // namespace o2::itsmft::tracking::detail

#endif /* ALICEO2_ITSMFT_TRACKING_DETAIL_MCTRACKLABELS_H_ */
