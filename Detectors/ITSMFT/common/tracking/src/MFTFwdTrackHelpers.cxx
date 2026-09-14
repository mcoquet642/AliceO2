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
/// \file MFTFwdTrackHelpers.cxx
/// \brief MFT CA full-track Kalman refit via standalone TrackFitter
///

#include "ITSMFTTracking/detail/MFTFwdTrackHelpers.h"

#include <algorithm>
#include <cmath>

#include "CommonConstants/MathConstants.h"
#include "ITSMFTTracking/Cell.h"
#include "ITSMFTTracking/SurfaceTrackState.h"
#include "ITSMFTTracking/detail/SurfaceTrackStateLegacyAdapters.h"
#include "ITStracking/Constants.h"
#include "MFTTracking/Cluster.h"
#include "MFTTracking/MFTTrackingParam.h"
#include "MFTTracking/TrackCA.h"
#include "MFTTracking/TrackFitter.h"

namespace o2::itsmft::tracking
{

namespace
{
template <typename TrackLTFType>
bool refitWithTrackFitter(TrackLTFType& inward, TrackLTFType& outward, float bz, int trackModel)
{
  const auto& mftParam = o2::mft::MFTTrackingParam::Instance();
  o2::mft::TrackFitter<TrackLTFType> fitter;
  fitter.setBz(bz);
  fitter.setMFTRadLength(mftParam.MFTRadLength);
  fitter.setVerbosity(mftParam.verbose);
  fitter.setAlignResiduals(mftParam.alignResidual);
  fitter.setTrackModel(trackModel);
  // Same sequence as Tracker::fitTracks: independent inward and outward seeds.
  return fitter.initTrack(inward) && fitter.fit(inward) &&
         fitter.initTrack(outward, true) && fitter.fit(outward, true);
}

template <typename TrackLTFType>
bool fillTrackLTF(TrackLTFType& track,
                  const TrackSeed& seed,
                  gsl::span<const gsl::span<const GlobalMeasurement>> layerGlobals)
{
  const auto hitMask = seed.getHitLayerMask();
  for (int layer = 0; layer < o2::mft::constants::mft::LayersNumber; ++layer) {
    if (!hitMask.has(layer)) {
      continue;
    }
    const int clIdx = seed.getCluster(layer);
    if (clIdx == o2::its::constants::UnusedIndex) {
      continue;
    }
    if (clIdx < 0 || static_cast<std::size_t>(clIdx) >= layerGlobals[layer].size()) {
      return false;
    }
    const auto& global = layerGlobals[layer][clIdx];
    const o2::mft::Cluster cluster{global.x, global.y, global.z, global.phi, global.radius, clIdx, 0,
                                   global.covariance.xx, global.covariance.yy, 0};
    track.setPoint(cluster, layer, clIdx, {}, clIdx, 0);
  }
  if (track.getNumberOfPoints() < 2) {
    return false;
  }
  track.sort();
  return true;
}

bool acceptFittedTrack(const o2::track::TrackParCovFwd& inward, int nClusters,
                       gsl::span<const float> minPt, float maxChi2NDF)
{
  const int ndf = std::max(1, 2 * nClusters - 5);
  if (static_cast<float>(inward.getTrackChi2()) / static_cast<float>(ndf) > maxChi2NDF) {
    return false;
  }
  const int minPtSlot = o2::mft::constants::mft::LayersNumber - nClusters;
  if (minPtSlot >= 0 && minPtSlot < static_cast<int>(minPt.size()) && minPt[minPtSlot] > 0.f &&
      inward.getPt() < minPt[minPtSlot]) {
    return false;
  }
  return true;
}

template <typename TrackLTFType>
bool refitAndImport(const TrackSeed& seed,
                    gsl::span<const gsl::span<const GlobalMeasurement>> layerGlobals,
                    float bz,
                    int trackModel,
                    gsl::span<const float> minPt,
                    float maxChi2NDF,
                    SurfaceTrackState& inner,
                    SurfaceTrackState& outer,
                    float& chi2,
                    float& outChi2,
                    float& invQPtSeed,
                    float& chi2QPtSeed)
{
  TrackLTFType inward(true);
  if (!fillTrackLTF(inward, seed, layerGlobals)) {
    return false;
  }
  TrackLTFType outward = inward;
  if (!refitWithTrackFitter(inward, outward, bz, trackModel)) {
    return false;
  }
  inward.setOutParam(outward);
  if (!acceptFittedTrack(inward, inward.getNumberOfPoints(), minPt, maxChi2NDF)) {
    return false;
  }
  if (!legacy::importLegacyForwardTrackParCov(inward, inner) ||
      !legacy::importLegacyForwardTrackParCov(outward, outer)) {
    return false;
  }
  chi2 = static_cast<float>(inward.getTrackChi2());
  outChi2 = static_cast<float>(outward.getTrackChi2());
  invQPtSeed = static_cast<float>(inward.getInvQPtSeed());
  chi2QPtSeed = static_cast<float>(inward.getChi2QPtSeed());
  return true;
}
} // namespace

bool mftFwdRefitFullTrack(const TrackSeed& seed,
                          gsl::span<const gsl::span<const GlobalMeasurement>> layerGlobals,
                          float bz,
                          gsl::span<const float> minPt,
                          float maxChi2NDF,
                          SurfaceTrackState& inner,
                          SurfaceTrackState& outer,
                          float& chi2,
                          float& outChi2,
                          float& invQPtSeed,
                          float& chi2QPtSeed)
{
  if (layerGlobals.size() != static_cast<std::size_t>(o2::mft::constants::mft::LayersNumber)) {
    return false;
  }
  const auto& mftParam = o2::mft::MFTTrackingParam::Instance();
  const bool zeroField = mftParam.forceZeroField || std::abs(bz) < o2::constants::math::Almost0;
  if (zeroField) {
    // Magnet-off tracker uses TrackLTFL (linear model, invQPt seed 0).
    return refitAndImport<o2::mft::TrackLTFL>(seed, layerGlobals, 0.f, o2::mft::MFTTrackModel::Linear, minPt,
                                              maxChi2NDF, inner, outer, chi2, outChi2, invQPtSeed, chi2QPtSeed);
  }
  return refitAndImport<o2::mft::TrackLTF>(seed, layerGlobals, bz, mftParam.trackmodel, minPt, maxChi2NDF, inner,
                                          outer, chi2, outChi2, invQPtSeed, chi2QPtSeed);
}

} // namespace o2::itsmft::tracking
