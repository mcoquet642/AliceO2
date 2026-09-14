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
#include <type_traits>

#include "CommonConstants/MathConstants.h"
#include "ITSMFTTracking/Cell.h"
#include "ITSMFTTracking/Constants.h"
#include "ITSMFTTracking/SurfaceTrackState.h"
#include "ITSMFTTracking/detail/SurfaceTrackStateLegacyAdapters.h"
#include "MFTTracking/Cluster.h"
#include "MFTTracking/MFTTrackingParam.h"
#include "MFTTracking/TrackCA.h"
#include "MFTTracking/TrackFitter.h"

namespace o2::itsmft::tracking
{

namespace
{
// Same geometry seed as TrackFitter::initTrack, but q/pT (and the matching phi
// correction) come from a previous Kalman pass instead of the fast-circle fit.
bool seedTrackAtEnd(o2::mft::TrackLTF& track, double invQPt, float bz, bool outward)
{
  const int nPoints = track.getNumberOfPoints();
  if (nPoints < 2 || !std::isfinite(invQPt) || invQPt == 0.) {
    return false;
  }
  const int first = outward ? 0 : nPoints - 1;
  const int next = outward ? nPoints - 1 : 0;
  const double k = std::abs(o2::constants::math::B2C * bz);
  const double hz = std::copysign(1., static_cast<double>(bz));
  const auto& x = track.getXCoordinates();
  const auto& y = track.getYCoordinates();
  const auto& z = track.getZCoordinates();

  const double dRtan = std::hypot(x[1] - x[0], y[1] - y[0]);
  if (dRtan < 1.e-6) {
    return false;
  }
  const double tanl0 = -std::abs((z[1] - z[0]) / dRtan);
  const double dX = x[first] - x[next];
  const double dY = y[first] - y[next];
  const double dZ = z[first] - z[next];
  double phi0 = outward ? std::atan2(-dY, -dX) : std::atan2(dY, dX);
  if (std::abs(tanl0) > 1.e-6) {
    phi0 -= 0.5 * hz * invQPt * dZ * k / tanl0;
  }

  track.setX(x[first]);
  track.setY(y[first]);
  track.setZ(z[first]);
  track.setPhi(phi0);
  track.setTanl(tanl0);
  track.setInvQPt(invQPt);

  o2::track::SMatrix55Sym covariance;
  const double qptSigma = std::clamp(std::abs(invQPt), 1., 10.);
  covariance(0, 0) = covariance(1, 1) = covariance(2, 2) = covariance(3, 3) = 1.;
  covariance(4, 4) = qptSigma;
  track.setCovariances(covariance);
  track.setTrackChi2(0.);
  return true;
}

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
  if (!fitter.initTrack(inward) || !fitter.fit(inward)) {
    return false;
  }
  if constexpr (std::is_same_v<TrackLTFType, o2::mft::TrackLTF>) {
    // Second inward pass: linearize the helix at the first-pass curvature so
    // the hit updates actually pull q/pT. Position and tanl stay measurement-driven.
    const double fittedInvQPt = inward.getInvQPt();
    TrackLTFType firstPass = inward;
    if (!(seedTrackAtEnd(inward, fittedInvQPt, bz, false) && fitter.fit(inward))) {
      inward = firstPass;
    }
    const double seedInvQPt = inward.getInvQPt();
    if (seedTrackAtEnd(outward, seedInvQPt, bz, true) && fitter.fit(outward, true)) {
      return true;
    }
  }
  return fitter.initTrack(outward, true) && fitter.fit(outward, true);
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
  // Optimized transports the covariance with the small-angle Jacobian, which
  // under-corrects q/pT on the short MFT lever arm. Helix matches the parameter step.
  int trackModel = mftParam.trackmodel;
  if (trackModel == o2::mft::MFTTrackModel::Optimized) {
    trackModel = o2::mft::MFTTrackModel::Helix;
  }
  return refitAndImport<o2::mft::TrackLTF>(seed, layerGlobals, bz, trackModel, minPt, maxChi2NDF, inner, outer,
                                          chi2, outChi2, invQPtSeed, chi2QPtSeed);
}

} // namespace o2::itsmft::tracking
