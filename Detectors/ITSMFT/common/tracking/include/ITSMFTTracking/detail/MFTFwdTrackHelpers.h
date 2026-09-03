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
/// \file MFTFwdTrackHelpers.h
/// \brief Forward-track coordinate helpers for MFT CA candidate finding
///

#ifndef ALICEO2_ITSMFT_TRACKING_MFTFWDTRACKHELPERS_H_
#define ALICEO2_ITSMFT_TRACKING_MFTFWDTRACKHELPERS_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <gsl/span>

#include "CommonConstants/MathConstants.h"
#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/Constants.h"
#include "ITSMFTTracking/GlobalMeasurement.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "MFTTracking/Constants.h"
#include "ReconstructionDataFormats/TrackFwd.h"

namespace o2::itsmft::tracking::detail
{

/// MFT CA uses o2::mft::constants::mft::LayersNumber half-disk layers (same index as GeometryTGeo::getLayer).
/// Physical disk index is halfLayer / 2; ROFOverlapTable stores one LayerTiming per half-layer.

inline bool isMftTopology(int nLayers) noexcept
{
  return nLayers == MFTNLayers;
}

inline float mftLayerZ(int layer)
{
  return o2::mft::constants::mft::LayerZCoordinate()[layer];
}

inline float mftLayerMSAngle(int layer, const TrackingParameters& params)
{
  const float invP = 1.f / params.TrackletMinPt;
  const float zLayer = mftLayerZ(layer);
  const float rRef = params.LayerRadii[layer];
  const float tanlRef = (std::abs(rRef) > 1e-6f) ? zLayer / rRef : 0.f;
  const float absTanl = std::abs(tanlRef);
  const float cscLambda = (absTanl > 1e-6f) ? std::sqrt(1.f + tanlRef * tanlRef) / absTanl : 1e6f;
  return 0.0136f * invP * std::sqrt(params.LayerxX0[layer] * cscLambda);
}

inline void mftTrackletProject(float xCl, float yCl, float zCl, float pvX, float pvY, float pvZ,
                               float zFrom, float zTo, float bz, float minPt,
                               float& xProj, float& yProj)
{
  if (std::abs(bz) > 0.01f && minPt > 0.f) {
    const float dxTan = xCl - pvX;
    const float dyTan = yCl - pvY;
    const float dzTan = zCl - pvZ;
    const float drTan = std::sqrt(dxTan * dxTan + dyTan * dyTan);
    float invQPt = 1.f / minPt;
    float tanl = (drTan > 1e-6f) ? -std::abs(dzTan) / drTan : -1.f;
    float phi = (drTan > 1e-6f) ? std::atan2(dyTan, dxTan) : 0.f;
    if (std::abs(tanl) > 1e-6f) {
      const float k = std::abs(o2::constants::math::B2C * bz);
      const float hz = (bz > 0.f) ? 1.f : -1.f;
      phi -= 0.5f * hz * invQPt * dzTan * k / tanl;
    }
    ROOT::Math::SVector<double, 5> params{xCl, yCl, phi, tanl, invQPt};
    ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>> cov{};
    cov(0, 0) = cov(1, 1) = cov(2, 2) = cov(3, 3) = 1.;
    const double qptSigma = std::clamp(static_cast<double>(std::abs(invQPt)), 1., 10.);
    cov(4, 4) = qptSigma * qptSigma;
    o2::track::TrackParCovFwd track{zCl, params, cov, 0.};
    track.propagateToZhelix(zTo, bz);
    xProj = static_cast<float>(track.getX());
    yProj = static_cast<float>(track.getY());
  } else {
    const float dz0 = zFrom - pvZ;
    if (std::abs(dz0) < 1e-6f) {
      xProj = xCl;
      yProj = yCl;
      return;
    }
    const float w = (zTo - pvZ) / dz0;
    xProj = pvX + w * (xCl - pvX);
    yProj = pvY + w * (yCl - pvY);
  }
}

inline void mftTrackletProject(float xCl, float yCl, float zCl, float pvX, float pvY, float pvZ,
                               int fromLayer, int toLayer, float bz, float minPt,
                               float& xProj, float& yProj)
{
  mftTrackletProject(xCl, yCl, zCl, pvX, pvY, pvZ, mftLayerZ(fromLayer), mftLayerZ(toLayer),
                     bz, minPt, xProj, yProj);
}

inline void mftTrackletSigmaXY(float x0, float y0, float pvX, float pvY, float pvZ,
                               float sigma2X0, float sigma2Y0, float sigma2PvX, float sigma2PvY, float sigma2PvZ,
                               float zFrom, float zTo, float rLayerFrom, float meanDeltaZ, float msAngle,
                               float bendingAngle, float xProj, float yProj, float& sigmaX, float& sigmaY)
{
  const float dz0 = zFrom - pvZ;
  const float tanlRef = (std::abs(rLayerFrom) > 1e-6f) ? zFrom / rLayerFrom : 0.f;
  const float sigma2MS = meanDeltaZ * meanDeltaZ * msAngle * msAngle * (tanlRef * tanlRef + 1.f);
  if (std::abs(dz0) < o2::its::constants::Tolerance) {
    sigmaX = std::sqrt(sigma2X0 + sigma2PvX + sigma2MS);
    sigmaY = std::sqrt(sigma2Y0 + sigma2PvY + sigma2MS);
  } else {
    const float w = (zTo - pvZ) / dz0;
    const float invDz0 = w / dz0;
    const float sigma2W = invDz0 * invDz0 * sigma2PvZ;
    const float dx0 = x0 - pvX;
    const float dy0 = y0 - pvY;
    const float oneMinusW = 1.f - w;
    sigmaX = std::sqrt(oneMinusW * oneMinusW * sigma2PvX + w * w * sigma2X0 + dx0 * dx0 * sigma2W + sigma2MS);
    sigmaY = std::sqrt(oneMinusW * oneMinusW * sigma2PvY + w * w * sigma2Y0 + dy0 * dy0 * sigma2W + sigma2MS);
  }
  const float rProj = std::hypot(xProj, yProj);
  if (rProj > 1e-6f && bendingAngle > 0.f) {
    const float dr = rProj * bendingAngle;
    const float invR = 1.f / rProj;
    const float sinPhi = yProj * invR;
    const float cosPhi = xProj * invR;
    sigmaX = std::sqrt(sigmaX * sigmaX + dr * dr * sinPhi * sinPhi);
    sigmaY = std::sqrt(sigmaY * sigmaY + dr * dr * cosPhi * cosPhi);
  }
}

inline void mftTrackletSigmaXY(float x0, float y0, float pvX, float pvY, float pvZ,
                               float sigma2X0, float sigma2Y0, float sigma2PvX, float sigma2PvY, float sigma2PvZ,
                               int fromLayer, int toLayer, float rLayerFrom, float meanDeltaZ, float msAngle,
                               float bendingAngle, float xProj, float yProj, float& sigmaX, float& sigmaY)
{
  mftTrackletSigmaXY(x0, y0, pvX, pvY, pvZ, sigma2X0, sigma2Y0, sigma2PvX, sigma2PvY, sigma2PvZ,
                     mftLayerZ(fromLayer), mftLayerZ(toLayer), rLayerFrom, meanDeltaZ, msAngle,
                     bendingAngle, xProj, yProj, sigmaX, sigmaY);
}

inline void mftFwdPropagateToZ(o2::track::TrackParCovFwd& track, float z, float bz)
{
  if (std::abs(bz) > 0.01f) {
    track.propagateToZhelix(z, bz);
  } else {
    track.propagateToZlinear(z);
  }
}

inline float mftFwdPredictedChi2(const o2::track::TrackParCovFwd& track, float x, float y, float sigma2X, float sigma2Y)
{
  const float dx = x - static_cast<float>(track.getX());
  const float dy = y - static_cast<float>(track.getY());
  const float vx = static_cast<float>(track.getSigma2X()) + sigma2X;
  const float vy = static_cast<float>(track.getSigma2Y()) + sigma2Y;
  if (vx <= 0.f || vy <= 0.f) {
    return o2::constants::math::VeryBig;
  }
  return dx * dx / vx + dy * dy / vy;
}

inline float mftFwdStateChi2(const o2::track::TrackParCovFwd& current, const o2::track::TrackParCovFwd& rhs)
{
  ROOT::Math::SVector<double, 5> diff{
    rhs.getX() - current.getX(),
    rhs.getY() - current.getY(),
    rhs.getPhi() - current.getPhi(),
    rhs.getTanl() - current.getTanl(),
    rhs.getInvQPt() - current.getInvQPt()};
  auto cov = current.getCovariances();
  cov += rhs.getCovariances();
  if (!cov.Invert()) {
    return o2::constants::math::VeryBig;
  }
  return static_cast<float>(ROOT::Math::Similarity(cov, diff));
}

inline bool mftFwdAttachCluster(o2::track::TrackParCovFwd& track, float z, float x, float y,
                                float sigma2X, float sigma2Y, float xOverX0, float bz, float maxChi2,
                                float& chi2, bool checkChi2OnLast = false)
{
  mftFwdPropagateToZ(track, z, bz);
  if (xOverX0 > 0.f) {
    track.addMCSEffect(xOverX0);
  }
  const float predChi2 = mftFwdPredictedChi2(track, x, y, sigma2X, sigma2Y);
  if (checkChi2OnLast && predChi2 > maxChi2) {
    return false;
  }
  const std::array<float, 2> p{x, y};
  const std::array<float, 2> cov{sigma2X, sigma2Y};
  if (!track.update(p, cov)) {
    return false;
  }
  chi2 += predChi2;
  return true;
}

/// Squared transverse distance from cluster c to the seed line c1→c2 (legacy MFT getDistanceToSeed).
inline float mftDistanceToSeedSquared(const GlobalMeasurement& c1, const GlobalMeasurement& c2, const GlobalMeasurement& c)
{
  const float dxSeed = c2.x - c1.x;
  const float dySeed = c2.y - c1.y;
  const float dzSeed = c2.z - c1.z;
  if (std::abs(dzSeed) < 1e-9f) {
    return std::numeric_limits<float>::max();
  }
  const float invdzSeed = (c.z - c1.z) / dzSeed;
  const float xSeed = c1.x + dxSeed * invdzSeed;
  const float ySeed = c1.y + dySeed * invdzSeed;
  const float dx = c.x - xSeed;
  const float dy = c.y - ySeed;
  return dx * dx + dy * dy;
}

/// Conical road scale (1 + dz/z_from)^2 between half-layers (legacy ROADclsRCut behaviour).
inline float mftConicalRoadR2Scale(int layerFrom, int layerTo)
{
  const float zFrom = mftLayerZ(layerFrom);
  if (std::abs(zFrom) < 1e-6f) {
    return 1.f;
  }
  const float dCone = 1.f + (mftLayerZ(layerTo) - zFrom) / zFrom;
  return dCone * dCone;
}

/// Cheap geometric pre-cut before forward cell fit (CellRoadRCut / ROADclsRCut).
inline bool validateMFTCellClusters(const GlobalMeasurement& c0, int layer0,
                                    const GlobalMeasurement& c1, int layer1,
                                    const GlobalMeasurement& c2, int layer2,
                                    float r2Cut)
{
  const float r2 = r2Cut * r2Cut;
  return mftDistanceToSeedSquared(c0, c2, c1) < r2 * mftConicalRoadR2Scale(layer0, layer1) &&
         mftDistanceToSeedSquared(c0, c1, c2) < r2 * mftConicalRoadR2Scale(layer0, layer2) &&
         mftDistanceToSeedSquared(c1, c2, c0) < r2 * mftConicalRoadR2Scale(layer1, layer0);
}

/// Build inward forward seed at the outer cluster and Kalman-fit the three cell clusters.
inline bool mftFwdFitCellClusters(const std::array<GlobalMeasurement, 3>& measurements,
                                  const std::array<int, 3>& hitLayers,
                                  gsl::span<const NominalSurfaceMaterial> layerMaterial,
                                  float trackletMinPt,
                                  float bz,
                                  float maxChi2,
                                  o2::track::TrackParCovFwd& track,
                                  float& chi2)
{
  const auto& cInner = measurements[0];
  const auto& cMid = measurements[1];
  const auto& cOuter = measurements[2];
  if (cInner.z <= cOuter.z + 1.e-6f) {
    return false;
  }

  const float dxTan = cMid.x - cInner.x;
  const float dyTan = cMid.y - cInner.y;
  const float dzTan = cMid.z - cInner.z;
  const float drTan = std::sqrt(dxTan * dxTan + dyTan * dyTan);
  const float dxPhi = cOuter.x - cInner.x;
  const float dyPhi = cOuter.y - cInner.y;
  const float dzPhi = cOuter.z - cInner.z;
  const float drPhi = std::sqrt(dxPhi * dxPhi + dyPhi * dyPhi);
  if (drTan < 1.e-6f || std::abs(dzTan) < 1.e-6f || drPhi < 1.e-6f || std::abs(dzPhi) < 1.e-6f) {
    return false;
  }

  const float invQPt = (trackletMinPt > 0.f) ? 1.f / trackletMinPt : 0.f;
  float tanl{0.f};
  float phi{0.f};
  if (std::abs(bz) > 0.01f) {
    tanl = -std::abs(dzTan) / drTan;
    phi = std::atan2(dyPhi, dxPhi);
    if (std::abs(tanl) > 1.e-6f) {
      const float k = std::abs(o2::constants::math::B2C * bz);
      const float hz = (bz > 0.f) ? 1.f : -1.f;
      phi -= 0.5f * hz * invQPt * dzPhi * k / tanl;
    }
  } else {
    tanl = -std::abs(dzPhi) / drPhi;
    phi = std::atan2(dyPhi, dxPhi);
  }

  const float sigma2XOuter = cOuter.covariance.xx > 0.f ? cOuter.covariance.xx : 1.f;
  const float sigma2YOuter = cOuter.covariance.yy > 0.f ? cOuter.covariance.yy : 1.f;
  ROOT::Math::SVector<double, 5> seedParams{cOuter.x, cOuter.y, phi, tanl, invQPt};
  ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>> seedCov{};
  seedCov(0, 0) = sigma2XOuter;
  seedCov(1, 1) = sigma2YOuter;
  seedCov(2, 2) = seedCov(3, 3) = 1.;
  const double qptSigma = std::clamp(static_cast<double>(std::abs(invQPt)), 1., 10.);
  seedCov(4, 4) = qptSigma * qptSigma;
  track = {cOuter.z, seedParams, seedCov, 0.};

  chi2 = 0.f;
  for (int iC{2}; iC >= 0; --iC) {
    const int layer = hitLayers[iC];
    if (layer < 0 || static_cast<std::size_t>(layer) >= layerMaterial.size()) {
      return false;
    }
    const auto& measurement = measurements[iC];
    const float sigma2X = measurement.covariance.xx > 0.f ? measurement.covariance.xx : 1.f;
    const float sigma2Y = measurement.covariance.yy > 0.f ? measurement.covariance.yy : 1.f;
    if (!mftFwdAttachCluster(track, measurement.z, measurement.x, measurement.y,
                             sigma2X, sigma2Y, layerMaterial[layer].xOverX0, bz, maxChi2, chi2, iC == 0)) {
      return false;
    }
  }
  return true;
}

/// Compatibility of two adjacent MFT cells via forward-state χ² (mft-time-aware).
inline bool mftFwdCellsAreCompatible(const std::array<GlobalMeasurement, 3>& current,
                                     const std::array<int, 3>& currentLayers,
                                     const std::array<GlobalMeasurement, 3>& next,
                                     const std::array<int, 3>& nextLayers,
                                     gsl::span<const NominalSurfaceMaterial> layerMaterial,
                                     float trackletMinPt,
                                     float bz,
                                     float maxChi2)
{
  o2::track::TrackParCovFwd currentFwd;
  o2::track::TrackParCovFwd nextFwd;
  float currentChi2 = 0.f;
  float nextChi2 = 0.f;
  if (!mftFwdFitCellClusters(current, currentLayers, layerMaterial, trackletMinPt, bz, maxChi2, currentFwd, currentChi2) ||
      !mftFwdFitCellClusters(next, nextLayers, layerMaterial, trackletMinPt, bz, maxChi2, nextFwd, nextChi2)) {
    return false;
  }
  mftFwdPropagateToZ(nextFwd, static_cast<float>(currentFwd.getZ()), bz);
  return mftFwdStateChi2(currentFwd, nextFwd) <= maxChi2;
}

} // namespace o2::itsmft::tracking::detail

#endif /* ALICEO2_ITSMFT_TRACKING_MFTFWDTRACKHELPERS_H_ */
