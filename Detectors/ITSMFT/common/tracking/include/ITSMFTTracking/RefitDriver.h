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

#ifndef ALICEO2_ITSMFT_TRACKING_REFITDRIVER_H_
#define ALICEO2_ITSMFT_TRACKING_REFITDRIVER_H_

#include "GPUCommonDef.h"

#ifndef GPUCA_GPUCODE

#include <algorithm>
#include <array>
#include <cmath>

#include <gsl/span>

#include "CommonConstants/MathConstants.h"
#include "ITSMFTTracking/Cell.h"
#include "ITSMFTTracking/GlobalMeasurement.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/Propagator.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ReconstructionDataFormats/TrackParametrization.h"

// Descriptor-driven refit built on Propagator operations.
namespace o2::itsmft::tracking
{

namespace detail
{

struct RefitMeasurementSlot {
  SurfaceMeasurement measurement{};
  LayerId surface{};
  bool present{false};
};

/// Builds an ordered refit leg; holes remain explicit.
inline gsl::span<const RefitMeasurementSlot> assembleRefitLegSlots(
  const TrackSeed& seed,
  const TimeFrame& frame,
  gsl::span<const gsl::span<const GlobalMeasurement>> layerGlobals,
  int start, int end, int step,
  gsl::span<RefitMeasurementSlot> out,
  bool& valid) noexcept
{
  valid = layerGlobals.size() <= MaxLayoutSurfaces;
  int position = 0;
  for (int surfacePosition = start; surfacePosition != end && position < static_cast<int>(out.size()); surfacePosition += step) {
    const int clsIdx = seed.getCluster(surfacePosition);
    if (clsIdx == o2::its::constants::UnusedIndex) {
      out[position++] = {};
      continue;
    }
    if (!valid || clsIdx < 0 || static_cast<std::size_t>(clsIdx) >= layerGlobals[surfacePosition].size()) {
      valid = false;
      return {};
    }
    const auto& global = layerGlobals[surfacePosition][clsIdx];
    const auto surface = LayerId{static_cast<uint16_t>(surfacePosition)};
    const auto* measurement = frame.getSurfaceMeasurement(surface, global.clusterId);
    if (measurement == nullptr) {
      valid = false;
      return {};
    }
    out[position++] = RefitMeasurementSlot{*measurement, surface, true};
  }
  return gsl::span<const RefitMeasurementSlot>(out.data(), position);
}

// Missing hits are not Kalman-updated. Disk MCS still walks catalog layers
// between present hits, including holes. Present slots must resolve to a
// descriptor. Commit state, reference, chi2 and count only after the full
// leg succeeds.
inline bool driveRefitLeg(SurfaceTrackState& state, SurfaceTrackParameters& linRef,
                          float& chi2, uint32_t& acceptedHitCount,
                          gsl::span<const RefitMeasurementSlot> orderedSlots, SurfaceCatalogView surfaceCatalog,
                          float bz, material::MaterialTraversalDirection direction,
                          bool shiftReferenceToMeasurement, float maxChi2,
                          float alignResidual = 0.f) noexcept
{
  if (chi2 < 0.f) {
    return false;
  }

  SurfaceTrackState scratchState = state;
  SurfaceTrackParameters scratchLinRef = linRef;
  float scratchChi2 = chi2;
  uint32_t scratchAcceptedHitCount = 0;
  constexpr uint32_t kChi2GateMinAcceptedHits = 3;
  LayerId previousSurface{};
  for (const auto& slot : orderedSlots) {
    if (!slot.present) {
      continue;
    }
    if (!slot.surface.isValid() || !(surfaceCatalog.nSurfaces == 0 || surfaceCatalog.surfaces != nullptr) ||
        !(slot.surface.value() < surfaceCatalog.nSurfaces)) {
      return false;
    }
    const SurfaceDescriptor& descriptor = surfaceCatalog.getSurface(slot.surface);
    SurfaceMeasurement measurement = slot.measurement;
    if (descriptor.kind == SurfaceKind::Disk) {
      measurement.covariance.uv = 0.f;
      measurement.covariance.uu += alignResidual;
      measurement.covariance.vv += alignResidual;
    }
    if (!Propagator::propagateToMeasurement(scratchState, scratchLinRef, descriptor, measurement, bz, direction,
                                            scratchAcceptedHitCount >= kChi2GateMinAcceptedHits, maxChi2, scratchChi2,
                                            shiftReferenceToMeasurement, surfaceCatalog, previousSurface)) {
      return false;
    }
    previousSurface = slot.surface;
    ++scratchAcceptedHitCount;
  }
  state = scratchState;
  linRef = scratchLinRef;
  chi2 = scratchChi2;
  acceptedHitCount = scratchAcceptedHitCount;
  return true;
}

struct DiskRefitHits {
  std::array<float, MaxLayoutSurfaces> x{};
  std::array<float, MaxLayoutSurfaces> y{};
  std::array<float, MaxLayoutSurfaces> z{};
  std::array<float, MaxLayoutSurfaces> sigmaX2{};
  std::array<float, MaxLayoutSurfaces> sigmaY2{};
  int nHits{0};
};

// Present hits in inner-to-outer catalog order, matching TrackLTF point order.
inline bool collectDiskRefitHits(gsl::span<const RefitMeasurementSlot> innerToOuterSlots,
                                 DiskRefitHits& hits) noexcept
{
  hits.nHits = 0;
  for (const auto& slot : innerToOuterSlots) {
    if (!slot.present) {
      continue;
    }
    if (hits.nHits >= static_cast<int>(MaxLayoutSurfaces)) {
      return false;
    }
    hits.x[hits.nHits] = slot.measurement.frame.u;
    hits.y[hits.nHits] = slot.measurement.frame.v;
    hits.z[hits.nHits] = slot.measurement.frame.q;
    hits.sigmaX2[hits.nHits] = slot.measurement.covariance.uu;
    hits.sigmaY2[hits.nHits] = slot.measurement.covariance.vv;
    ++hits.nHits;
  }
  return hits.nHits >= 2;
}

inline bool linearRegression(int nVal, const double* xVal, const double* yVal, const double* yErr,
                             double& B, double& A) noexcept
{
  double s1 = 0., sxy = 0., sx = 0., sy = 0., sxx = 0.;
  for (int i = 0; i < nVal; ++i) {
    const double invYErr2 = 1. / (yErr[i] * yErr[i]);
    s1 += invYErr2;
    sxy += xVal[i] * yVal[i] * invYErr2;
    sx += xVal[i] * invYErr2;
    sy += yVal[i] * invYErr2;
    sxx += xVal[i] * xVal[i] * invYErr2;
  }
  const double delta = sxx * s1 - sx * sx;
  if (delta == 0.) {
    return false;
  }
  B = (sxy * s1 - sx * sy) / delta;
  A = (sy * sxx - sx * sxy) / delta;
  return true;
}

// Fast Circle Fit on all attached hits (Hansroul, Jeremie, Savard). Matches
// TrackFitter::invQPtFromFCF, including treating stored variances as the
// xErr/yErr inputs of that implementation.
inline float invQPtFromFCF(const DiskRefitHits& hits, float bz) noexcept
{
  constexpr float kFailedInvQPt = 1.f / 100.f;
  const int nPoints = hits.nHits;
  if (nPoints < 2) {
    return kFailedInvQPt;
  }
  std::array<double, MaxLayoutSurfaces> xVal{};
  std::array<double, MaxLayoutSurfaces> yVal{};
  std::array<double, MaxLayoutSurfaces> xErr{};
  std::array<double, MaxLayoutSurfaces> yErr{};
  std::array<double, MaxLayoutSurfaces> uVal{};
  std::array<double, MaxLayoutSurfaces> vVal{};
  std::array<double, MaxLayoutSurfaces> vErr{};
  for (int np = 0; np < nPoints; ++np) {
    xErr[np] = hits.sigmaX2[np];
    yErr[np] = hits.sigmaY2[np];
    if (np > 0) {
      xVal[np] = hits.x[np] - hits.x[0] + xVal[0];
      yVal[np] = hits.y[np] - hits.y[0] + yVal[0];
    } else {
      xVal[np] = 0.00001;
      yVal[np] = 0.;
    }
  }
  for (int i = 0; i < nPoints; ++i) {
    const double x2 = xVal[i] * xVal[i];
    const double y2 = yVal[i] * yVal[i];
    const double invx2y2 = 1. / (x2 + y2);
    uVal[i] = xVal[i] * invx2y2;
    vVal[i] = yVal[i] * invx2y2;
    vErr[i] = std::sqrt(8. * xErr[i] * xErr[i] * x2 * y2 + 2. * yErr[i] * yErr[i] * (x2 - y2) * (x2 - y2)) * invx2y2 * invx2y2;
  }
  double A = 0., B = 0.;
  if (!linearRegression(nPoints, uVal.data(), vVal.data(), vErr.data(), B, A) || A == 0.) {
    return kFailedInvQPt;
  }
  const double b = 1. / (2. * A);
  const double a = -B * b;
  const double r = std::sqrt(a * a + b * b);
  if (!(r > 0.) || bz == 0.f) {
    return kFailedInvQPt;
  }
  const double invpt = 1. / (o2::constants::math::B2C * bz * r);
  const double x = xVal[nPoints - 1];
  const double y = yVal[nPoints - 1];
  const double slope = std::atan2(y, x);
  const double cosSlope = std::cos(slope);
  const double sinSlope = std::sin(slope);
  const double ryRot = a * sinSlope - b * cosSlope;
  const int qfcf = (ryRot > 0.) ? -1 : 1;
  const float invQPt = static_cast<float>(qfcf * invpt);
  return std::isfinite(invQPt) ? invQPt : kFailedInvQPt;
}

// Independent LTF-style seed for one Kalman leg. outward=true starts at the
// innermost hit (MCH matching); false starts at the outermost (vertexing).
inline bool initDiskRefitLeg(SurfaceTrackState& state, const DiskRefitHits& hits, float bz, bool outward) noexcept
{
  if (hits.nHits < 2) {
    return false;
  }
  const bool fieldOn = std::abs(bz) > 0.01f;
  const float invQPt = fieldOn ? invQPtFromFCF(hits, bz) : 0.f;
  const int first = outward ? 0 : hits.nHits - 1;
  const int next = outward ? hits.nHits - 1 : 0;
  const float innerDeltaX = hits.x[1] - hits.x[0];
  const float innerDeltaY = hits.y[1] - hits.y[0];
  const float innerDeltaZ = hits.z[1] - hits.z[0];
  const float innerDeltaR = std::hypot(innerDeltaX, innerDeltaY);
  if (innerDeltaR == 0.f) {
    return false;
  }
  const float tanl0 = fieldOn ? -std::abs(innerDeltaZ / innerDeltaR) : -std::abs(innerDeltaZ) / innerDeltaR;
  if (tanl0 == 0.f) {
    return false;
  }
  const float deltaX = hits.x[first] - hits.x[next];
  const float deltaY = hits.y[first] - hits.y[next];
  const float deltaZ = hits.z[first] - hits.z[next];
  float phi0 = outward ? std::atan2(-deltaY, -deltaX) : std::atan2(deltaY, deltaX);
  if (fieldOn) {
    const float k = std::abs(o2::constants::math::B2C * bz);
    const float fieldSign = std::copysign(1.f, bz);
    phi0 -= 0.5f * fieldSign * invQPt * deltaZ * k / tanl0;
  }
  state.kind = SurfaceKind::Disk;
  state.alpha = 0.f;
  state.referenceCoordinate = hits.z[first];
  state.parameters[0] = hits.x[first];
  state.parameters[1] = hits.y[first];
  state.parameters[2] = phi0;
  state.parameters[3] = tanl0;
  state.parameters[4] = invQPt;
  for (auto& element : state.covariance) {
    element = 0.f;
  }
  state.covariance[packedCovarianceIndex(0, 0)] = 1.f;
  state.covariance[packedCovarianceIndex(1, 1)] = 1.f;
  state.covariance[packedCovarianceIndex(2, 2)] = 1.f;
  state.covariance[packedCovarianceIndex(3, 3)] = 1.f;
  state.covariance[packedCovarianceIndex(4, 4)] = fieldOn ? std::clamp(std::abs(invQPt), 1.f, 10.f) : 0.f;
  return true;
}

} // namespace detail

// Reset a refit leg to a loose diagonal covariance.
GPUhdi() void resetCovarianceForRefit(SurfaceTrackState& state) noexcept
{
  for (auto& element : state.covariance) {
    element = 0.f;
  }
  if (state.kind == SurfaceKind::Cylinder) {
    state.covariance[packedCovarianceIndex(0, 0)] = o2::track::kCY2max;
    state.covariance[packedCovarianceIndex(1, 1)] = o2::track::kCZ2max;
    state.covariance[packedCovarianceIndex(2, 2)] = o2::track::kCSnp2max;
    state.covariance[packedCovarianceIndex(3, 3)] = o2::track::kCTgl2max;
    const float q2pt = state.parameters[4];
    state.covariance[packedCovarianceIndex(4, 4)] = q2pt * q2pt * o2::track::kC1Pt2max;
  } else {
    state.covariance[packedCovarianceIndex(0, 0)] = 1.f;
    state.covariance[packedCovarianceIndex(1, 1)] = 1.f;
    state.covariance[packedCovarianceIndex(2, 2)] = 1.f;
    state.covariance[packedCovarianceIndex(3, 3)] = 1.f;
    state.covariance[packedCovarianceIndex(4, 4)] = std::clamp(std::abs(state.parameters[4]), 1.f, 10.f);
  }
}

// parameters[4] is signed q/pT for both coordinate conventions.
GPUhdi() float ptFromQOverPt(float q2pt, uint8_t absCharge) noexcept
{
  float ptInv = std::abs(q2pt);
  if (ptInv < o2::track::MinPTInv) {
    ptInv = o2::track::MinPTInv;
  }
  if (absCharge > 1) {
    ptInv /= static_cast<float>(absCharge);
  }
  return 1.f / ptInv;
}

// Refit inward, outward, then optionally inward again; commit on success.
inline bool fitTrackSeedLegs(
  const TrackSeed& seed,
  const TimeFrame& frame,
  gsl::span<const gsl::span<const GlobalMeasurement>> layerGlobals,
  SurfaceCatalogView surfaceCatalog,
  float bz,
  bool shiftReferenceToMeasurement,
  float maxChi2ClusterAttachment,
  float maxChi2NDF,
  bool repeatRefitOut,
  gsl::span<const float> minPt,
  SurfaceTrackState& outParamIn,
  SurfaceTrackState& outParamOut,
  float& outChi2,
  float alignResidual = 0.f) noexcept
{
  if (layerGlobals.empty() || layerGlobals.size() > MaxLayoutSurfaces) {
    return false;
  }
  // Legs run sequentially; reuse bounded storage without allocating inside
  // this noexcept refit. Only the active portion is exposed to the assembler.
  std::array<detail::RefitMeasurementSlot, MaxLayoutSurfaces> slotsBuffer{};
  const gsl::span<detail::RefitMeasurementSlot> activeSlots{slotsBuffer.data(), layerGlobals.size()};
  auto legAcceptable = [](const SurfaceTrackState& state, float chi2, uint32_t acceptedHitCount,
                          float maxQoverPt, float maxChi2NDFValue) noexcept -> bool {
    if (!(std::abs(state.parameters[4]) < maxQoverPt)) {
      return false;
    }
    return chi2 < maxChi2NDFValue * static_cast<float>(static_cast<int>(acceptedHitCount) * 2 - 5);
  };

  // Leg A: inner → outer (TrackFitter outward seed).
  SurfaceTrackState stateA = seed.state();
  SurfaceTrackParameters linRefA{stateA};
  float chi2A = 0.f;
  uint32_t acceptedA = 0;
  const int activeSurfaceCount = static_cast<int>(layerGlobals.size());
  bool validSlots = false;
  const auto slotsA = detail::assembleRefitLegSlots(seed, frame, layerGlobals, 0, activeSurfaceCount, 1, activeSlots, validSlots);
  if (!validSlots) {
    return false;
  }
  const bool diskRefit = seed.state().kind == SurfaceKind::Disk;
  detail::DiskRefitHits diskHits{};
  if (diskRefit) {
    if (!detail::collectDiskRefitHits(slotsA, diskHits) ||
        !detail::initDiskRefitLeg(stateA, diskHits, bz, true)) {
      return false;
    }
    linRefA = SurfaceTrackParameters{stateA};
  } else {
    resetCovarianceForRefit(stateA);
  }
  if (!detail::driveRefitLeg(stateA, linRefA, chi2A, acceptedA, slotsA, surfaceCatalog, bz,
                             material::MaterialTraversalDirection::AlongMomentum, shiftReferenceToMeasurement,
                             maxChi2ClusterAttachment, alignResidual)) {
    return false;
  }
  if (!legAcceptable(stateA, chi2A, acceptedA, o2::constants::math::VeryBig, maxChi2NDF)) {
    return false;
  }

  // Leg B: outer → inner (TrackFitter inward seed). Disk legs are inited from
  // hits, not from the previous fitted state.
  SurfaceTrackState stateB = diskRefit ? seed.state() : stateA;
  SurfaceTrackParameters linRefB{stateB};
  float chi2B = 0.f;
  uint32_t acceptedB = 0;
  const auto slotsB = detail::assembleRefitLegSlots(seed, frame, layerGlobals, activeSurfaceCount - 1, -1, -1, activeSlots, validSlots);
  if (!validSlots) {
    return false;
  }
  if (diskRefit) {
    if (!detail::initDiskRefitLeg(stateB, diskHits, bz, false)) {
      return false;
    }
    linRefB = SurfaceTrackParameters{stateB};
  } else {
    resetCovarianceForRefit(stateB);
  }
  if (!detail::driveRefitLeg(stateB, linRefB, chi2B, acceptedB, slotsB, surfaceCatalog, bz,
                             material::MaterialTraversalDirection::OppositeMomentum, shiftReferenceToMeasurement,
                             maxChi2ClusterAttachment, alignResidual)) {
    return false;
  }
  if (!legAcceptable(stateB, chi2B, acceptedB, 50.f, maxChi2NDF)) {
    return false;
  }

  // MinPt uses the seed's attached-cluster count.
  const int nClAttached = seed.getHitLayerMask().count();
  const int minPtSlot = activeSurfaceCount - nClAttached;
  if (minPtSlot >= 0 && minPtSlot < static_cast<int>(minPt.size())) {
    const float minPtThreshold = minPt[minPtSlot];
    if (minPtThreshold > 0.f && ptFromQOverPt(stateB.parameters[4], stateB.absCharge) < minPtThreshold) {
      return false;
    }
  }

  // Optional leg C: inward again.
  SurfaceTrackState stateOut = stateA;
  if (repeatRefitOut) {
    SurfaceTrackState stateC = diskRefit ? seed.state() : stateB;
    SurfaceTrackParameters linRefC{stateC};
    float chi2C = 0.f;
    uint32_t acceptedC = 0;
    const auto slotsC = detail::assembleRefitLegSlots(seed, frame, layerGlobals, 0, activeSurfaceCount, 1, activeSlots, validSlots);
    if (!validSlots) {
      return false;
    }
    if (diskRefit) {
      if (!detail::initDiskRefitLeg(stateC, diskHits, bz, true)) {
        return false;
      }
      linRefC = SurfaceTrackParameters{stateC};
    } else {
      resetCovarianceForRefit(stateC);
    }
    if (!detail::driveRefitLeg(stateC, linRefC, chi2C, acceptedC, slotsC, surfaceCatalog, bz,
                               material::MaterialTraversalDirection::AlongMomentum, shiftReferenceToMeasurement,
                               maxChi2ClusterAttachment, alignResidual)) {
      return false;
    }
    if (!legAcceptable(stateC, chi2C, acceptedC, o2::constants::math::VeryBig, maxChi2NDF)) {
      return false;
    }
    stateOut = stateC;
  }

  outParamIn = stateB;
  outParamOut = stateOut;
  outChi2 = chi2B;
  return true;
}

} // namespace o2::itsmft::tracking

#endif // GPUCA_GPUCODE

#endif /* ALICEO2_ITSMFT_TRACKING_REFITDRIVER_H_ */
