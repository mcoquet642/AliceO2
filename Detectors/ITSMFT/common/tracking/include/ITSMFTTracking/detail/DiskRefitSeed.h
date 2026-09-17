// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ITSMFT_TRACKING_DETAIL_DISKREFITSEED_H_
#define ALICEO2_ITSMFT_TRACKING_DETAIL_DISKREFITSEED_H_

#include <algorithm>
#include <array>
#include <cmath>

#include <gsl/span>

#include "CommonConstants/MathConstants.h"
#include "ITSMFTTracking/IdTypes.h"
#include "ITSMFTTracking/SurfaceMeasurement.h"
#include "ITSMFTTracking/SurfaceTrackState.h"

namespace o2::itsmft::tracking::detail
{

struct DiskHit {
  float x{0.f};
  float y{0.f};
  float z{0.f};
  float sigmaX2{0.f};
  float sigmaY2{0.f};
};

inline bool linearRegression(int nVal, const double* xVal, const double* yVal, const double* yErr,
                             double& B, double& A) noexcept
{
  if (nVal < 2) {
    return false;
  }
  double S1 = 0.;
  double SXY = 0.;
  double SX = 0.;
  double SY = 0.;
  double SXX = 0.;
  for (int i = 0; i < nVal; ++i) {
    const double err = yErr[i];
    if (!(err > 0.) || !std::isfinite(err)) {
      return false;
    }
    const double invYErr2 = 1. / (err * err);
    if (!std::isfinite(invYErr2)) {
      return false;
    }
    S1 += invYErr2;
    SXY += xVal[i] * yVal[i] * invYErr2;
    SX += xVal[i] * invYErr2;
    SY += yVal[i] * invYErr2;
    SXX += xVal[i] * xVal[i] * invYErr2;
  }
  const double delta = SXX * S1 - SX * SX;
  if (delta == 0.) {
    return false;
  }
  B = (SXY * S1 - SX * SY) / delta;
  A = (SY * SXX - SX * SXY) / delta;
  return std::isfinite(A) && std::isfinite(B);
}

// TrackFitter::invQPtFromFCF (Hansroul, Jeremie, Savard). Failure falls back
// to 1/100, matching standalone LTF.
inline double invQPtFromFCF(const DiskHit* hits, int nPoints, double bFieldZ) noexcept
{
  if (nPoints < 2 || bFieldZ == 0.) {
    return 1. / 100.;
  }
  std::array<double, MaxLayoutSurfaces> xVal{};
  std::array<double, MaxLayoutSurfaces> yVal{};
  std::array<double, MaxLayoutSurfaces> uVal{};
  std::array<double, MaxLayoutSurfaces> vVal{};
  std::array<double, MaxLayoutSurfaces> vErr{};
  for (int np = 0; np < nPoints; ++np) {
    if (np > 0) {
      xVal[np] = hits[np].x - hits[0].x + xVal[0];
      yVal[np] = hits[np].y - hits[0].y + yVal[0];
    } else {
      xVal[np] = .00001;
      yVal[np] = 0.;
    }
  }
  for (int i = 0; i < nPoints; ++i) {
    const double x2 = xVal[i] * xVal[i];
    const double y2 = yVal[i] * yVal[i];
    const double invx2y2 = 1. / (x2 + y2);
    uVal[i] = xVal[i] * invx2y2;
    vVal[i] = yVal[i] * invx2y2;
    const double xErr = hits[i].sigmaX2;
    const double yErr = hits[i].sigmaY2;
    vErr[i] = std::sqrt(8. * xErr * xErr * x2 * y2 + 2. * yErr * yErr * (x2 - y2) * (x2 - y2)) * invx2y2 * invx2y2;
  }
  double A = 0.;
  double B = 0.;
  if (!linearRegression(nPoints, uVal.data(), vVal.data(), vErr.data(), B, A) || A == 0.) {
    return 1. / 100.;
  }
  const double b = 1. / (2. * A);
  const double a = -B * b;
  const double r = std::sqrt(a * a + b * b);
  if (!(r > 0.) || !std::isfinite(r)) {
    return 1. / 100.;
  }
  const double invpt = 1. / (o2::constants::math::B2C * bFieldZ * r);
  const double x = xVal[nPoints - 1];
  const double y = yVal[nPoints - 1];
  const double slope = std::atan2(y, x);
  const double cosSlope = std::cos(slope);
  const double sinSlope = std::sin(slope);
  const double ryRot = a * sinSlope - b * cosSlope;
  const int qfcf = (ryRot > 0.) ? -1 : +1;
  const double invqpt = static_cast<double>(qfcf) * invpt;
  return std::isfinite(invqpt) ? invqpt : 1. / 100.;
}

inline void setDiskLTFCovariance(SurfaceTrackState& state, bool magnetOn) noexcept
{
  for (auto& element : state.covariance) {
    element = 0.f;
  }
  state.covariance[packedCovarianceIndex(0, 0)] = 1.f;
  state.covariance[packedCovarianceIndex(1, 1)] = 1.f;
  state.covariance[packedCovarianceIndex(2, 2)] = 1.f;
  state.covariance[packedCovarianceIndex(3, 3)] = 1.f;
  if (magnetOn) {
    const float qptsigma = std::clamp(std::abs(state.parameters[4]), 1.f, 10.f);
    state.covariance[packedCovarianceIndex(4, 4)] = qptsigma;
  }
}

// TrackFitter::initTrack for vertexing: FCF q/pT, tanλ from the two innermost
// hits, helix φ at the outermost hit, LTF diagonal covariance.
template <typename Slot>
inline bool initDiskRefitState(SurfaceTrackState& state, gsl::span<const Slot> innerToOuter, float bz) noexcept
{
  std::array<DiskHit, MaxLayoutSurfaces> hits{};
  int nPoints = 0;
  for (const auto& slot : innerToOuter) {
    if (!slot.present) {
      continue;
    }
    if (nPoints >= static_cast<int>(MaxLayoutSurfaces)) {
      return false;
    }
    const auto& frame = slot.measurement.frame;
    const auto& cov = slot.measurement.covariance;
    if (!std::isfinite(frame.u) || !std::isfinite(frame.v) || !std::isfinite(frame.q) ||
        !std::isfinite(cov.uu) || !std::isfinite(cov.vv) || cov.uu < 0.f || cov.vv < 0.f) {
      return false;
    }
    hits[nPoints] = DiskHit{frame.u, frame.v, frame.q, cov.uu, cov.vv};
    ++nPoints;
  }
  if (nPoints < 2) {
    return false;
  }

  const auto& inner = hits[0];
  const auto& nextInner = hits[1];
  const auto& outer = hits[nPoints - 1];
  const float tanlDx = nextInner.x - inner.x;
  const float tanlDy = nextInner.y - inner.y;
  const float tanlDz = nextInner.z - inner.z;
  const float tanlDr = std::hypot(tanlDx, tanlDy);
  if (!(tanlDr > 0.f) || !std::isfinite(tanlDr)) {
    return false;
  }
  const float tanl0 = -std::abs(tanlDz / tanlDr);
  if (tanl0 == 0.f || !std::isfinite(tanl0)) {
    return false;
  }

  const float deltaX = outer.x - inner.x;
  const float deltaY = outer.y - inner.y;
  const float deltaZ = outer.z - inner.z;
  const bool magnetOn = std::abs(bz) > o2::constants::math::Almost0;
  float invQPt0 = 0.f;
  float phi0 = std::atan2(deltaY, deltaX);
  if (magnetOn) {
    invQPt0 = static_cast<float>(invQPtFromFCF(hits.data(), nPoints, bz));
    const float k = std::abs(o2::constants::math::B2C * bz);
    const float Hz = std::copysign(1.f, bz);
    phi0 -= 0.5f * Hz * invQPt0 * deltaZ * k / tanl0;
  }
  if (!std::isfinite(phi0) || !std::isfinite(invQPt0)) {
    return false;
  }

  state.referenceCoordinate = outer.z;
  state.parameters[0] = outer.x;
  state.parameters[1] = outer.y;
  state.parameters[2] = phi0;
  state.parameters[3] = tanl0;
  state.parameters[4] = invQPt0;
  state.kind = SurfaceKind::Disk;
  state.alpha = 0.f;
  setDiskLTFCovariance(state, magnetOn);
  return true;
}

} // namespace o2::itsmft::tracking::detail

#endif /* ALICEO2_ITSMFT_TRACKING_DETAIL_DISKREFITSEED_H_ */
