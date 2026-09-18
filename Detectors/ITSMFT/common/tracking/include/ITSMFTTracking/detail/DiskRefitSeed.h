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
  float sigmaX2{0.f}; // variance of x [cm²]
  float sigmaY2{0.f}; // variance of y [cm²]
};

struct InvQPtSeed {
  double invQPt{1. / 100.};
  double variance{0.}; // Var(invQPt) [(GeV/c)⁻²]
  bool fromFCF{false};
};

inline float transverseVariance(const DiskHit& hit) noexcept
{
  return 0.5f * (hit.sigmaX2 + hit.sigmaY2);
}

inline float chordPerpVariance(const DiskHit& a, const DiskHit& b) noexcept
{
  return 0.5f * (transverseVariance(a) + transverseVariance(b));
}

inline float finiteVarianceFloor(float value, float floor) noexcept
{
  if (!std::isfinite(value) || value < floor) {
    return floor;
  }
  return value;
}

// Weighted linear regression y = B * x + A. yErr[i] is the standard deviation
// of y[i]. Parameter variances come from the weighted normal equations.
inline bool linearRegression(int nVal, const double* xVal, const double* yVal, const double* yErr,
                             double& B, double& Berr, double& A, double& Aerr) noexcept
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
  if (!(std::abs(delta) > 0.) || !std::isfinite(delta)) {
    return false;
  }
  B = (SXY * S1 - SX * SY) / delta;
  A = (SY * SXX - SX * SXY) / delta;
  // Cov([B,A]) = [[S1, -SX], [-SX, SXX]] / delta for the weighted design.
  const double varB = S1 / delta;
  const double varA = SXX / delta;
  if (!(varB >= 0.) || !(varA >= 0.) || !std::isfinite(varB) || !std::isfinite(varA)) {
    return false;
  }
  Berr = std::sqrt(varB);
  Aerr = std::sqrt(varA);
  return std::isfinite(A) && std::isfinite(B);
}

inline double geometricInvQPtVariance(const DiskHit* hits, int nPoints, double bFieldZ) noexcept
{
  if (nPoints < 2 || !(std::abs(bFieldZ) > 0.)) {
    return 0.;
  }
  const auto& inner = hits[0];
  const auto& outer = hits[nPoints - 1];
  const double lxy = std::hypot(static_cast<double>(outer.x - inner.x), static_cast<double>(outer.y - inner.y));
  if (!(lxy > 0.) || !std::isfinite(lxy)) {
    return 0.;
  }
  const double sigmaPerp = std::sqrt(static_cast<double>(chordPerpVariance(inner, outer)));
  const double k = std::abs(o2::constants::math::B2C * bFieldZ);
  if (!(k > 0.) || !std::isfinite(sigmaPerp)) {
    return 0.;
  }
  // Sagitta scale: σ(1/R) ∼ σ_⊥ / L², then σ(q/pT) = σ(1/R) / |B2C Bz|.
  const double sigmaInvR = sigmaPerp / (lxy * lxy);
  const double sigmaInvQPt = sigmaInvR / k;
  const double variance = sigmaInvQPt * sigmaInvQPt;
  return std::isfinite(variance) ? variance : 0.;
}

// Hansroul / Jeremie / Savard fast circle fit. Hit sigmaX2/sigmaY2 are variances.
// Conformal v-errors use σ = √variance (unlike historical TrackFitter which fed
// variances into a formula that squared them again).
inline InvQPtSeed invQPtFromFCF(const DiskHit* hits, int nPoints, double bFieldZ) noexcept
{
  InvQPtSeed seed{};
  seed.variance = geometricInvQPtVariance(hits, nPoints, bFieldZ);
  if (nPoints < 2 || !(std::abs(bFieldZ) > 0.)) {
    return seed;
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
    const double sigmaX = std::sqrt(std::max(0., static_cast<double>(hits[i].sigmaX2)));
    const double sigmaY = std::sqrt(std::max(0., static_cast<double>(hits[i].sigmaY2)));
    // Error propagation of v = y / (x²+y²) with independent σ_x, σ_y.
    vErr[i] = std::sqrt(8. * sigmaX * sigmaX * x2 * y2 + 2. * sigmaY * sigmaY * (x2 - y2) * (x2 - y2)) * invx2y2 * invx2y2;
    if (!(vErr[i] > 0.) || !std::isfinite(vErr[i])) {
      seed.invQPt = 1. / 100.;
      return seed;
    }
  }

  double A = 0.;
  double B = 0.;
  double Aerr = 0.;
  double Berr = 0.;
  if (!linearRegression(nPoints, uVal.data(), vVal.data(), vErr.data(), B, Berr, A, Aerr) || A == 0.) {
    seed.invQPt = 1. / 100.;
    return seed;
  }

  const double b = 1. / (2. * A);
  const double a = -B * b;
  const double r2 = a * a + b * b;
  const double r = std::sqrt(r2);
  if (!(r > 0.) || !std::isfinite(r)) {
    seed.invQPt = 1. / 100.;
    return seed;
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
  if (!std::isfinite(invqpt)) {
    seed.invQPt = 1. / 100.;
    return seed;
  }

  // r² = a² + b², b = 1/(2A), a = -B b.
  // ∂b/∂A = -1/(2 A²), ∂a/∂A = -B ∂b/∂A, ∂a/∂B = -b, ∂b/∂B = 0.
  const double db_dA = -1. / (2. * A * A);
  const double da_dA = -B * db_dA;
  const double da_dB = -b;
  const double dr_dA = (a * da_dA + b * db_dA) / r;
  const double dr_dB = (a * da_dB) / r;
  const double sigmaR2 = dr_dA * dr_dA * Aerr * Aerr + dr_dB * dr_dB * Berr * Berr;
  double variance = invqpt * invqpt * (sigmaR2 / (r * r));
  if (!(variance > 0.) || !std::isfinite(variance)) {
    variance = seed.variance;
  }

  seed.invQPt = invqpt;
  seed.variance = variance;
  seed.fromFCF = true;
  return seed;
}

// Diagonal seed covariance from hit resolutions, chord geometry, and FCF Var(q/pT).
// Position variances are taken at seedHitIndex (the plane where the state lives).
// No MCS prior — material is applied during Kalman propagation.
inline void setDiskPhysicalSeedCovariance(SurfaceTrackState& state, const DiskHit* hits, int nPoints,
                                          float /*bz*/, bool magnetOn, double invQPtVariance,
                                          int seedHitIndex = -1) noexcept
{
  for (auto& element : state.covariance) {
    element = 0.f;
  }
  if (nPoints < 2) {
    return;
  }
  if (seedHitIndex < 0 || seedHitIndex >= nPoints) {
    seedHitIndex = nPoints - 1;
  }

  const auto& inner = hits[0];
  const auto& nextInner = hits[1];
  const auto& outer = hits[nPoints - 1];
  const auto& seedHit = hits[seedHitIndex];

  const float meanHitVar = std::max(1e-8f, 0.25f * (transverseVariance(inner) + transverseVariance(outer) +
                                                    transverseVariance(nextInner) + seedHit.sigmaX2));
  const float posFloor = 1e-4f * meanHitVar;
  const float angleFloor = 1e-8f;

  state.covariance[packedCovarianceIndex(0, 0)] = finiteVarianceFloor(seedHit.sigmaX2, posFloor);
  state.covariance[packedCovarianceIndex(1, 1)] = finiteVarianceFloor(seedHit.sigmaY2, posFloor);

  // φ from outer−inner chord: Var(φ) ≈ σ_⊥² / L_xy².
  const float phiDx = outer.x - inner.x;
  const float phiDy = outer.y - inner.y;
  const float phiLxy2 = phiDx * phiDx + phiDy * phiDy;
  float varPhi = angleFloor;
  if (phiLxy2 > 0.f) {
    varPhi = finiteVarianceFloor(chordPerpVariance(outer, inner) / phiLxy2, angleFloor);
  }
  state.covariance[packedCovarianceIndex(2, 2)] = varPhi;

  // tanλ = −|Δz|/Δr from the two innermost hits; Δz treated as exact.
  // Var(tanλ) ≈ tanλ² · (σ_⊥² / L_xy²).
  const float tanlDx = nextInner.x - inner.x;
  const float tanlDy = nextInner.y - inner.y;
  const float tanlLxy2 = tanlDx * tanlDx + tanlDy * tanlDy;
  const float tanl = state.parameters[3];
  float varTanl = angleFloor;
  if (tanlLxy2 > 0.f) {
    varTanl = finiteVarianceFloor(tanl * tanl * (chordPerpVariance(nextInner, inner) / tanlLxy2), angleFloor);
  }
  state.covariance[packedCovarianceIndex(3, 3)] = varTanl;

  if (magnetOn) {
    const float qFloor = 1e-4f * meanHitVar; // numerical floor only, not |q/pT|-scaled prior
    state.covariance[packedCovarianceIndex(4, 4)] =
      finiteVarianceFloor(static_cast<float>(invQPtVariance), qFloor);
  }
}

// Keep filtered parameters; replace covariance with a physical prior at the hit
// nearest the current reference z (start of the next Kalman pass).
inline void resetDiskCovarianceForNextPass(SurfaceTrackState& state, const DiskHit* hits, int nPoints,
                                           float bz) noexcept
{
  if (nPoints < 2) {
    return;
  }
  int seedHitIndex = nPoints - 1;
  float bestDz = std::abs(hits[seedHitIndex].z - state.referenceCoordinate);
  for (int i = 0; i < nPoints; ++i) {
    const float dz = std::abs(hits[i].z - state.referenceCoordinate);
    if (dz < bestDz) {
      bestDz = dz;
      seedHitIndex = i;
    }
  }
  const bool magnetOn = std::abs(bz) > o2::constants::math::Almost0;
  double invQPtVariance = 0.;
  if (magnetOn) {
    invQPtVariance = invQPtFromFCF(hits, nPoints, bz).variance;
  }
  setDiskPhysicalSeedCovariance(state, hits, nPoints, bz, magnetOn, invQPtVariance, seedHitIndex);
}

template <typename Slot>
inline bool collectDiskHits(gsl::span<const Slot> innerToOuter, DiskHit* hits, int& nPoints) noexcept
{
  nPoints = 0;
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
  return nPoints >= 2;
}

// TrackFitter::initTrack for vertexing: FCF q/pT, tanλ from the two innermost
// hits, helix φ at the outermost hit, physically derived diagonal covariance.
template <typename Slot>
inline bool initDiskRefitState(SurfaceTrackState& state, gsl::span<const Slot> innerToOuter, float bz) noexcept
{
  std::array<DiskHit, MaxLayoutSurfaces> hits{};
  int nPoints = 0;
  if (!collectDiskHits(innerToOuter, hits.data(), nPoints)) {
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
  double invQPtVariance = 0.;
  float phi0 = std::atan2(deltaY, deltaX);
  if (magnetOn) {
    const InvQPtSeed qSeed = invQPtFromFCF(hits.data(), nPoints, bz);
    invQPt0 = static_cast<float>(qSeed.invQPt);
    invQPtVariance = qSeed.variance;
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
  setDiskPhysicalSeedCovariance(state, hits.data(), nPoints, bz, magnetOn, invQPtVariance, nPoints - 1);
  return true;
}

} // namespace o2::itsmft::tracking::detail

#endif /* ALICEO2_ITSMFT_TRACKING_DETAIL_DISKREFITSEED_H_ */
