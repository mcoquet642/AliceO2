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
/// \file CellConstructionDiagnostics.h
/// \brief Opt-in ROOT histograms for MFT/ITS cell-construction observables
///

#ifndef ALICEO2_ITSMFT_TRACKING_CELLCONSTRUCTIONDIAGNOSTICS_H_
#define ALICEO2_ITSMFT_TRACKING_CELLCONSTRUCTIONDIAGNOSTICS_H_

#include <memory>
#include <mutex>
#include <string>

#include "TFile.h"
#include "TH1F.h"

#include "Framework/Logger.h"

namespace o2::itsmft::tracking::detail
{

/// Process-wide accumulator for cell-construction diagnostic histograms.
/// Fill is thread-safe; histograms are written once on destruction when any fill occurred.
class CellConstructionDiagnostics
{
 public:
  static CellConstructionDiagnostics& instance()
  {
    static CellConstructionDiagnostics diagnostics;
    return diagnostics;
  }

  void fill(float absDeltaTanLambda, float absDeltaLambda, float absDeltaPhi,
            float phiTolerance, float edgeMSAngle)
  {
    std::lock_guard lock{mMutex};
    ensureHistograms();
    mDeltaTanLambda->Fill(absDeltaTanLambda);
    mDeltaLambda->Fill(absDeltaLambda);
    mDeltaPhi->Fill(absDeltaPhi);
    mPhiTolerance->Fill(phiTolerance);
    mEdgeMSAngle->Fill(edgeMSAngle);
    mFilled = true;
  }

  CellConstructionDiagnostics(const CellConstructionDiagnostics&) = delete;
  CellConstructionDiagnostics& operator=(const CellConstructionDiagnostics&) = delete;

 private:
  CellConstructionDiagnostics() = default;

  ~CellConstructionDiagnostics()
  {
    writeIfNeeded();
  }

  void ensureHistograms()
  {
    if (mDeltaTanLambda) {
      return;
    }
    mDeltaTanLambda = std::make_unique<TH1F>("hDeltaTanLambda",
                                             "|#Delta tan#lambda|;|#Delta tan#lambda|;entries",
                                             200, 0.f, 0.2f);
    mDeltaLambda = std::make_unique<TH1F>("hDeltaLambda",
                                          "|#Delta#lambda|;|#Delta#lambda| [rad];entries",
                                          200, 0.f, 0.2f);
    mDeltaPhi = std::make_unique<TH1F>("hDeltaPhi",
                                       "|#Delta#varphi|;|#Delta#varphi| [rad];entries",
                                       200, 0.f, 0.5f);
    mPhiTolerance = std::make_unique<TH1F>("hPhiTolerance",
                                           "#theta_{bend}+MS/|cos#lambda|;tolerance [rad];entries",
                                           200, 0.f, 1.f);
    mEdgeMSAngle = std::make_unique<TH1F>("hEdgeMSAngle",
                                          "edge MS angle;MS angle [rad];entries",
                                          200, 0.f, 0.05f);
  }

  void writeIfNeeded()
  {
    std::lock_guard lock{mMutex};
    if (!mFilled || !mDeltaTanLambda) {
      return;
    }
    TFile out{mOutputFile.c_str(), "RECREATE"};
    if (out.IsZombie()) {
      LOGP(error, "Failed to write cell diagnostics to {}", mOutputFile);
      return;
    }
    mDeltaTanLambda->Write();
    mDeltaLambda->Write();
    mDeltaPhi->Write();
    mPhiTolerance->Write();
    mEdgeMSAngle->Write();
    out.Close();
    LOGP(info, "Wrote cell-construction diagnostics to {}", mOutputFile);
  }

  std::mutex mMutex;
  bool mFilled{false};
  std::string mOutputFile{"mft_cell_diagnostics.root"};
  std::unique_ptr<TH1F> mDeltaTanLambda;
  std::unique_ptr<TH1F> mDeltaLambda;
  std::unique_ptr<TH1F> mDeltaPhi;
  std::unique_ptr<TH1F> mPhiTolerance;
  std::unique_ptr<TH1F> mEdgeMSAngle;
};

} // namespace o2::itsmft::tracking::detail

#endif /* ALICEO2_ITSMFT_TRACKING_CELLCONSTRUCTIONDIAGNOSTICS_H_ */
