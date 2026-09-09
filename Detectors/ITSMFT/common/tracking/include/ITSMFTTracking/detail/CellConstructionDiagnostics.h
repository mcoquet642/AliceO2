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
/// \brief Opt-in cell-construction diagnostic tree (no ROOT use in static dtors)
///

#ifndef ALICEO2_ITSMFT_TRACKING_CELLCONSTRUCTIONDIAGNOSTICS_H_
#define ALICEO2_ITSMFT_TRACKING_CELLCONSTRUCTIONDIAGNOSTICS_H_

#include <mutex>
#include <string>
#include <vector>

#include "TFile.h"
#include "TTree.h"

#include "Framework/Logger.h"

namespace o2::itsmft::tracking::detail
{

/// Process-wide accumulator for cell-construction diagnostics.
/// Samples are POD-only during tracking; ROOT I/O runs only via explicit flush().
class CellConstructionDiagnostics
{
 public:
  struct Sample {
    float absDeltaTanLambda{0.f};
    float absDeltaLambda{0.f};
    float absDeltaPhi{0.f};
    float phiTolerance{0.f};
    float edgeMSAngle{0.f};
  };

  static CellConstructionDiagnostics& instance()
  {
    static CellConstructionDiagnostics diagnostics;
    return diagnostics;
  }

  void fill(float absDeltaTanLambda, float absDeltaLambda, float absDeltaPhi,
            float phiTolerance, float edgeMSAngle)
  {
    std::lock_guard lock{mMutex};
    mSamples.push_back({absDeltaTanLambda, absDeltaLambda, absDeltaPhi, phiTolerance, edgeMSAngle});
  }

  /// Append accumulated samples to TTree cellDiag and clear the buffer.
  /// Must be called while ROOT is still alive (e.g. end of Tracker::run).
  void flush()
  {
    std::vector<Sample> samples;
    {
      std::lock_guard lock{mMutex};
      if (mSamples.empty()) {
        return;
      }
      samples.swap(mSamples);
    }

    const char* mode = mFileInitialized ? "UPDATE" : "RECREATE";
    TFile out{mOutputFile.c_str(), mode};
    if (out.IsZombie()) {
      LOGP(error, "Failed to write cell diagnostics to {}", mOutputFile);
      std::lock_guard lock{mMutex};
      mSamples.insert(mSamples.end(), samples.begin(), samples.end());
      return;
    }

    Sample row{};
    TTree* tree = mFileInitialized ? out.Get<TTree>("cellDiag") : nullptr;
    if (tree == nullptr) {
      out.cd();
      tree = new TTree("cellDiag", "MFT cell-construction observables");
      tree->Branch("absDeltaTanLambda", &row.absDeltaTanLambda);
      tree->Branch("absDeltaLambda", &row.absDeltaLambda);
      tree->Branch("absDeltaPhi", &row.absDeltaPhi);
      tree->Branch("phiTolerance", &row.phiTolerance);
      tree->Branch("edgeMSAngle", &row.edgeMSAngle);
      mFileInitialized = true;
    } else {
      tree->SetBranchAddress("absDeltaTanLambda", &row.absDeltaTanLambda);
      tree->SetBranchAddress("absDeltaLambda", &row.absDeltaLambda);
      tree->SetBranchAddress("absDeltaPhi", &row.absDeltaPhi);
      tree->SetBranchAddress("phiTolerance", &row.phiTolerance);
      tree->SetBranchAddress("edgeMSAngle", &row.edgeMSAngle);
    }

    for (const auto& sample : samples) {
      row = sample;
      tree->Fill();
    }
    out.cd();
    tree->Write(nullptr, TObject::kOverwrite);
    out.Close();
    LOGP(info, "Appended {} cell-construction samples to {} (TTree cellDiag)", samples.size(), mOutputFile);
  }

  CellConstructionDiagnostics(const CellConstructionDiagnostics&) = delete;
  CellConstructionDiagnostics& operator=(const CellConstructionDiagnostics&) = delete;

 private:
  CellConstructionDiagnostics() = default;
  // Intentionally empty: never touch ROOT from a static destructor.
  ~CellConstructionDiagnostics() = default;

  std::mutex mMutex;
  std::vector<Sample> mSamples;
  std::string mOutputFile{"mft_cell_diagnostics.root"};
  bool mFileInitialized{false};
};

} // namespace o2::itsmft::tracking::detail

#endif /* ALICEO2_ITSMFT_TRACKING_CELLCONSTRUCTIONDIAGNOSTICS_H_ */
