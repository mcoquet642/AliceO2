// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DCAFitterN.cxx
/// \brief Defintions for N-prongs secondary vertex fit
/// \author ruben.shahoyan@cern.ch

#include "DetectorsVertexing/FwdDCAFitterN.h"

namespace o2
{
namespace vertexing
{

void __dummy_instance__()
{
  FwdDCAFitter2 ft2;
  FwdDCAFitter3 ft3;
  o2::track::TrackParCovFwd tr;
  ft2.process(tr, tr);
  ft3.process(tr, tr, tr);
}

} // namespace vertexing
} // namespace o2
