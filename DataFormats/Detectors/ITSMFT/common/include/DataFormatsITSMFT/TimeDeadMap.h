#ifndef ALICEO2_ITSMFT_TIMEDEADMAP_H
#define ALICEO2_ITSMFT_TIMEDEADMAP_H

#include "Rtypes.h" 
#include "DetectorsCommonDataFormats/DetID.h"
#include <iostream>
#include <vector>
#include <map>

namespace o2 {

namespace itsmft {

class TimeDeadMap {
public:
  // Constructor
  TimeDeadMap(std::map<unsigned long, std::vector<uint16_t>>& deadmap)
  {
    mDeadMap.swap(deadmap);
  }

  /// Constructor
  TimeDeadMap() = default;
  /// Destructor
  ~TimeDeadMap() = default;

  void fillMap(unsigned long firstOrbit, std::vector<uint16_t> deadVect)
  {
    mDeadMap[firstOrbit] = deadVect;
  };

  void clear()
  {
    mDeadMap.clear();
  }

  // TODO: use o2::itsmft::ChipMappingITS functionalities
  std::vector<int> expandITSCable(int lane)
  { // returns list of chips in the lane

    int NCHIPIB = 432;
    int firstchip = NCHIPIB + 7 * (lane - NCHIPIB) * (lane >= NCHIPIB);
    int vecsize = (lane < NCHIPIB) ? 1 : 7;
    std::vector<int> chipList(vecsize);
    std::generate(chipList.begin(), chipList.end(), [&firstchip]() { return firstchip++; });
    return chipList;
  };

  void decodeITSMFTTimeDeadMap(unsigned long orbit, o2::itsmft::NoiseMap* noisemap, o2::detectors::DetID detID)
  {

    if (mMAP_VERSION != "2") {
      LOG(error) << "Trying to decode time-dependent deadmap version " << mMAP_VERSION << ". Not implemented, doing nothing.";
      return;
    }

    if (mDeadMap.empty()) {
      LOG(warning) << "Time-dependent dead map is empty. Doing nothing.";
      return;
    } else if (orbit > mDeadMap.rbegin()->first + 11000 * 300 || orbit < mDeadMap.begin()->first - 11000 * 300) {
      // the map should not leave several minutes uncovered.
      LOG(warning) << "Time-dependent dead map: the requested orbit " << orbit << " seems to be out of the range stored in the map.";
    }

    auto closestVec = mDeadMap.upper_bound(orbit)->second;

    // vector encoding: if 1<<15 = 0x8000 is set, the word encodes the first element of a range, with mask (1<<15)-1 = 0x7FFF. The last element of the range is the next in the vector.

    // MFT specific: map entry = chip ID
    if (detID == o2::detectors::DetID::MFT) {
      for (int iel = 0; iel < closestVec.size(); iel++) {
        uint16_t w = closestVec.at(iel);
        noisemap->maskFullChip(w & 0x7FFF);
        if (w & 0x8000) {
          for (int w2 = (w & 0x7FFF) + 1; w2 < closestVec.at(iel + 1); w2++) {
            noisemap->maskFullChip(w2);
          }
        }
      }
    }
    // ITS specific: map entry = lane ID.
    // element w is replaced with loop over vector expandITSCable(w)
    else if (detID == o2::detectors::DetID::ITS) {
      for (int iel = 0; iel < closestVec.size(); iel++) {
        uint16_t w = closestVec.at(iel);
        for (int ichip : expandITSCable(w & 0x7FFF)) {
          noisemap->maskFullChip(ichip);
        }
        if (w & 0x8000) {
          for (int w2 = (w & 0x7FFF) + 1; w2 < closestVec.at(iel + 1); w2++) {
            for (int ichip : expandITSCable(w2)) {
              noisemap->maskFullChip(ichip);
            }
          }
        }
      }
    }
  };

  std::string getMapVersion() { return mMAP_VERSION; };

private:
  std::string mMAP_VERSION = "2";
  std::map<unsigned long, std::vector<uint16_t>> mDeadMap; ///< Internal dead pixel map representation

  ClassDefNV(TimeDeadMap, 2);
};

} // namespace itsmft
} // namespace o2

#endif /* ALICEO2_ITSMFT_TIMEDEADMAP_H */

