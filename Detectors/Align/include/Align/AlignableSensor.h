// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AlignableSensor.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  End-chain alignment volume in detector branch, where the actual measurement is done.

#ifndef ALIGNABLESENSOR_H
#define ALIGNABLESENSOR_H

#include <TMath.h>
#include <TObjArray.h>

#include "Align/AlignableVolume.h"
#include "Align/DOFStatistics.h"
#include "Align/utils.h"

//class AliTrackPointArray;
//class AliESDtrack;
class TCloneArray;

namespace o2
{
namespace align
{

class AlignableDetector;
class AlignmentPoint;

class AlignableSensor : public AlignableVolume
{
 public:
  //
  AlignableSensor(const char* name = 0, int vid = 0, int iid = 0);
  virtual ~AlignableSensor();
  //
  virtual void addChild(AlignableVolume*);
  //
  void setDetector(AlignableDetector* det) { mDet = det; }
  AlignableDetector* getDetector() const { return mDet; }
  //
  int getSID() const { return mSID; }
  void setSID(int s) { mSID = s; }
  //
  void incrementStat() { mNProcPoints++; }
  //
  // derivatives calculation
  virtual void dPosTraDParCalib(const AlignmentPoint* pnt, double* deriv, int calibID, const AlignableVolume* parent = 0) const;
  virtual void dPosTraDParGeom(const AlignmentPoint* pnt, double* deriv, const AlignableVolume* parent = 0) const;
  //
  virtual void dPosTraDParGeomLOC(const AlignmentPoint* pnt, double* deriv) const;
  virtual void dPosTraDParGeomTRA(const AlignmentPoint* pnt, double* deriv) const;
  virtual void dPosTraDParGeomLOC(const AlignmentPoint* pnt, double* deriv, const AlignableVolume* parent) const;
  virtual void dPosTraDParGeomTRA(const AlignmentPoint* pnt, double* deriv, const AlignableVolume* parent) const;
  //
  void getModifiedMatrixT2LmodLOC(TGeoHMatrix& matMod, const double* delta) const;
  void getModifiedMatrixT2LmodTRA(TGeoHMatrix& matMod, const double* delta) const;
  //
  virtual void applyAlignmentFromMPSol();
  //
  void setAddError(double y, double z)
  {
    mAddError[0] = y;
    mAddError[1] = z;
  }
  const double* getAddError() const { return mAddError; }
  //
  virtual void prepareMatrixT2L();
  //
  virtual void setTrackingFrame();
  virtual bool isSensor() const { return true; }
  virtual void Print(const Option_t* opt = "") const;
  //
  virtual void updatePointByTrackInfo(AlignmentPoint* pnt, const trackParam_t* t) const;
  virtual void updateL2GRecoMatrices(const TClonesArray* algArr, const TGeoHMatrix* cumulDelta);
  //
  //  virtual AlignmentPoint* TrackPoint2AlgPoint(int pntId, const AliTrackPointArray* trpArr, const AliESDtrack* t) = 0; TODO(milettri): needs AliTrackPointArray AliESDtrack
  //
  virtual int finalizeStat(DOFStatistics* h = 0);
  //
  virtual void prepareMatrixClAlg();
  virtual void prepareMatrixClAlgReco();
  const TGeoHMatrix& getMatrixClAlg() const { return mMatClAlg; }
  const TGeoHMatrix& getMatrixClAlgReco() const { return mMatClAlgReco; }
  void setMatrixClAlg(const TGeoHMatrix& m) { mMatClAlg = m; }
  void setMatrixClAlgReco(const TGeoHMatrix& m) { mMatClAlgReco = m; }
  //
 protected:
  //
  virtual bool IsSortable() const { return true; }
  virtual int Compare(const TObject* a) const;
  //
  // --------- dummies -----------
  AlignableSensor(const AlignableSensor&);
  AlignableSensor& operator=(const AlignableSensor&);
  //
 protected:
  //
  int mSID;                  // sensor id in detector
  double mAddError[2];       // additional error increment for measurement
  AlignableDetector* mDet;   // pointer on detector
  TGeoHMatrix mMatClAlg;     // reference cluster alignment matrix in tracking frame
  TGeoHMatrix mMatClAlgReco; // reco-time cluster alignment matrix in tracking frame

  //
  ClassDef(AlignableSensor, 1)
};
} // namespace align
} // namespace o2

#endif