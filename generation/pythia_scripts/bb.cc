#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"
#include "HepMC3/WriterAscii.h"
#include <iostream>

using namespace Pythia8;

int main()
{
  Pythia pythia;

  // ---------------------------
  // LHE input configuration
  // ---------------------------
  pythia.readString("Beams:frameType = 4");
  pythia.readString("Beams:LHEF = /mnt/48D4698ED4697ED6/Python/MSc_Project_Upgrade/generation/LHE_files/ee_bb_250GeV_60k_train.lhe");

  // ---------------------------
  // Hadronization
  // ---------------------------
  pythia.readString("HadronLevel:all = on");

  // ---------------------------
  // Initialize
  // ---------------------------
  if (!pythia.init())
  {
    std::cerr << "Error: Pythia initialization failed." << std::endl;
    return 1;
  }

  std::cout << "Reading LHE file with Pythia..." << std::endl;

  // ---------------------------
  // HepMC3 writer
  // ---------------------------
  HepMC3::WriterAscii writer(
      "/mnt/48D4698ED4697ED6/Python/MSc_Project_Upgrade/generation/hepmc_files/ee_bb_250GeV_60k_train.hepmc");

  HepMC3::Pythia8ToHepMC3 toHepMC;

  int nProcessed = 0;
  int nAccepted = 0;
  int nSkipped = 0;

  // ---------------------------
  // Event loop
  // ---------------------------
  while (true)
  {
    if (!pythia.next())
    {
      if (pythia.info.atEndOfFile())
      {
        std::cout << "Reached end of LHE file." << std::endl;
        break;
      }
      nSkipped++;
      continue;
    }

    nProcessed++;

    HepMC3::GenEvent evt;
    toHepMC.fill_next_event(pythia, &evt);
    writer.write_event(evt);
    nAccepted++;

    if (nAccepted % 10000 == 0)
      std::cout << "Processed " << nAccepted << " events..." << std::endl;
  }

  writer.close();
  pythia.stat();

  std::cout << "\n=====================================" << std::endl;
  std::cout << "Total events processed : " << nProcessed << std::endl;
  std::cout << "Total events accepted  : " << nAccepted << std::endl;
  std::cout << "Total events skipped   : " << nSkipped << std::endl;
  std::cout << "=====================================\n"
            << std::endl;

  return 0;
}