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
  pythia.readString("Beams:LHEF = /mnt/48D4698ED4697ED6/Python/MSc_Project_Upgrade/generation/LHE_files/ee_tau_250GeV_175k_train.lhe");

  // ---------------------------
  // Physics settings
  // ---------------------------
  pythia.readString("HadronLevel:all = on");

  // Force ONLY hadronic tau decays (correct way)
  pythia.readString("TauDecays:mode = 2");

  // Optional: reproducibility
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = 12345");

  // Optional (cleaner for e+e-)
  pythia.readString("PartonLevel:ISR = off");

  // ---------------------------
  // Initialize
  // ---------------------------
  if (!pythia.init())
  {
    std::cerr << "Error: Pythia initialization failed." << std::endl;
    return 1;
  }

  std::cout << "Processing events..." << std::endl;

  // ---------------------------
  // HepMC3 writer
  // ---------------------------
  HepMC3::WriterAscii writer(
      "/mnt/48D4698ED4697ED6/Python/MSc_Project_Upgrade/generation/hepmc_files/ee_tau_250GeV_175k_train.hepmc");

  HepMC3::Pythia8ToHepMC3 toHepMC;

  int nProcessed = 0;

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
      continue;
    }

    nProcessed++;

    HepMC3::GenEvent evt;
    toHepMC.fill_next_event(pythia, &evt);
    writer.write_event(evt);

    if (nProcessed % 10000 == 0)
      std::cout << "Processed " << nProcessed << " events..." << std::endl;
  }

  writer.close();
  pythia.stat();

  std::cout << "\n=====================================" << std::endl;
  std::cout << "Total events processed : " << nProcessed << std::endl;
  std::cout << "=====================================\n"
            << std::endl;

  return 0;
}