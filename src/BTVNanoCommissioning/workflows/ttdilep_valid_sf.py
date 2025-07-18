import collections, awkward as ak, numpy as np
import os
import uproot
from coffea import processor
from coffea.analysis_tools import Weights

# functions to load SFs, corrections
from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    weight_manager,
    common_shifts,
)

# user helper function
from BTVNanoCommissioning.helpers.func import flatten, update, dump_lumi, PFCand_link, add_discriminators
from BTVNanoCommissioning.helpers.update_branch import missing_branch

## load histograms & selctions for this workflow
from BTVNanoCommissioning.utils.histogrammer import histogrammer, histo_writter
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    jet_id,
    mu_idiso,
    ele_cuttightid,
    btag_wp,
)


class NanoProcessor(processor.ProcessorABC):
        # NanoProcessor is a custom processor class that inherits from class NanoProcessor(processor.ProcessorABC (abstract base class)):
    # Define histograms

    # Here is the initializer
    # like if call the class in a way sth = NanoProcessor(processor.ProcessorABC)
    # that sth will be the self inside the class
    def __init__(
        self,
        year="2022", # these are default settings if one does not pass any when call the class
        campaign="Summer22Run3",
        name="",
        isSyst=False,
        isArray=False,
        noHist=False,
        chunksize=75000,
        selectionModifier="tt_dilep",
    ):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.lumiMask = load_lumi(self._campaign)
        self.chunksize = chunksize
        self.selMod = selectionModifier
        ## Load corrections
        self.SF_map = load_SF(self._year, self._campaign)

    @property # with @property, one can call a method like a regular attribute
    def accumulator(self):
        return self._accumulator

    ## Apply corrections on momentum/mass on MET, Jet, Muon
    def process(self, events):
        events = missing_branch(events) # adds missing fields or fixes old formats
        shifts = common_shifts(self, events)

        return processor.accumulate(
            self.process_shift(update(events, collections), name)
            for collections, name in shifts
        )

    def process_shift(self, events, shift_name):
        # self: refers to the current object (an instance of the NanoProcessor class)
        # it allows the method access other parts of the class, like self._year

        # events: Awkward Array representing one batch of events (particle collisions)
        # it comes from the ROOT file, processed through uproot

        # shift_name tells if the event is shifted or not



        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        # built-in Python function: hasattr(object, "attribute_name")
        # It returns: 
        # True if the object has that attribute
        # False if it doesn’t
        # so hasattr(events, "genWeight") gives True if the batch of events have a field named genWeight
        # and 'not' reverse it, so if we get a real data, we get a True



        ## Create histograms

        # initially, at the very begining, we have noHist=False, which means, by default we make histogram for sure
        # i.e. if self.noHist: (i.e. if self.noHist == True)# skip histograms
        # if not self.noHist: # make histograms

        if self.selMod == "ttdilep_sf_2Dcalib":
            output = {} if self.noHist else histogrammer(events, "ttdilep_sf_2Dcalib")
        else:
            output = {} if self.noHist else histogrammer(events, "ttdilep_sf")




        if shift_name is None:
            if isRealData:
                output["sumw"] = len(events)
                # so if this is real data, each event has weight of 1
                # so sum of the w gives the real number of events directly
                # I assusme isRealData is something stored inside, to tell if this is data or MC
            else:
                output["sumw"] = ak.sum(events.genWeight)
                # so if this is not real data, i.e. it's a MC
                # then sum of weight is the number of the events


        ####################
        #    Selections    #
        ####################
        ## Lumimask
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)
        # only dump for nominal case
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)

        ## HLT
        if 'mumu' in self.selMod:
            triggers = ["Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8"]
            req_trig = HLT_helper(events, triggers)
        elif 'ee' in self.selMod:
            triggers = ["Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"]
            req_trig = HLT_helper(events, triggers)
        else:
            triggers = [
                "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            ]
            req_trig = HLT_helper(events, triggers)

        ##### Muon cuts #####
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        events.Muon = events.Muon[
            (events.Muon.pt > 30) & mu_idiso(events, self._campaign)
        ]
        ## It's such a crime to not simply create mask via ak.sum() here...
        events.Muon = ak.pad_none(events.Muon, 1, axis=1)
        req_muon = ak.count(events.Muon.pt, axis=1) == 1
        req_muon_2 = ak.count(events.Muon.pt, axis=1) == 2
        req_muon_0 = ak.count(events.Muon.pt, axis=1) == 0

        ##### Electron cuts #####
        # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        events.Electron = events.Electron[
            (events.Electron.pt > 30) & ele_cuttightid(events, self._campaign)
        ]
        events.Electron = ak.pad_none(events.Electron, 1, axis=1)
        req_ele = ak.count(events.Electron.pt, axis=1) == 1
        req_ele_2 = ak.count(events.Electron.pt, axis=1) == 2
        req_ele_0 = ak.count(events.Electron.pt, axis=1) == 0


        ##### Jet cuts #####
        jetsel = ak.fill_none(
            jet_id(events, self._campaign)
            & (
                ak.all(
                    events.Jet.metric_table(events.Muon) > 0.4,
                    axis=2,
                    mask_identity=True,
                )
            )
            & (
                ak.all(
                    events.Jet.metric_table(events.Electron) > 0.4,
                    axis=2,
                    mask_identity=True,
                )
            ),
            False,
        )
        event_jet = events.Jet[jetsel]
        req_jets = ak.num(event_jet.pt) >= 2

        ##### CHarge cuts #####
        if 'mumu' in self.selMod:
            ## charge for mumu
            req_opposite_charge_mumu = (
                events.Muon[:, 0:1].charge * events.Muon[:, 1:2].charge
                # get a slice stops before index 1 if exists and pick out the 0 index (first) muon (can do [0:2] as well)
                # get the other slice stops before index 2 if exists and pick our the 1 index (second) muon
            ) == -1
            req_opposite_charge_mumu = ak.fill_none(req_opposite_charge_mumu, False)
            req_opposite_charge_mumu = ak.flatten(req_opposite_charge_mumu)

        elif 'ee' in self.selMod:
            ## charge for ee
            req_opposite_charge_ee = (
                events.Electron[:, 0:1].charge * events.Electron[:, 1:2].charge
            ) == -1
            req_opposite_charge_ee = ak.fill_none(req_opposite_charge_ee, False)
            req_opposite_charge_ee = ak.flatten(req_opposite_charge_ee)

        else:
        ## charge for emu
            req_opposite_charge = (
                events.Electron[:, 0:1].charge * events.Muon[:, 0:1].charge
                # [0:1], a slice (list) starting at index 0 and stopping before index 1
                # basically, in every event, take the first electron if it exists
                # because it returns a list, so even if it's empty list, it's safe
            ) == -1
            req_opposite_charge = ak.fill_none(req_opposite_charge, False)
            req_opposite_charge = ak.flatten(req_opposite_charge)


        ## store jet index for PFCands, create mask on the jet index
        jetindx = ak.mask(ak.local_index(events.Jet.pt), jetsel)
        jetindx = ak.pad_none(jetindx, 2)
        jetindx = jetindx[:, :2]

        # Mask that includes everything
        if 'mumu' in self.selMod:
            event_level = (
                req_trig & req_lumi & req_muon_2 & req_ele_0 & req_jets & req_opposite_charge_mumu
            )
        elif 'ee' in self.selMod:
            event_level = (
                req_trig & req_lumi & req_ele_2 & req_muon_0 & req_jets & req_opposite_charge_ee
            )
        else:
            event_level = (
                req_trig & req_lumi & req_muon & req_ele & req_jets & req_opposite_charge
            )

        event_level = ak.fill_none(event_level, False)
        if len(events[event_level]) == 0:
            if self.isArray:
                array_writer(
                    self,
                    events[event_level],
                    events,
                    None,
                    ["nominal"],
                    dataset,
                    isRealData,
                    empty=True,
                )
            return {dataset: output}

        ####################
        # Selected objects #
        ####################
        # Keep the structure of events and pruned the object size
        pruned_ev = events[event_level]
        pruned_ev["SelJet"] = event_jet[event_level][:, :2]

        ##### This part also needed to be edited for the mumu ee cases #####
        
        if 'mumu' in self.selMod:
            pruned_ev["SelMuon0"] = events.Muon[event_level][:, 0]
            # Here should be okay to just using index 1 without the slice, because, event_level picks out strictly two muons events
            pruned_ev["SelMuon1"] = events.Muon[event_level][:, 1]
            ### What i should name here and do here? like, there are two muons now, no electron
            ### and i suppose this "SelElectron" called somewhere else to make histograms
            ### so where else should i edit it?


            ########## Big problem to define like this, for in the histogrammer, there only is a SelMuon, no SelMuon1 ##########
            ########## BUt like in ctag_DY, there is a ElectronPlus, which is not inside the histogrammer????? ##########



            
            pruned_ev["dr_mu0jet0"] = pruned_ev.SelMuon0.delta_r(pruned_ev.Jet[:, 0])
            pruned_ev["dr_mu0jet1"] = pruned_ev.SelMuon0.delta_r(pruned_ev.Jet[:, 1]) 
            pruned_ev["dr_mu1jet0"] = pruned_ev.SelMuon1.delta_r(pruned_ev.Jet[:, 0])
            pruned_ev["dr_mu1jet1"] = pruned_ev.SelMuon1.delta_r(pruned_ev.Jet[:, 1]) 
            ### What i should name here and do here? like, there are two muons now
        
        elif 'ee' in self.selMod:
            pruned_ev["SelElectron0"] = events.Electron[event_level][:, 0]
            ### What i should name here and do here? like, there are two electrons now, no muon
            pruned_ev["SelElectron1"] = events.Electron[event_level][:, 1]
            pruned_ev["dr_el0jet0"] = pruned_ev.SelElectron0.delta_r(pruned_ev.Jet[:, 0])
            pruned_ev["dr_el0jet1"] = pruned_ev.SelElectron0.delta_r(pruned_ev.Jet[:, 1]) 
            pruned_ev["dr_el1jet0"] = pruned_ev.SelElectron1.delta_r(pruned_ev.Jet[:, 0])
            pruned_ev["dr_el1jet1"] = pruned_ev.SelElectron1.delta_r(pruned_ev.Jet[:, 1]) 
            ### What i should name here and do here? like, there are no muon now

        else:
            pruned_ev["SelMuon"] = events.Muon[event_level][:, 0]
            pruned_ev["SelElectron"] = events.Electron[event_level][:, 0]
            pruned_ev["dr_mujet0"] = pruned_ev.SelMuon.delta_r(pruned_ev.Jet[:, 0])
            pruned_ev["dr_mujet1"] = pruned_ev.SelMuon.delta_r(pruned_ev.Jet[:, 1]) 
        ##### Why for muon case, only calculated muon and jet delta_r? what about electron????? ##### 
            ##### Add electron ones here #####
            # pruned_ev["dr_eljet0"] = pruned_ev.SelElectron.delta_r(pruned_ev.Jet[:, 0])
            # pruned_ev["dr_eljet1"] = pruned_ev.SelElectron.delta_r(pruned_ev.Jet[:, 1]) 

        ##### Above part #####

        pruned_ev["njet"] = ak.count(event_jet[event_level].pt, axis=1)
        # Find the PFCands associate with selected jets. Search from jetindex->JetPFCands->PFCand
        if "PFCands" in events.fields:
            pruned_ev.PFCands = PFCand_link(events, event_level, jetindx)


        ##### What about mumu and ee? Below????? #####

        if "2Dcalib" in self.selMod:
            taggers =  ["DeepFlav", "PNet", "UParTAK4"] #RobustParTAK4
            for tagger in taggers:
                pruned_ev["SelJet"] = add_discriminators(pruned_ev["SelJet"], tagger)

        ##### What about mumu and ee? Above????? #####

        ####################
        #     Output       #
        ####################
        # Configure SFs
        weights = weight_manager(pruned_ev, self.SF_map, self.isSyst)
        # Configure systematics
        if shift_name is None:
            systematics = ["nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]

        # Configure histograms
        if not self.noHist:
            output = histo_writter(
                pruned_ev, output, weights, systematics, self.isSyst, self.SF_map
            )
        # Output arrays
        if self.isArray:
            array_writer(
                self, pruned_ev, events, weights, systematics, dataset, isRealData
            )
        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
