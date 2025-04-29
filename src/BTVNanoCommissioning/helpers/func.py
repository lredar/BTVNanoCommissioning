import awkward as ak
import numpy as np
from coffea import processor
import psutil, os, gzip, importlib, cloudpickle
import uproot
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory
from coffea.lookup_tools import extractor

import collections


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    return mem


def flatten(ar):  # flatten awkward into a 1d array to hist
    return ak.flatten(ar, axis=None)


def normalize(val, cut):
    if cut is None:
        ar = ak.to_numpy(ak.fill_none(val, np.nan))
        return ar
    else:
        ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
        return ar


def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


# return run & lumiblock in pairs
def dump_lumi(events, output):
    pairs = np.vstack((events.run.to_numpy(), events.luminosityBlock.to_numpy()))
    # remove replicas
    pairs = np.unique(np.transpose(pairs), axis=0)
    pairs = pairs[
        np.lexsort(([pairs[:, i] for i in range(pairs.shape[1] - 1, -1, -1)]))
    ]
    output["fname"] = processor.set_accumulator([events.metadata["filename"]])
    output["run"] = processor.column_accumulator(pairs[:, 0])
    output["lumi"] = processor.column_accumulator(pairs[:, 1])
    return output


def num(ar):
    return ak.num(ak.fill_none(ar[~ak.is_none(ar)], 0), axis=0)


## Based on https://github.com/CoffeaTeam/coffea/discussions/735
def _is_rootcompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = ak.type(a)
    if isinstance(t, ak._ext.ArrayType):
        if isinstance(t.type, ak._ext.PrimitiveType):
            return True
        if isinstance(t.type, ak._ext.ListType) and isinstance(
            t.type.type, ak._ext.PrimitiveType
        ):
            return True
    return False


jec_name_map = {
    "JetPt": "pt",
    "JetPhi": "phi",
    "JetMass": "mass",
    "JetEta": "eta",
    "JetA": "area",
    "ptGenJet": "pt_gen",
    "ptRaw": "pt_raw",
    "massRaw": "mass_raw",
    "Rho": "event_rho",
    "METpt": "pt",
    "METphi": "phi",
    "JetPhi": "phi",
    "UnClusteredEnergyDeltaX": "MetUnclustEnUpDeltaX",
    "UnClusteredEnergyDeltaY": "MetUnclustEnUpDeltaY",
}


def _load_jmefactory(year, campaign, jme_compiles):
    _jet_path = f"BTVNanoCommissioning.data.JME.{year}_{campaign}"
    with importlib.resources.path(_jet_path, jme_compiles) as filename:
        with gzip.open(filename) as fin:
            jme_facrory = cloudpickle.load(fin)

    return jme_facrory


def __jet_factory_factory__(files):
    ext = extractor()
    ext.add_weight_sets([f"* * {file}" for file in files])
    ext.finalize()
    jec_stack = JECStack(ext.make_evaluator())
    return CorrectedJetsFactory(jec_name_map, jec_stack)


def _jet_factories_(campaign, factory_map):
    factory_info = {
        j: __jet_factory_factory__(files=factory_map[j]) for j in factory_map.keys()
    }
    return factory_info


def _compile_jec_(year, campaign, factory_map, name):
    # jme stuff not pickleable in coffea
    import cloudpickle

    # add postfix to txt files
    update_factory = {}
    directory_path = f"src/BTVNanoCommissioning/data/JME/{year}_{campaign}/"
    files_in_directory = os.listdir(directory_path)
    for t in factory_map:
        if t == "name":
            continue
        update_factory[t] = []
        for f in factory_map[t]:
            if "Resolution" in f:
                if not os.path.exists(
                    f"src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.jr.txt"
                ) and os.path.exists(
                    f"src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.txt"
                ):
                    os.system(
                        f"mv src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.txt src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.jr.txt"
                    )
                update_factory[t].append(
                    f"src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.jr.txt"
                )
            elif "SF" in f:
                if not os.path.exists(
                    f"src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.jersf.txt"
                ) and os.path.exists(
                    f"src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.txt"
                ):
                    os.system(
                        f"mv src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.txt src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.jersf.txt"
                    )
                update_factory[t].append(
                    f"src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.jersf.txt"
                )
            elif "Uncertainty" in f:
                if not os.path.exists(
                    f"src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.junc.txt"
                ) and os.path.exists(
                    f"src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.txt"
                ):
                    os.system(
                        f"mv src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.txt src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.junc.txt"
                    )
                update_factory[t].append(
                    f"src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.junc.txt"
                )
            else:
                if not os.path.exists(
                    f"src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.jec.txt"
                ) and os.path.exists(
                    f"src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.txt"
                ):
                    os.system(
                        f"mv src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.txt src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.jec.txt"
                    )
                update_factory[t].append(
                    f"src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{f}.jec.txt"
                )

    with gzip.open(
        f"src/BTVNanoCommissioning/data/JME/{year}_{campaign}/{name}.pkl.gz", "wb"
    ) as fout:
        cloudpickle.dump(
            {
                "jet_factory": _jet_factories_(campaign, update_factory),
                "met_factory": CorrectedMETFactory(jec_name_map),
            },
            fout,
        )


def PFCand_link(events, event_level, jetindx):
    if str(ak.type(jetindx)).count("*") > 1:
        jetindx = jetindx[event_level]
        spfcands = collections.defaultdict(dict)
        for i in range(len(jetindx[0])):
            spfcands[i] = events[event_level].PFCands[
                events[event_level]
                .JetPFCands[events[event_level].JetPFCands.jetIdx == jetindx[:, i]]
                .pFCandsIdx
            ]

    else:
        spfcands = events[event_level].PFCands[
            events[event_level]
            .JetPFCands[events[event_level].JetPFCands.jetIdx == jetindx[event_level]]
            .pFCandsIdx
        ]
    return spfcands


def uproot_writeable(events, include=["events", "run", "luminosityBlock"]):
    ev = {}
    include = np.array(include)
    no_filter = False
    if len(include) == 1 and include[0] == "*":
        no_filter = False
    for bname in events.fields:
        if not events[bname].fields:
            if not no_filter and bname not in include:
                continue
            ev[bname] = ak.fill_none(
                ak.packed(ak.without_parameters(events[bname])), -99
            )
        else:
            b_nest = {}
            no_filter_nest = False
            if all(np.char.startswith(include, bname) == False):
                continue
            include_nest = [
                i[i.find(bname) + len(bname) + 1 :]
                for i in include
                if i.startswith(bname)
            ]

            if len(include_nest) == 1 and include_nest[0] == "*":
                no_filter_nest = True

            if not no_filter_nest:
                mask_wildcard = np.char.find(include_nest, "*") != -1
                include_nest = np.char.replace(include_nest, "*", "")
            for n in events[bname].fields:
                ## make selections to the filter case, keep cross-ref ("Idx")
                if (
                    not no_filter_nest
                    and all(np.char.find(n, include_nest) == -1)
                    and "Idx" not in n
                    and "Flavor" not in n
                ):
                    continue
                evnums = ak.num(events[bname][n], axis=0)
                if not isinstance(evnums, int):
                    continue
                if not _is_rootcompat(events[bname][n]) and evnums != len(
                    flatten(events[bname][n])
                ):
                    continue
                # skip IdxG
                if "IdxG" in n:
                    continue
                b_nest[n] = ak.fill_none(
                    ak.packed(ak.without_parameters(events[bname][n])), -99
                )
            if bool(b_nest):
                ev[bname] = ak.zip(b_nest)
    return ev

def add_discriminators(jets, tagger):
    """
    Computes additional discriminators for a given jet collection.

    Parameters:
    - jets: awkward.Array representing the jet collection (e.g. events.Jet, pruned_ev.SelJet)
    - tagger: str, "DeepFlav", "PNet", "RobustParTAK4", "UParTAK4" or "UParTAK4_v2"

    Returns:
    - updated jets: ["BvC", "BvCt", "HFvsLFt", "HFvsLF", "probc", "probusdg", ("probbc", "probs", "probbbblepb")]
    """

    if tagger == "DeepFlav":
        B = jets.btagDeepFlavB
        probc = jets.btagDeepFlavC
        CvB = jets.btagDeepFlavCvB
        CvL = jets.btagDeepFlavCvL
    elif tagger == "PNet":
        CvB = jets.btagPNetCvB
        CvNotB = jets.btagPNetCvNotB
        CvL = jets.btagPNetCvL
    elif tagger == "RobustParTAK4":
        CvB = jets.btagRobustParTAK4CvB
        CvNotB = jets.btagRobustParTAK4CvNotB
        CvL = jets.btagRobustParTAK4CvL
    elif "UParTAK4" in tagger:
        CvB = jets.btagUParTAK4CvB
        CvNotB = jets.btagUParTAK4CvNotB
        CvL = jets.btagUParTAK4CvL
        if tagger == "UParTAK4_v2":
            probudg = jets.btagUParTAK4UDG
            SvUDG = jets.btagUParTAK4SvUDG
    else:
        raise ValueError(f"Unknown tagger: {tagger}")

    if tagger == "DeepFlav":
        probudsg = ak.where((CvL > 0) & (probc > 0), (1.0 - CvL) * probc / CvL, -1.0)
        BvC = ak.where(CvB > 0, 1.0 - CvB, -1.0)
        HFvLF = ak.where(
            (B > 0) & (probc > 0) & (probudsg > 0),
            (B + probc) / (B + probc + probudsg),
            -1.0,
        )
    elif tagger == "UParTAK4_v2":
        probs = ak.Array(np.where((SvUDG > 0.0) & (probudg > 0.0), SvUDG * probudg / (1.0 - SvUDG), -1.0))
        probc = ak.Array(np.where((CvL > 0.0) & (probs > 0.0) & (probudg > 0.0),CvL * (probs + probudg) / (1.0 - CvL),-1.0,))
        probbbblepb = ak.Array(np.where((CvB > 0.0) & (probc > 0.0), (1.0 - CvB) * probc / CvB, -1.0))
        BvC = ak.Array(np.where(CvB > 0.0, 1.0 - CvB, -1.0))
        HFvLF = ak.Array(
            np.where(
                (probbbblepb > 0.0) & (probc > 0.0) & (probs > 0.0) & (probudg > 0.0),
                (probbbblepb + probc) / (probbbblepb + probc + probs + probudg),
                -1.0,
            )
        )
    else:
        probc = ak.where((CvB > 0) & (CvNotB > 0), CvB * CvNotB / (CvB - CvNotB * CvB + CvNotB), -1.0)
        probbc = ak.where((CvB > 0) & (CvNotB > 0), CvNotB / (CvB - CvNotB * CvB + CvNotB), -1.0)
        probudsg = ak.where((CvL > 0) & (probc > 0), (1.0 - CvL) * probc / CvL, -1.0)
        BvC = ak.where(CvB > 0, 1.0 - CvB, -1.0)
        HFvLF = ak.where(
            (probbc > 0) & (probudsg > 0),
            probbc / (probbc + probudsg),
            -1.0,
        )
        jets = ak.with_field(jets, probbc, f"btag{tagger}probbc")

    if tagger == "UParTAK4_v2":
        tagger = tagger.replace("_v2", "")
        jets = ak.with_field(jets, probs, f"btag{tagger}probs")
        jets = ak.with_field(jets, probbbblepb, f"btag{tagger}probbbblepb")

    # Add common fields
    jets = ak.with_field(jets, probc, f"btag{tagger}probc")
    jets = ak.with_field(jets, probudsg, f"btag{tagger}probudsg")
    jets = ak.with_field(jets, BvC, f"btag{tagger}BvC")
    jets = ak.with_field(jets, HFvLF, f"btag{tagger}HFvLF")
    jets = ak.with_field(jets, ak.where(BvC > 0, 1.0 - (1.0 - BvC) ** 0.5, -1.0), f"btag{tagger}BvCt")
    jets = ak.with_field(jets, ak.where(HFvLF > 0, 1.0 - (1.0 - HFvLF) ** 0.5, -1.0), f"btag{tagger}HFvLFt")

    return jets