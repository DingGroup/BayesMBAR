# Compute relative protein-ligand binding free energy


```python
import numpy as np
from itertools import product
import pandas as pd
import openmm as mm
import openmm.app as app
from openatom.functions import compute_mcs_VF2, make_alchemical_system, make_psf_from_topology
import os
import pickle
import openmm.unit as unit
import mdtraj
from openmm import XmlSerializer
import xml.etree.ElementTree as ET
import time
```


```python
pairs = [
    ("ejm_31", "ejm_42"),
    ("ejm_42", "ejm_50"),
    ("ejm_50", "ejm_31"),
    ("ejm_31", "ejm_55"),
    ("ejm_55", "ejm_50"),
    ("ejm_55", "ejm_54"),
    ("ejm_54", "ejm_43"),
    ("ejm_43", "ejm_42"),
]

```


```python
for liga, ligb in pairs:
    # load prmtop and inpcrd files
    prmtopa = app.AmberPrmtopFile(
        f"./structure/water_phase/{liga}.prmtop"
    )
    prmtopb = app.AmberPrmtopFile(
        f"./structure/water_phase/{ligb}.prmtop"
    )
    
    for atom in prmtopa.topology.atoms():
        if atom.name == 'C14':
            atom.label = liga
    for atom in prmtopb.topology.atoms():
        if atom.name == 'C14':
            atom.label = ligb

    mcs = compute_mcs_VF2(prmtopa.topology, prmtopb.topology, timeout=5)


    os.makedirs(f"./output/{liga}_{ligb}", exist_ok=True)
    with open(f"./output/{liga}_{ligb}/mcs.pkl", "wb") as f:
        pickle.dump(mcs, f)

    print(f"{liga} and {ligb} MCS: {len(mcs)}")
    print(mcs)
```

```text
ejm_31 and ejm_42 MCS: 28
{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 20: 20, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29}
ejm_42 and ejm_50 MCS: 28
{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 20: 20, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29}
ejm_50 and ejm_31 MCS: 28
{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 20: 20, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29}
ejm_31 and ejm_55 MCS: 28
{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 20: 19, 22: 20, 23: 21, 24: 22, 25: 23, 26: 24, 27: 25, 28: 26, 29: 27}
ejm_55 and ejm_50 MCS: 28
{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 20, 20: 22, 21: 23, 22: 24, 23: 25, 24: 26, 25: 27, 26: 28, 27: 29}
ejm_55 and ejm_54 MCS: 28
{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27}
ejm_54 and ejm_43 MCS: 28
{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 20, 20: 22, 21: 23, 22: 24, 23: 25, 24: 26, 25: 27, 26: 28, 27: 29}
ejm_43 and ejm_42 MCS: 28
{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 20: 20, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29}
```



```python
lambdas_list = [
    [(1.0, 1.0), (0.0, 0.0)],
    [(0.5, 1.0), (0.0, 0.0)],
    [(0.0, 1.0), (0.0, 0.0)],
    [(0.0, 0.9), (0.0, 0.1)],
    [(0.0, 0.7), (0.0, 0.3)],
    [(0.0, 0.5), (0.0, 0.5)],
    [(0.0, 0.3), (0.0, 0.7)],
    [(0.0, 0.1), (0.0, 0.9)],
    [(0.0, 0.0), (0.0, 1.0)],
    [(0.0, 0.0), (0.5, 1.0)],
    [(0.0, 0.0), (1.0, 1.0)],
]

for phase in ["water", "protein"]:
    if phase == "protein":
        envi_prmtop = app.AmberPrmtopFile(
            f"./structure/{phase}_phase/env.prmtop"
        )
        envi_coor = app.PDBFile(
            f"./structure/{phase}_phase/env.pdb"
        ).getPositions()
    else:
        envi_prmtop = app.AmberPrmtopFile(
            "./structure/water_phase/env.prmtop"
        )
        envi_coor = app.PDBFile(
            "./structure/water_phase/env.pdb"
        ).getPositions()

    envi_system = envi_prmtop.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        switchDistance=0.9 * unit.nanometer,
    )
    envi_top = envi_prmtop.topology
    envi_coor = np.array(envi_coor.value_in_unit(unit.nanometer))

    for liga_name, ligb_name in pairs:
        print(phase, liga_name, ligb_name)

        # load prmtop and xyz files
        if phase == "protein":
            liga_prmtop = app.AmberPrmtopFile(
                f"./structure/{phase}_phase/{liga_name}.prmtop",
                envi_top.getPeriodicBoxVectors(),
            )
        else:
            liga_prmtop = app.AmberPrmtopFile(
                f"./structure/water_phase/{liga_name}.prmtop",
                envi_top.getPeriodicBoxVectors(),
            )

        liga_top = liga_prmtop.topology
        liga_system = liga_prmtop.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
            switchDistance=0.9 * unit.nanometer,
        )

        if phase == "protein":
            ligb_prmtop = app.AmberPrmtopFile(
                f"./structure/{phase}_phase/{ligb_name}.prmtop",
                envi_top.getPeriodicBoxVectors(),
            )
        else:
            ligb_prmtop = app.AmberPrmtopFile(
                f"./structure/water_phase/{ligb_name}.prmtop",
                envi_top.getPeriodicBoxVectors(),
            )

        ligb_top = ligb_prmtop.topology
        ligb_system = ligb_prmtop.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
            switchDistance=0.9 * unit.nanometer,
        )

        if phase == "protein":
            liga_coor = mdtraj.load_xyz(
                f"./structure/{phase}_phase/{liga_name}_aligned.xyz",
                top=mdtraj.Topology.from_openmm(liga_prmtop.topology),
            ).xyz[0]
            ligb_coor = mdtraj.load_xyz(
                f"./structure/{phase}_phase/{ligb_name}_aligned.xyz",
                top=mdtraj.Topology.from_openmm(ligb_prmtop.topology),
            ).xyz[0]
        else:
            liga_coor = mdtraj.load_xyz(
                f"./structure/water_phase/{liga_name}_aligned.xyz",
                top=mdtraj.Topology.from_openmm(liga_prmtop.topology),
            ).xyz[0]
            ligb_coor = mdtraj.load_xyz(
                f"./structure/water_phase/{ligb_name}_aligned.xyz",
                top=mdtraj.Topology.from_openmm(ligb_prmtop.topology),
            ).xyz[0]

        ligs_coor = [liga_coor, ligb_coor]

        envi_xml = XmlSerializer.serializeSystem(envi_system)
        liga_xml = XmlSerializer.serializeSystem(liga_system)
        ligb_xml = XmlSerializer.serializeSystem(ligb_system)

        envi_et = ET.fromstring(envi_xml)
        liga_et = ET.fromstring(liga_xml)
        ligb_et = ET.fromstring(ligb_xml)
        ligs_et = [liga_et, ligb_et]

        with open(f"./output/{liga_name}_{ligb_name}/mcs.pkl", "rb") as f:
            mcs = pickle.load(f)

        liga_common_atoms = list(mcs.keys())
        ligb_common_atoms = [mcs[i] for i in liga_common_atoms]
        ligs_common_atoms = [liga_common_atoms, ligb_common_atoms]

        os.makedirs(f"./output/{liga_name}_{ligb_name}/{phase}_phase", exist_ok=True)
        with open(
            f"./output/{liga_name}_{ligb_name}/{phase}_phase/lambdas.pkl", "wb"
        ) as f:
            pickle.dump(lambdas_list, f)

        for lambdas in lambdas_list:
            print(lambdas)
            if phase == "vacuum":
                system_xml, top, coor = make_alchemical_system(
                    ligs_et,
                    [liga_top, ligb_top],
                    ligs_common_atoms,
                    ligs_coor,
                    lambdas,
                    None,
                    None,
                    None,
                )
            else:
                system_xml, top, coor = make_alchemical_system(
                    ligs_et,
                    [liga_top, ligb_top],
                    ligs_common_atoms,
                    ligs_coor,
                    lambdas,
                    envi_et,
                    envi_top,
                    envi_coor,
                )

            tree = ET.ElementTree(system_xml)
            ET.indent(tree.getroot())
            (elec0, vdw0), (elec1, vdw1) = lambdas
            os.makedirs(
                f"./output/{liga_name}_{ligb_name}/{phase}_phase/sys", exist_ok=True
            )
            tree.write(
                f"./output/{liga_name}_{ligb_name}/{phase}_phase/sys/{elec0:.2f}_{vdw0:.2f}_{elec1:.2f}_{vdw1:.2f}.xml",
                xml_declaration=True,
                method="xml",
                encoding="utf-8",
            )

            with open(
                f"./output/{liga_name}_{ligb_name}/{phase}_phase/system.xyz", "w"
            ) as f:
                f.write(f"{len(coor)}\n")
                f.write(f"{liga_name}_{ligb_name}\n")
                for atom, xyz in zip(top.atoms(), coor):
                    xyz = xyz * 10
                    f.write(
                        f"{atom.element.symbol} {xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}\n"
                    )

            mm.app.PDBFile.writeFile(
                top,
                coor * 10,
                f"./output/{liga_name}_{ligb_name}/{phase}_phase/system.pdb",
                keepIds=True,
            )

            with open(
                f"./output/{liga_name}_{ligb_name}/{phase}_phase/topology.pkl", "wb"
            ) as file_handle:
                pickle.dump(top, file_handle)

            make_psf_from_topology(
                top, f"./output/{liga_name}_{ligb_name}/{phase}_phase/topology.psf"
            )
```

```text
water ejm_31 ejm_42
[(1.0, 1.0), (0.0, 0.0)]


<class 'networkx.utils.decorators.argmap'> compilation 17:3: FutureWarning: 

single_target_shortest_path_length will return a dict instead of
an iterator in version 3.5


[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
water ejm_42 ejm_50
[(1.0, 1.0), (0.0, 0.0)]
[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
water ejm_50 ejm_31
[(1.0, 1.0), (0.0, 0.0)]
[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
water ejm_31 ejm_55
[(1.0, 1.0), (0.0, 0.0)]
[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
water ejm_55 ejm_50
[(1.0, 1.0), (0.0, 0.0)]
[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
water ejm_55 ejm_54
[(1.0, 1.0), (0.0, 0.0)]
[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
water ejm_54 ejm_43
[(1.0, 1.0), (0.0, 0.0)]
[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
water ejm_43 ejm_42
[(1.0, 1.0), (0.0, 0.0)]
[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
protein ejm_31 ejm_42
[(1.0, 1.0), (0.0, 0.0)]
[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
protein ejm_42 ejm_50
[(1.0, 1.0), (0.0, 0.0)]
[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
protein ejm_50 ejm_31
[(1.0, 1.0), (0.0, 0.0)]
[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
protein ejm_31 ejm_55
[(1.0, 1.0), (0.0, 0.0)]
[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
protein ejm_55 ejm_50
[(1.0, 1.0), (0.0, 0.0)]
[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
protein ejm_55 ejm_54
[(1.0, 1.0), (0.0, 0.0)]
[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
protein ejm_54 ejm_43
[(1.0, 1.0), (0.0, 0.0)]
[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
protein ejm_43 ejm_42
[(1.0, 1.0), (0.0, 0.0)]
[(0.5, 1.0), (0.0, 0.0)]
[(0.0, 1.0), (0.0, 0.0)]
[(0.0, 0.9), (0.0, 0.1)]
[(0.0, 0.7), (0.0, 0.3)]
[(0.0, 0.5), (0.0, 0.5)]
[(0.0, 0.3), (0.0, 0.7)]
[(0.0, 0.1), (0.0, 0.9)]
[(0.0, 0.0), (0.0, 1.0)]
[(0.0, 0.0), (0.5, 1.0)]
[(0.0, 0.0), (1.0, 1.0)]
```



```python
def run_simulation(liga_name, ligb_name, phase, idx_lambda):
    print(f"Running simulation for {liga_name} and {ligb_name} in {phase}", flush=True)
    lambdas = lambdas_list[idx_lambda]
    (elec0, vdw0), (elec1, vdw1) = lambdas
    lambdas_str = f"{elec0:.2f}_{vdw0:.2f}_{elec1:.2f}_{vdw1:.2f}"
    print(f"Running simulation for lambdas {lambdas_str}", flush=True)

    ## deserialize the system
    with open(
        f"./output/{liga_name}_{ligb_name}/{phase}_phase/sys/{elec0:.2f}_{vdw0:.2f}_{elec1:.2f}_{vdw1:.2f}.xml",
        "r",
    ) as f:
        system = mm.XmlSerializer.deserialize(f.read())


    ## add barostat
    system.addForce(mm.MonteCarloBarostat(1 * unit.atmospheres, 298.15 * unit.kelvin))

    with open(f"./output/{liga_name}_{ligb_name}/{phase}_phase/topology.pkl", "rb") as f:
        topology = pickle.load(f)

    pdb = app.PDBFile(f"./output/{liga_name}_{ligb_name}/{phase}_phase/system.pdb")

    integrator = mm.LangevinMiddleIntegrator(
        298.15 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    platform = mm.Platform.getPlatformByName("CPU")
    simulation = app.Simulation(topology, system, integrator, platform)

    simulation.context.setPositions(pdb.positions)
    print("Minimizing energy", flush=True)
    simulation.minimizeEnergy()

    print("Equilibrating", flush=True)
    #simulation.step(100_000)
    simulation.step(1_000)

    os.makedirs(f"./output/{liga_name}_{ligb_name}/{phase}_phase/traj", exist_ok=True)
    simulation.reporters.append(
        app.DCDReporter(
            f"./output/{liga_name}_{ligb_name}/{phase}_phase/traj/{lambdas_str}.dcd",
            1_000,
            append=False,
        )
    )

    print("Running simulation", flush=True)
    start_time = time.time()
    #simulation.step(12_500_000)
    simulation.step(5_000)

    simulation.saveCheckpoint(
        f"./output/{liga_name}_{ligb_name}/{phase}_phase/traj/{lambdas_str}.chk"
    )

    print(f"Simulation finished in {time.time() - start_time:.2f} seconds", flush=True)
```


```python
for (liga_name, ligb_name), phase, idx_lambda in product(
    pairs, ["water", "protein"], range(len(lambdas_list))
):
    run_simulation(liga_name, ligb_name, phase, idx_lambda)
```

```text
Running simulation for ejm_31 and ejm_42 in water
Running simulation for lambdas 1.00_1.00_0.00_0.00
here
Minimizing energy
Equilibrating
Running simulation
Simulation finished in 23.91 seconds
Running simulation for ejm_31 and ejm_42 in water
Running simulation for lambdas 0.50_1.00_0.00_0.00
here
Minimizing energy
Equilibrating
Running simulation
Simulation finished in 26.05 seconds
Running simulation for ejm_31 and ejm_42 in water
Running simulation for lambdas 0.00_1.00_0.00_0.00
here
Minimizing energy
Equilibrating
Running simulation
Simulation finished in 24.64 seconds
Running simulation for ejm_31 and ejm_42 in water
Running simulation for lambdas 0.00_0.90_0.00_0.10
here
Minimizing energy
Equilibrating
Running simulation
Simulation finished in 24.69 seconds
Running simulation for ejm_31 and ejm_42 in water
Running simulation for lambdas 0.00_0.70_0.00_0.30
here
Minimizing energy
Equilibrating
Running simulation
Simulation finished in 25.04 seconds
Running simulation for ejm_31 and ejm_42 in water
Running simulation for lambdas 0.00_0.50_0.00_0.50
here
Minimizing energy
Equilibrating
Running simulation
Simulation finished in 25.96 seconds
Running simulation for ejm_31 and ejm_42 in water
Running simulation for lambdas 0.00_0.30_0.00_0.70
here
Minimizing energy
Equilibrating
Running simulation
Simulation finished in 26.08 seconds
Running simulation for ejm_31 and ejm_42 in water
Running simulation for lambdas 0.00_0.10_0.00_0.90
here
Minimizing energy
Equilibrating
Running simulation
Simulation finished in 26.92 seconds
Running simulation for ejm_31 and ejm_42 in water
Running simulation for lambdas 0.00_0.00_0.00_1.00
here
Minimizing energy
Equilibrating
Running simulation
Simulation finished in 25.62 seconds
Running simulation for ejm_31 and ejm_42 in water
Running simulation for lambdas 0.00_0.00_0.50_1.00
here
Minimizing energy
Equilibrating
Running simulation
Simulation finished in 24.39 seconds
Running simulation for ejm_31 and ejm_42 in water
Running simulation for lambdas 0.00_0.00_1.00_1.00
here
Minimizing energy
Equilibrating
Running simulation
Simulation finished in 24.51 seconds
Running simulation for ejm_31 and ejm_42 in protein
Running simulation for lambdas 1.00_1.00_0.00_0.00
here
Minimizing energy
Equilibrating
Running simulation



---------------------------------------------------------------------------

KeyboardInterrupt                         Traceback (most recent call last)

/var/folders/gm/khlf0vn90q162bnc2c48bwfsyk_2dr/T/ipykernel_75472/3005229222.py in ?()
      1 for (liga_name, ligb_name), phase, idx_lambda in product(
      2     pairs, ["water", "protein"], range(len(lambdas_list))
      3 ):
----> 4     run_simulation(liga_name, ligb_name, phase, idx_lambda)


/var/folders/gm/khlf0vn90q162bnc2c48bwfsyk_2dr/T/ipykernel_75472/2715927706.py in ?(liga_name, ligb_name, phase, idx_lambda)
     48 
     49     print("Running simulation", flush=True)
     50     start_time = time.time()
     51     #simulation.step(12_500_000)
---> 52     simulation.step(5_000)
     53 
     54     simulation.saveCheckpoint(
     55         f"./output/{liga_name}_{ligb_name}/{phase}_phase/traj/{lambdas_str}.chk"


~/apps/miniconda3/envs/xd/lib/python3.12/site-packages/openmm/app/simulation.py in ?(self, steps)
    145     def step(self, steps):
    146         """Advance the simulation by integrating a specified number of time steps."""
--> 147         self._simulate(endStep=self.currentStep+steps)


~/apps/miniconda3/envs/xd/lib/python3.12/site-packages/openmm/app/simulation.py in ?(self, endStep, endTime)
    208                     nextSteps = nextReport[i][0]
    209                     anyReport = True
    210             stepsToGo = nextSteps
    211             while stepsToGo > 10:
--> 212                 self.integrator.step(10) # Only take 10 steps at a time, to give Python more chances to respond to a control-c.
    213                 stepsToGo -= 10
    214                 if endTime is not None and datetime.now() >= endTime:
    215                     return


~/apps/miniconda3/envs/xd/lib/python3.12/site-packages/openmm/openmm.py in ?(self, steps)
  11952         ----------
  11953         steps : int
  11954             the number of time steps to take
  11955         """
> 11956         return _openmm.LangevinMiddleIntegrator_step(self, steps)


KeyboardInterrupt: 
```



```python
(liga_name, ligb_name) = pairs.values[idx_pair]
print(f'compute energy for {liga_name} and {ligb_name}', flush=True)

with open(f"./output/{liga_name}_{ligb_name}/{phase}_phase/lambdas.pkl", "rb") as f:
    lambdas_list = pickle.load(f)
lambdas = lambdas_list[idx_lambda]
(elec0, vdw0), (elec1, vdw1) = lambdas
lambdas_str = f"{elec0:.2f}_{vdw0:.2f}_{elec1:.2f}_{vdw1:.2f}"
print(f"for lambdas {lambdas_str}", flush=True)


## deserialize the system
with open(
    f"./output/{liga_name}_{ligb_name}/{phase}_phase/sys/{elec0:.2f}_{vdw0:.2f}_{elec1:.2f}_{vdw1:.2f}.xml",
    "r",
) as f:
    system = mm.XmlSerializer.deserialize(f.read())

## add barostat
if phase != 'vacuum':
    system.addForce(mm.MonteCarloBarostat(1 * unit.atmospheres, 298.15 * unit.kelvin))

with open(f"./output/{liga_name}_{ligb_name}/{phase}_phase/topology.pkl", "rb") as f:
    topology = pickle.load(f)
topology = mdtraj.Topology.from_openmm(topology)

pdb = app.PDBFile(f"./output/{liga_name}_{ligb_name}/{phase}_phase/system.pdb")

integrator = mm.LangevinMiddleIntegrator(
    298.15 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
)
kbT = 298.15 * unit.kelvin * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

platform = mm.Platform.getPlatformByName("CPU")
simulation = app.Simulation(topology, system, integrator, platform)

print("Simulation object created", flush=True)

start_time = time.time()

## load trajectories
reduced_u = []
for lambdas in lambdas_list:
    (elec0, vdw0), (elec1, vdw1) = lambdas
    lambdas_str_traj = f"{elec0:.2f}_{vdw0:.2f}_{elec1:.2f}_{vdw1:.2f}"
    traj_0 = mdtraj.load_dcd(
        f"./output/{liga_name}_{ligb_name}/{phase}_phase/traj/{lambdas_str_traj}.dcd",
        top=topology,
        stride = 1
    )

    traj_1 = mdtraj.load_dcd(
        f"./output/{liga_name}_{ligb_name}/{phase}_phase/traj_1/{lambdas_str_traj}.dcd",
        top=topology,
        stride = 1
    )

    traj = mdtraj.join([traj_0, traj_1])
    

    reduced_u.append([])
    for xyz, unit_cell_vectors in zip(traj.xyz, traj.unitcell_vectors):
        simulation.context.setPositions(xyz)
        simulation.context.setPeriodicBoxVectors(*unit_cell_vectors)
        u = simulation.context.getState(getEnergy=True).getPotentialEnergy() / kbT
        reduced_u[-1].append(u)

    print(f"Computed energies for {lambdas_str_traj}", flush=True)


reduced_u = np.array(reduced_u)

print(f"Time taken: {time.time() - start_time}", flush=True)

os.makedirs(f"./output/{liga_name}_{ligb_name}/{phase}_phase/reduced_potentials", exist_ok=True)
with open(f"./output/{liga_name}_{ligb_name}/{phase}_phase/reduced_potentials/{lambdas_str}.pkl", "wb") as f:
    pickle.dump(reduced_u, f)
```


```python
pairs = pd.read_csv("./script/pairs.csv")

kbT = 298.15 * unit.kelvin * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
kbT = kbT.value_in_unit(unit.kilocalorie_per_mole)

#task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
task_id = 7
t = np.linspace(math.sqrt(1.0 / 10), math.sqrt(1 / 2500), num=8)
n_list = 1 / t**2
n_list = n_list.astype(int)

n = n_list[task_id % len(n_list)]
repeat_id = task_id // len(n_list)
repeat_size = 2500

start_frame = repeat_id * repeat_size
end_frame = repeat_id * repeat_size + n


## read potential energy
us = {}
for (liga_name, ligb_name), phase in product(
    pairs.values, ["water", "protein"]
):
    print(f"{liga_name} and {ligb_name} in {phase}", flush=True)

    with open(f"./output/{liga_name}_{ligb_name}/{phase}_phase/lambdas.pkl", "rb") as f:
        lambdas_list = pickle.load(f)

    us_tmp = []
    for lambdas in lambdas_list:
        (elec0, vdw0), (elec1, vdw1) = lambdas
        lambdas_str = f"{elec0:.2f}_{vdw0:.2f}_{elec1:.2f}_{vdw1:.2f}"

        with open(
            f"./output/{liga_name}_{ligb_name}/{phase}_phase/reduced_potentials/{lambdas_str}.pkl",
            "rb",
        ) as f:
            u = pickle.load(f)
            
            u = u[:, start_frame:end_frame]
            u = u.reshape(-1)
            us_tmp.append(u)
    us[(liga_name, ligb_name, phase)] = np.array(us_tmp)


# ## run fastmbar for each edge seperately
# deltaF_mbar = {}
# for k, u in us.items():
#     num_conf = np.array([u.shape[1] // u.shape[0] for _ in range(u.shape[0])])
#     mbar = FastMBAR(u, num_conf, verbose=True, method="L-BFGS-B")
#     deltaF_mbar[k] = {
#         "mode": mbar.DeltaF[0, -1].item()*kbT,
#     }


## run bayesmbar for each edge seperately
deltaF_bmbar = {}
for k, u in us.items():
    num_conf = np.array([u.shape[1] // u.shape[0] for _ in range(u.shape[0])])
    bayesmbar = BayesMBAR(u, num_conf, random_seed=task_id, verbose=False)
    deltaF_bmbar[k] = {
        "mode": bayesmbar.DeltaF_mode[0, -1].item() * kbT,
        "mean": bayesmbar.DeltaF_mean[0, -1].item() * kbT,
        "std": bayesmbar.DeltaF_std[0, -1].item() * kbT,
    }
    print(k, flush=True)

start_time = time.time()
## run bayescmbar
deltaF_bcmbar = {}
for phase in ["water", "protein"]:
    u_list = [us[(liga, ligb, phase)] for (liga, ligb) in pairs.values]
    num_conf_list = [
        [u.shape[1] // u.shape[0] for i in range(u.shape[0])] for u in u_list
    ]

    end_states = defaultdict(list)
    for i, (liga, ligb) in enumerate(pairs.values):
        end_states[liga].append((i, 0))
        end_states[ligb].append((i, len(lambdas_list) - 1))

    identical_states = [v for k, v in end_states.items()]
    bcmbar = BayesCMBAR(
        u_list,
        num_conf_list,
        identical_states,
        random_seed=task_id,
        verbose=False,
        sample_size=1000,
    )

    for i, (liga, ligb) in enumerate(pairs.values):
        deltaF_bcmbar[(liga, ligb, phase)] = {
            "mode": bcmbar.DeltaF_mode[i][0, -1].item() * kbT,
            "mean": bcmbar.DeltaF_mean[i][0,-1].item()*kbT,
            "std": bcmbar.DeltaF_std[i][0,-1].item()*kbT,
        }

    print(phase, flush=True)

results = {'bmbar': deltaF_bmbar, 'bcmbar': deltaF_bcmbar}

print('Time:', time.time() - start_time)

os.makedirs("./output/results", exist_ok=True)
with open(f"./output/results/n_{n}_repeat_{repeat_id}.pkl", "wb") as f:
    pickle.dump(results, f)
```
