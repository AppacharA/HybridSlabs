from hybrid_slab_gen import scale_amorphous_region, set_selective_dynamics
from mpmorph.workflows.converge import get_converge_wf
from mpmorph.runners.amorphous_maker import get_random_packed
from fireworks import LaunchPad, Workflow
from atomate.vasp.fireworks.core import OptimizeFW

from pymatgen import Composition
from pymatgen.io.vasp import Poscar
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.ext.matproj import MPRester
from pymatgen.core.surface import SlabGenerator, generate_all_slabs

from math import ceil
from datetime import datetime

#A script that will generate the hybrid slab of a given material, and then create the appropriate workflow for amorphization.

#General parameters (what orientation of slab to make, how many layers to amorphize, etc.)

def get_hybrid_slab_amorph_wf(base_matl, vol_exp=1.2)

    orientation = (0, 0, 1)
    n_total_layers = 10
    n_layers_to_amorph = 5
    volume_scale_factor = vol_exp

    #First, we retrieve the structure.
    #with MPRester() as m:
    #        base_matl = m.get_structure_by_material_id("mp-127")

    #Then, generate the slab using Pymatgen's builtin functionality.
    slabgen = SlabGenerator(base_matl, orientation, n_total_layers, 1, in_unit_planes= True, max_normal_search = 1)

    all_slabs = slabgen.get_slabs()

    matl_slab = all_slabs[0]

    matl_slab_unit_cell = matl_slab.oriented_unit_cell

    #Make supercell of the slab and its unit cell.
    matl_slab.make_supercell([3,3,1])

    matl_slab_unit_cell.make_supercell([3,3,1])

    #Amorphize the slab and set selective dynamics.
    hybrid_matl_slab = scale_amorphous_region(matl_slab, matl_slab_unit_cell, n_layers_to_amorph, volume_scale_factor)

    sd_poscar = set_selective_dynamics(matl_slab_unit_cell, matl_slab, n_layers_to_amorph)

    #Write the poscar file, read back in the structure.
    sd_poscar.write_file(base_matl.formula + "_poscar_SD.vasp")

    sd_slab = sd_poscar.structure


    #create convergence workflow.
    prod_args =  {'optional_fw_params': {'override_default_vasp_params': {'user_incar_settings': {'LCHARG': 'TRUE', 'NCORE': '36'}}}} #Overriding default vasp settings to force LCHARG to true...

    wf = get_converge_wf(sd_slab, temperature = 5000, target_steps = 10000, preconverged=True, prod_args = prod_args)

    return wf


def get_bulk_amorph_wf(base_matl, vol_exp=1.2):
    #Get reduced formula, and also identify how many atoms to put into random packing.
    reduced_formula = base_matl.composition.reduced_formula
    reduced_comp = Composition(reduced_formula)

    #We want it to be the lowest number above 100 that perfectly represents the formula.
    natoms = ceil((100/reduced_comp.num_atoms)) * reduced_comp.num_atoms
    
    #To ensure it will be greater than 100...(edge case for if 100 atoms perfectly represents full formula).
    if natoms == 100:
        natoms = natoms + reduced_comp.num_atoms
    


    base_vol_per_atom = base_matl.volume / (len(base_matl.sites))

    #Assuming that the material is in ground state....
    randomPacking = get_random_packed(reduced_formula, target_atoms = natoms, vol_per_atom = vol_exp * base_vol_per_atom)

    prod_args =  {'optional_fw_params': {'override_default_vasp_params': {'user_incar_settings': {'LCHARG': 'TRUE', 'NCORE': '36'}}}} #Overriding default vasp settings to force LCHARG to true...

    wf = get_converge_wf(randomPacking, temperature = 5000, target_steps = 10000, preconverged = True, prod_args = prod_args)

    return wf


def get_bulk_crystal_wf(base_matl, custom_incar_settings = {}):
    #Do we need to supercell this...>? Or can we just submit it to a crystalline calculation....

    tag = datetime.utcnow(().strftime('%Y-%m-%d-%H-%M-%S-%f')
    
    vasp_set = MPRelaxSet(base_matl, user_incar_settings = custom_incar_settings) #Do we need a force_gamma=True here?

    fws = [OptimizeFW(structure=base_matl, vasp_input_set=vasp_set, name="{} Crystalline Structure Optimization".format(tag))]
    
    wf = Workflow(fws, name=base_matl.composition.reduced_formula + "_Bulk_Crystalline_Energy")

    return wf

    

def get_all_interfacial_workflows(base_matl):

    #Get hybrid slab.
    hybrid_slab_amorph_wf = get_hybrid_slab_amorph_wf(base_matl)

    #Get bulk amorph wf.
    bulk_amorph_wf = get_bulk_amorph_wf(base_matl)

    #Get bulk crystalline..
    bulk_crystal_wf = get_bulk_crystal_wf(base_matl)

    lp = LaunchPad.auto_load()
    lp.add_wf(bulk_crystal_wf)
    lp.add_wf(bulk_amorph_wf)
    lp.add_wf(hybrid_slab_amorph_wf)

if __name__ == "__main__":

    with MPRester() as m:
        base_matl = m.get_structure_by_material_id("mp-127")


    lp = LaunchPad.auto_load()
    lp.add_wf(wf)
