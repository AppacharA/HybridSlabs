from hybrid_slab_gen import scale_amorphous_region, set_selective_dynamics
from mpmorph.workflows.converge import get_converge_wf
from fireworks import LaunchPad
from pymatgen.io.vasp import Poscar
from pymatgen.ext.matproj import MPRester
from pymatgen.core.surface import SlabGenerator, generate_all_slabs

#A script that will generate the hybrid slab of a given material, and then create the appropriate workflow for amorphization.

#General parameters (what orientation of slab to make, how many layers to amorphize, etc.)

orientation = (0, 0, 1)
n_total_layers = 10
n_layers_to_amorph = 5
volume_scale_factor = 1.2

#First, we retrieve the structure.
with MPRester() as m:
	base_matl = m.get_structure_by_material_id("mp-127")

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

lp = LaunchPad.auto_load()
lp.add_wf(wf)
