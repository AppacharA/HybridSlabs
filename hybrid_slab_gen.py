#A basic script to take in a slab, and then scale it accordingly.

from packmol_parallelipiped_generator import packmol_gen_parallelipiped_random_packing

from pymatgen.core.surface import SlabGenerator, generate_all_slabs, Structure, Lattice

from pymatgen.io.vasp import Poscar

from pymatgen.ext.matproj import MPRester

import numpy as np
from math import floor

import os
from pathlib import Path

def scale_amorphous_region(orig_slab, orig_oriented_unit_cell, percentage_amorphized, scale_factor, debug=False):#Take in as arguments the original slab structure, and the number of layers to amorphize, and the scaling factor. 
#Return a Structure representing a hybrid slab.
	
	#Orig_slab and orig_oriented_unit_cell are passed in separately. This is to allow for compatibility with structures that are not Pymatgen Slab objects, e.g.
	#Supercelled structures.

	#Determine how many layers the slab has.This is done by looking at how many atoms are in the same c-coordinate position...
	c_coord = orig_slab.frac_coords[0, 2] #First fractional c-coordinate in the structure.
	
	natoms_per_layer = len(orig_slab.frac_coords[orig_slab.frac_coords[:, 2] == c_coord])
	nlayers = orig_slab.num_sites / natoms_per_layer	
	

	if debug:
		print("natoms per layer: " + str(natoms_per_layer))
		print("nlayers: " + str(nlayers))

	#First, some basic bookkeeping checks. Make sure that number of layers specified is not greater than layers in slab.
	
	if percentage_amorphized > 1:
		print ("Fraction of slab to be amorphized exceeds 100%. The slab has not been modified.")
	
		#Return unmodified original slab. 
		return orig_slab
		

	else: #Suppose you have specified a proper number of layers to  be amorphized. They shall be amorphized from the end region.

	#The basic algorithm here is straightforward.
	#First, we remove the vacuum in the slab.
	#Then, we identify the region that needs to be amorphized, and the region that will remain crystalline.
	#For the crystalline region, we copy over the coordinates.
	#For the amorphous region, we use packmol to get the coordinates of a random packing, relative to the origin.
	#Then, we add back in the length of the crystalline region.
	#Then, we combine those coordinates with the coordinates of the crystalline region.
	#Finally, we normalize all the coordinates in the structure against the expanded lattice length (since the amorphous part of the slab was just stretched, we want to avoid the amorphous region having coordinates > 1).
		
		#use percentage to determine how many amorphous layers must be made...
		n_amorph_layers = floor(nlayers * percentage_amorphized)
		
		#First, we will extract the vacuumless slab from the original slab. 

		#Get Lattice matrix from unit cell.
		unit_lattice_matrix = orig_oriented_unit_cell.lattice.matrix
		
                #All of our transforms occur on the c axis, so we'll store the c-vectors and their lengths directly, to minimize function calls later.
		unit_c_vector = unit_lattice_matrix[2]
		unit_c_vector_len = np.linalg.norm(unit_c_vector) 
#create new lattice vector that is equal to n-unit cells stacked together on c-axis.n is determined by comparison of atoms in unit cell to atoms total in the structure.

		nonvac_lattice_matrix = unit_lattice_matrix.copy()
	
		nonvac_lattice_matrix[2] = (orig_slab.num_sites/orig_oriented_unit_cell.num_sites) * unit_c_vector

		#nonvac_lattice_matrix[2] = nlayers * unit_c_vector

		nonvac_c_vector = nonvac_lattice_matrix[2]
		nonvac_c_vector_len = np.linalg.norm(nonvac_c_vector)
	
		#Next, we modify the atomic coordinates from the original slab. 
		
		#Get original fractional coordinates and c-vector.
		orig_frac_coords = orig_slab.frac_coords

		orig_c_vector = orig_slab.lattice.matrix[2]
		orig_c_vector_len = np.linalg.norm(orig_c_vector)
	

		#Create scaling matrix.
		scaling_matrix = np.array([
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, (orig_c_vector_len/nonvac_c_vector_len)]

			])

		#Use above scaling matrix to reposition atoms.We reassign fractional coordinates based on length of a slab that doesn't have a vacuum. 
		nonvac_frac_coords = np.matmul(orig_slab.frac_coords, scaling_matrix)	



		if debug == True:
			#For debugging, create and print the vacuumless slab as a POSCAR.
			nonVacSlab = Structure(nonvac_lattice_matrix, orig_slab.species, nonvac_frac_coords)
			nonVacSlab.to("poscar", "nonvac_slab.vasp")	
		#Now we have a slab without a vacuum. We will now begin to apply the amorphization transformation.
	        #We must account for a volume change during amorphization, so there is a scaling process involved as well.
                #Scaling only occurs along c-axis.

	
		#First, create the scaled lattice.
		scaled_lattice_matrix = nonvac_lattice_matrix.copy()


		#Identify the amorphous and crystalline regions.
		
		#amorphous_region_length = n_amorph_layers * unit_c_vector_len
		amorphous_region_length = (n_amorph_layers/nlayers) * nonvac_c_vector_len

		crystalline_region_length = nonvac_c_vector_len - amorphous_region_length

		#Get length of crystal region in fractional coordinates.
		frac_cryst_len = crystalline_region_length / nonvac_c_vector_len

		
		#Set length of C-vector in expanded lattice based on how much the amorphous layer is expanding. Keep for later.
		total_slab_expansion_factor = 1 + (((scale_factor - 1) * amorphous_region_length) / nonvac_c_vector_len)


		scaled_lattice_matrix[2] = scaled_lattice_matrix[2] * (total_slab_expansion_factor)

		scaled_lattice_c_vector = scaled_lattice_matrix[2]
		scaled_lattice_c_vector_len = np.linalg.norm(scaled_lattice_c_vector)
		
		#Next, get number of sites that will be amorphous.
		n_amorphous_sites = int(n_amorph_layers * natoms_per_layer)
			
		#Create array for scaled coords, and reorder them to be in strictly ascending value of c-position.
		scaled_frac_coords = nonvac_frac_coords.copy()
		scaled_frac_coords = scaled_frac_coords[scaled_frac_coords[:, 2].argsort()] #https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column/2828121#2828121
	
		if debug == True:
			print("Scaled Frac Coords: ")
			print(scaled_frac_coords)

		#Using packmol, get random packed structure of amorphous atoms.

		amorph_struct = packmol_gen_parallelipiped_random_packing(n_amorph_layers, natoms_per_layer, orig_oriented_unit_cell, scale_factor, cleanup=not debug)
		#Extract frac coords of amorphous structure. These are fractional in relation to scale_factor * original_amorphous_length
		amorph_frac_coords = amorph_struct.frac_coords


		#recompute coordinates in c direction to be relation to original, non_expanded slab length. So,
		#Each C-coordinate will be multiplied by scale factor * amorphous_length  and then divided by total original slab length.
		
		scaling_matrix = np.array([

		[1, 0, 0],
		[0, 1, 0],
		[0, 0, (scale_factor*amorphous_region_length)/(nonvac_c_vector_len)]

		])

	
		amorph_frac_coords = np.matmul(amorph_frac_coords, scaling_matrix)


		#Add back in the crystal length


		amorph_frac_coords[:, 2] = amorph_frac_coords[:, 2] + frac_cryst_len


		#Add everything back, then renormalize to expanded lattice.
		scaled_frac_coords[-1 * n_amorphous_sites:] = amorph_frac_coords


	
		normalization_matrix = np.array([
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, nonvac_c_vector_len/scaled_lattice_c_vector_len]

		])
	
		scaled_frac_coords = np.matmul(scaled_frac_coords, normalization_matrix)	



		#Finally, create new Structure from all the transforms.
		#Create Lattice.
		scaled_lattice = Lattice(scaled_lattice_matrix)


		#Obtain species from original slab.
		species = orig_slab.species	

		#Generate new structure.
		new_slab = Structure(scaled_lattice, species, scaled_frac_coords)




	
		return new_slab




def set_selective_dynamics(unit_cell, slab, percentage_amorphized): #A method to set selective dynamics flags for a hybrid slab.
	#Using Unit Cell, identify all sites that are amorphous and non-amorphous. Unit cell should be oriented.

	#Determine how many layers the slab has.This is done by looking at how many atoms are in the same c-coordinate position...
	c_coord = slab.frac_coords[0, 2] #First fractional c-coordinate in the structure.
	natoms_per_layer = len(slab.frac_coords[slab.frac_coords[:, 2] == c_coord])
	nlayers = slab.num_sites / natoms_per_layer	

	#use percentage to determine how many amorphous layers must be made...
	n_amorph_layers = floor(nlayers * percentage_amorphized)
	
	n_amorph_sites = int(n_amorph_layers * natoms_per_layer)

	n_crystalline_bulk_sites = int(slab.num_sites - n_amorph_sites) #- n_crystalline_interface_sites
	#Create 3x3 boolean arrays. By design, these hybrid slabs are such that amorphous region will be at the bottom of the file/end of the list of sites.

	#False is 0. All crystal sites will have Selective Dynamics False as their setting, because they are not allowed to move in the simulation.

	cols = 3
	crystal_bulk_flags = [[0 for i in range(cols)] for j in range(n_crystalline_bulk_sites)]

	#Set True flags for two interface layers.
	#crystal_interface_flags = [[1 for i in range(cols)] for j in range(n_crystalline_interface_sites)]




	amorph_flags = [[1 for i in range(cols)] for j in range(n_amorph_sites)]


	#Combine the flags.
	combined_flags = crystal_bulk_flags + amorph_flags #+ crystal_interface_flags


	#Create final poscar file with the Selective Dynamics Flags.
	poscar = Poscar(slab, selective_dynamics=combined_flags)

	return poscar


def generate_hybrid_slab(percent_amorphized, volume_scale_factor, supercell_params=[1,1,1], *args, **kwargs):
	#Helper function to create hybrid slab with desired selective dynamics flags.
	#Returns a Pymatgen POSCAR Object.
	#Takes in as input a percentage amorphization, and then the standard set of arguments to generate a slab through Pymatgen's preexisting functions.

	slabgen = SlabGenerator(*args, **kwargs)

	all_slabs = slabgen.get_slabs()

	orig_slab = all_slabs[0]

	supercell_matl_slab = orig_slab.copy()


	supercell_matl_slab_unit_cell = orig_slab.oriented_unit_cell.copy()

	#Make supercell of the slab and its unit cell.

	supercell_matl_slab.make_supercell(supercell_params)

	supercell_matl_slab_unit_cell.make_supercell(supercell_params)

	#Amorphize the slab and set selective dynamics.
	hybrid_matl_slab = scale_amorphous_region(supercell_matl_slab, supercell_matl_slab_unit_cell, percent_amorphized, volume_scale_factor, debug=True)


	hybrid_matl_slab_sd_poscar = set_selective_dynamics(supercell_matl_slab_unit_cell, hybrid_matl_slab, percent_amorphized)

	return hybrid_matl_slab_sd_poscar


if __name__ == "__main__":
	#with MPRester() as m:

	#	Sodium = m.get_structure_by_material_id("mp-127")
	#	GaN = m.get_structure_by_material_id("mp-804")
	#	
	#unit_cells = [("Sodium", Sodium)]
	#for matl in unit_cells:
	#	
	#	#Obtain a slab with 10 unit cell layers and 1 layer of vacuum, oriented in (111).


	#	slabgen001 = SlabGenerator(matl[1], (0, 0, 1), 10, 1, in_unit_planes = True, max_normal_search = 4)


	#	all_slabs001 = slabgen001.get_slabs()

	#	


	#	orig_slab_001 = all_slabs001[0]

	#

	#	#Now, we will make supercell of this slab, and of its unitcell.
	#	
	#	supercell_slab = orig_slab_001.copy()
	#	supercell_slab.make_supercell([3,3,1])

	#	unit_supercell = orig_slab_001.oriented_unit_cell.copy()
	#	unit_supercell.make_supercell([3,3,1])	


	#	#Next, modify slab. Amorphize 5 layers.
	#	scaled_supercell001 = scale_amorphous_region(supercell_slab, unit_supercell, 2.5, 1.0442)

	#	#Finally, set selective dynamics.
	#	sd_poscar = set_selective_dynamics(unit_supercell, scaled_supercell001, 2.5)


	#			
	#	scaled_supercell = (matl[0] + "_001", scaled_supercell001)

	#
	#	scaled_supercell[1].to("poscar", filename = scaled_supercell[0] + "_scaled_supercell_POSCAR_parallelogram.vasp")

	#	sd_poscar.write_file(scaled_supercell[0] + "_scaled_supercell_POSCAR_parallelogram_SD.vasp")



	#Running a test case with Nickel....
	orientation = (1, 1, 1)
	thickness = 14 #Thickness in angstroms.
	fraction_amorph = 0.5 #Proportion of slab you are amorphizing...
	volume_scale_factor = 1.1435 #molten density 7790kg/m3, solid density 8908kg/m3

	#First, we retrieve the structure.
	with MPRester() as m:
		base_matl = m.get_structure_by_material_id("mp-23")

	
	hybrid_Ni_slab = generate_hybrid_slab(fraction_amorph, volume_scale_factor, [3,3,1], base_matl, orientation, thickness, 1, max_normal_search=1)



	#Then, generate the slab using Pymatgen's builtin functionality.
	#slabgen = SlabGenerator(base_matl, orientation, thickness, 1, max_normal_search = 1)

	#all_slabs = slabgen.get_slabs()

	#orig_slab = all_slabs[0]

	#supercell_matl_slab = orig_slab.copy()


	#supercell_matl_slab_unit_cell = orig_slab.oriented_unit_cell.copy()

	##Make supercell of the slab and its unit cell.
	#supercell_matl_slab.make_supercell([3,3,1])

	#supercell_matl_slab_unit_cell.make_supercell([3,3,1])

	#print(supercell_matl_slab)
	#print(supercell_matl_slab_unit_cell)

	##Amorphize the slab and set selective dynamics.
	#hybrid_matl_slab = scale_amorphous_region(supercell_matl_slab, supercell_matl_slab_unit_cell, fraction_amorph, volume_scale_factor, debug=False)


	#sd_poscar = set_selective_dynamics(supercell_matl_slab_unit_cell, hybrid_matl_slab, fraction_amorph)

	#Write the poscar file, read back in the structure.
	filename = base_matl.formula + "poscar_SD.vasp_thickness"

	rootdir = Path(os.getcwd())

	hybrid_Ni_slab.write_file(rootdir/"NiHybridSlab"/filename)
	
