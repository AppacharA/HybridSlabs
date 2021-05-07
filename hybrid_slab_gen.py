#A basic script to take in a slab, and then scale it accordingly.

from packmol_parallelipiped_generator import packmol_gen_parallelipiped_random_packing

from pymatgen.core.surface import SlabGenerator, generate_all_slabs, Structure, Lattice

from pymatgen.io.vasp import Poscar

from pymatgen.ext.matproj import MPRester

import numpy as np

import os

def scale_amorphous_region(orig_slab, orig_oriented_unit_cell, n_amorph_layers, scale_factor, debug=False):#Take in as arguments the original slab structure, and the number of layers to amorphize, and the scaling factor. 
#Return a Structure representing a hybrid slab.
	
	#Orig_slab and orig_oriented_unit_cell are passed in separately. This is to allow for compatibility with structures that are not Pymatgen Slab objects, e.g.
	#Supercelled structures.

	#Determine how many layers the slab has by comparison with oriented unit cell.
	
	#orig_slab_unit_cell = orig_oriented_unit_cell

	nlayers = (orig_slab.num_sites) / (orig_oriented_unit_cell.num_sites)



	#First, some basic bookkeeping checks. Make sure that number of layers specified is not greater than layers in slab.
	
	if n_amorph_layers > nlayers:
		print ("Number of layers to be amorphized exceeds number of layers in slab. The slab has not been modified.")
	
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
		

		
		#First, we will extract the vacuumless slab from the original slab. 

		#Get Lattice matrix from unit cell.
		#unit_lattice_matrix = orig_slab_unit_cell.lattice.matrix
		unit_lattice_matrix = orig_oriented_unit_cell.lattice.matrix
		
		#unit_a_vector = unit_lattice_matrix[0]

		#unit_b_vector = unit_lattice_matrix[1]

                #All of our transforms occur on the c axis, so we'll store the lengths of the c-vectors directly, to minimize function calls later.
		unit_c_vector = unit_lattice_matrix[2]
                unit_c_vector_len = np.linalg.norm(unit_c_vector) 
                
                #create new lattice vector that is equal to n-unit cells stacked together on c-axis.

		nonvac_lattice_matrix = unit_lattice_matrix.copy()

		nonvac_lattice_matrix[2] = nlayers * unit_c_vector

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
		#nonvac_frac_coords = np.matmul(orig_frac_coords, scaling_matrix)
		nonvac_frac_coords = np.matmul(orig_slab.frac_coords, scaling_matrix)	



		if debug == True:
			#For debugging, create and print the vacuumless slab as a POSCAR.
			nonVacSlab = Structure(nonvac_lattice_matrix, orig_slab.species, nonvac_frac_coords)

		
		
		#Now we have a slab without a vacuum. We will now begin to apply the amorphization transformation.
	        #We must account for a volume change during amorphization, so there is a scaling process involved as well.
                #Scaling only occurs along c-axis.

	
		#First, create the scaled lattice.
		scaled_lattice_matrix = nonvac_lattice_matrix.copy()


		#Identify the amorphous and crystalline regions.
		amorphous_region_length = n_amorph_layers * unit_c_vector_len

		crystalline_region_length = nonvac_c_vector_len - amorphous_region_length

		#Get length of crystal region in fractional coordinates.
		frac_cryst_len = crystalline_region_length / nonvac_c_vector_len

		
		#Set length of C-vector in expanded lattice based on how much the amorphous layer is expanding. Keep for later.
		total_slab_expansion_factor = 1 + (((scale_factor - 1) * amorphous_region_length) / nonvac_c_vector_len)


		scaled_lattice_matrix[2] = scaled_lattice_matrix[2] * (total_slab_expansion_factor)

		scaled_lattice_c_vector = scaled_lattice_matrix[2]
                scaled_lattice_c_vector_len = np.linalg.norm(scaled_lattice_c_vector)
		
		#Next, get number of sites that will be amorphous.
		#n_amorphous_sites = n_amorph_layers * orig_slab_unit_cell.num_sites
		n_amorphous_sites = n_amorph_layers * orig_oriented_unit_cell.num_sites

			
		#Create array for scaled coords.
		scaled_frac_coords = nonvac_frac_coords.copy()

		#Using packmol, get random packed structure of amorphous atoms.
		#amorph_struct = packmol_gen_parallelipiped_random_packing(n_amorph_layers, orig_slab_unit_cell, scale_factor)
		amorph_struct = packmol_gen_parallelipiped_random_packing(n_amorph_layers, orig_oriented_unit_cell, scale_factor)
			
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




def set_selective_dynamics(unit_cell, slab, n_amorph_layers): #A method to set selective dynamics flags for a hybrid slab.
	#Using Unit Cell, identify all sites that are amorphous and non-amorphous. Unit cell does not need to be oriented.


	n_amorph_sites = n_amorph_layers * unit_cell.num_sites

	n_crystalline_interface_sites = 2 * unit_cell.num_sites

	n_crystalline_bulk_sites = slab.num_sites - n_amorph_sites - n_crystalline_interface_sites


	#Create 3x3 boolean arrays. By design, these hybrid slabs are such that amorphous region will be at the bottom of the file/end of the list of sites.

	#False is 0. All crystal sites will have Selective Dynamics False as their setting, because they are not allowed to move in the simulation.

	cols = 3
	crystal_bulk_flags = [[0 for i in range(cols)] for j in range(n_crystalline_bulk_sites)]

	#Set True flags for two interface layers.
	crystal_interface_flags = [[1 for i in range(cols)] for j in range(n_crystalline_interface_sites)]




	amorph_flags = [[1 for i in range(cols)] for j in range(n_amorph_sites)]


	#Combine the flags.
	combined_flags = crystal_bulk_flags + crystal_interface_flags + amorph_flags


	#Create final poscar file with the Selective Dynamics Flags.
	poscar = Poscar(slab, selective_dynamics=combined_flags)

	return poscar





if __name__ == "__main__":
	with MPRester() as m:

		Sodium = m.get_structure_by_material_id("mp-127")
		GaN = m.get_structure_by_material_id("mp-804")
		
	unit_cells = [("Sodium", Sodium)]
	for matl in unit_cells:
		
		#Obtain a slab with 10 unit cell layers and 1 layer of vacuum, oriented in (111).


		slabgen001 = SlabGenerator(matl[1], (0, 0, 1), 10, 1, in_unit_planes = True, max_normal_search = 1)



		all_slabs001 = slabgen001.get_slabs()

		


		orig_slab_001 = all_slabs001[0]

		

		#Now, we will make supercell of this slab, and of its unitcell.
		
		supercell_slab = orig_slab_001.copy()
		supercell_slab.make_supercell([3,3,1])

		unit_supercell = orig_slab_001.oriented_unit_cell.copy()
		unit_supercell.make_supercell([3,3,1])	
	

		#Next, modify slab. Amorphize 5 layers.
		
		print("HERE")
		
		
	
		#[scaled_slab001, non_vac_slab001] = scale_amorphous_region(orig_slab_001, 5, 1.2)

		scaled_supercell001 = scale_amorphous_region(supercell_slab, unit_supercell, 5, 1.2)
		

		#Finally, set selective dynamics.
		sd_poscar = set_selective_dynamics(unit_supercell, scaled_supercell001, 5)


				
		scaled_supercell = (matl[0] + "_001", scaled_supercell001)

	
		scaled_supercell[1].to("poscar", filename = scaled_supercell[0] + "_scaled_supercell_POSCAR_parallelogram.vasp")

		sd_poscar.write_file(scaled_supercell[0] + "_scaled_supercell_POSCAR_parallelogram_SD.vasp")
