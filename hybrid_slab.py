from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.sites import PeriodicSite

from pymatgen.core.surface import SlabGenerator, generate_all_slabs

from hybrid_slab_gen import scale_amorphous_region, set_selective_dynamics

import numpy as np

import warnings



#A Python Class representing the hybrid slab. It will extend the standard Pymatgen Structure class.

class HybridSlab(Structure):

	"""
	Subclass of Structure representing a Hybrid Slab, i.e. one that has an amorphous-crystalline interface.

	Specific attributes that HybridSlab will record, past those of just a regular structure are:

	1. Orientation of crystalline surface (inherited? from Slab)
	2. Volume Scale factor of amorphous region per atom (i.e. how much bigger slab is in amorphous region than crystalline)
	3. Number of amorphous sites
	4. Number of crystalline sites

	Note that the slab when amorphized only amorphizes in layers i.e. if you have 9 atoms per plane you can't amorphize 10 atoms, you can only amorphize multiples of 9. 



	"""

	def __init__(
		self,
		lattice, 
		species, 
		coords,
		miller_index,
		oriented_unit_cell,
		amorph_vol_expansion,
		num_amorph_sites,
		num_crystalline_sites,
		validate_proximity = False,
		to_unit_cell = False,
		coords_are_cartesian = False,
		site_properties = None,
	    ):


		self.miller_index = miller_index
		
		self.oriented_unit_cell = oriented_unit_cell

		self.amorph_vol_expansion = amorph_vol_expansion,

		self.num_amorph_sites = num_amorph_sites

		self.num_crystalline_sites = num_crystalline_sites



		

		#Final initialization.

		super().__init__(
		lattice,
		species,
		coords,
		validate_proximity=validate_proximity,
		to_unit_cell=to_unit_cell,
		coords_are_cartesian=coords_are_cartesian,
		site_properties=site_properties

	)
	

		#Do some sanity checks..

		all_dists = self.distance_matrix[np.triu_indices(len(self), 1)]

		shortest_dists = np.sort(all_dists)[:10]
		if np.any(shortest_dists <= 0.85): #This was the lowest limit we found before slabs consistently started to fail amorphization more often than not.

			raise ValueError("Atoms are too close to each other for proper amorphization.")

		elif np.any(shortest_dists <= 1):
	
 
			warnings.warn("Some atoms are less than 1A away from each other: The ten shortest distances between atoms are now {}. This may lead to issues during amorphization. Consider repacking the slab with different parameters.".format(shortest_dists))


		#If it passes tolerance check, we do a sanity check on the selective dynamics.

		selective_dynamics = site_properties.get("selective_dynamics", None)

		if not selective_dynamics:
			raise ValueError("No selective dynamics set on Hybrid Slab. Aborting structure creation.")

		else:

			sd_flags = np.asarray(selective_dynamics)

			if np.all(sd_flags):

				warnings.warn("All atoms are set to mobile. Perhaps your amorphization region was set to cover the whole slab?")

			elif np.all(sd_flags == False):
		
				warnings.warn("All atoms are set to be immobile. Did you specify an amorphization region?") 	
	


	def as_dict(self):
		"""
		:return: MSONAble dict
		"""
		d = super().as_dict()
		d["@module"] = type(self).__module__
		d["@class"] = type(self).__name__
		d["oriented_unit_cell"] = self.oriented_unit_cell.as_dict()
		d["miller_index"] = self.miller_index
		d["amorph_vol_expansion"] = self.amorph_vol_expansion
		d["num_amorph_sites"] = self.num_amorph_sites
		d["num_crystalline_sites"] = self.num_crystalline_sites
		return d

	@classmethod
	def from_dict(cls, d):
		"""
		:param d: dict
		:return: Creates HybridSlab from dict.
		"""
		lattice = Lattice.from_dict(d["lattice"])
		sites = [PeriodicSite.from_dict(sd, lattice) for sd in d["sites"]]
		s = Structure.from_sites(sites)

		return HybridSlab(
		    lattice=lattice,
		    species=s.species_and_occu,
		    coords=s.frac_coords,
		    miller_index=d["miller_index"],
		    oriented_unit_cell=Structure.from_dict(d["oriented_unit_cell"]),
		    amorph_vol_expansion = d["amorph_vol_expansion"],
		    num_amorph_sites = d["num_amorph_sites"],
		    num_crystalline_sites = d["num_crystalline_sites"],
		    site_properties=s.site_properties,
		)


class HybridSlabGenerator:


	def __init__(
	
		self,
		fraction_amorph,
		volume_scale_factor,
		slabgen_args,
		supercell_params=[1,1,1],
		amorphous_matl_unit_cell = None,
		debug = False,

		):


			self.fraction_amorph = fraction_amorph,
			self.volume_scale_factor = volume_scale_factor,
			self.supercell_params = supercell_params,
			self.slabgen_args = slabgen_args,
			self.amorphous_matl_unit_cell = amorphous_matl_unit_cell
			self.debug = debug



	def generate_hybrid_slab(self):
	
		percent_amorphized=self.fraction_amorph[0]
		volume_scale_factor=self.volume_scale_factor[0]
		supercell_params=self.supercell_params[0]
		slabgen_args= self.slabgen_args[0]
		amorphous_matl_unit_cell=self.amorphous_matl_unit_cell
		debug=self.debug


		#Helper function to create hybrid slab with desired selective dynamics flags.
		#Returns a Pymatgen POSCAR Object.
		#Takes in as input a percentage amorphization, and then the standard set of arguments to generate a slab through Pymatgen's preexisting functions.

		if not slabgen_args:
			raise ValueError("No arguments provided for slab generator.")
		

		slabgen = SlabGenerator(**slabgen_args)

		all_slabs = slabgen.get_slabs()

		orig_slab = all_slabs[0]

		supercell_matl_slab = orig_slab.copy()


		supercell_matl_slab_unit_cell = orig_slab.oriented_unit_cell.copy()

		#Make supercell of the slab and its unit cell.

		supercell_matl_slab.make_supercell(supercell_params)

		supercell_matl_slab_unit_cell.make_supercell(supercell_params)

		#If the amorphous matl is explicitly specified, this is a multicomposition slab, so we need to do some preprocessing.
		if amorphous_matl_unit_cell:
			cryst_matl_unit_cell = slabgen_args['initial_structure']

			cryst_atomic_vol = (cryst_matl_unit_cell.volume / cryst_matl_unit_cell.num_sites)

			amorph_atomic_vol = (amorphous_matl_unit_cell.volume/amorphous_matl_unit_cell.num_sites)
		
			#We identify how much the volume differs b/w the two materials, and incorporate it into the final volume expansion factor.
			atomic_vol_ratios =  (amorph_atomic_vol/cryst_atomic_vol)
			
			volume_scale_factor = volume_scale_factor * atomic_vol_ratios







		#Amorphize the slab and set selective dynamics.
		(hybrid_matl_slab, n_amorph_sites) = scale_amorphous_region(supercell_matl_slab, supercell_matl_slab_unit_cell, percent_amorphized, volume_scale_factor, debug=debug, c_tolerance=2.0)

		struct = set_selective_dynamics(supercell_matl_slab_unit_cell, hybrid_matl_slab, percent_amorphized)


		#Finally, adjust the species in the amorphous region to match the liquid matl species.
		if amorphous_matl_unit_cell:

			#Assuming single component material...
			amorph_species = amorphous_matl_unit_cell.species[0]

			heterogen_site_properties = struct.site_properties
		
				
			for i in range(n_amorph_sites):
		
				#we replace sites from the "back" of the molecule, since that is where the amorphous region is.
				struct.replace(-1 * (i+1), amorph_species)



				#We must also update the site properties....
				heterogen_site_properties['selective_dynamics'][-1 * (i+1)]=[True, True, True]	


			#Copy over the new structure.
			struct = struct.copy(site_properties = heterogen_site_properties)	


		Hybrid_Slab = HybridSlab(
		struct.lattice,
		struct.species,
		struct.frac_coords,
		orig_slab.miller_index, 
		supercell_matl_slab_unit_cell, 
		volume_scale_factor,  #WHY IS THIS ARGUMENT ASSIGNED AS TUPLE???
		n_amorph_sites,
		struct.num_sites - n_amorph_sites,
		site_properties = struct.site_properties,
		)


		return Hybrid_Slab # Poscar(Hybrid_Slab)


	
#		hybrid_matl_slab_sd_poscar = Poscar.from_string(Hybrid_Slab.to("poscar"))
#
#		return Hybrid_Slab, hybrid_matl_slab_sd_poscar
	#	print("A feasible packing was not found at tolerance {}A: The shortest distances between atoms are now {}. Consider adjusting your hybrid slab parameters: atomic distances less than 1A can lead to major VASP issues during amorphization.".format(tolerance, shortest_dists))
		
		#If all goes well and the structure is initialized, then print out the solution.
	#	print("A feasible packing was found at overall tolerance {}A.".format(tolerance))





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



	from pymatgen.ext.matproj import MPRester

	#Running a test case with Nickel....
	orientation = (1, 1, 1)
	thickness = 21 #Thickness in angstroms.
	fraction_amorph = 0.66 #Proportion of slab you are amorphizing...
	volume_scale_factor = 1.1435 #molten density 7790kg/m3, solid density 8908kg/m3

	#First, we retrieve the structure.
	with MPRester() as m:
		base_matl = m.get_structure_by_material_id("mp-81")


	
	
	slabgen_args = {"initial_structure":base_matl, "miller_index":orientation, "min_slab_size":thickness, "min_vacuum_size":1, "max_normal_search":1}
	
	hybrid_generator = HybridSlabGenerator(fraction_amorph, volume_scale_factor, supercell_params = [3,3,1], slabgen_args=slabgen_args, debug=False)

	hybrid_Au_slab = hybrid_generator.generate_hybrid_slab()

	
	hybrid_Au_slab.to("poscar", f"{hybrid_Au_slab.composition.reduced_formula}_{thickness}AngstromSlab_test.vasp")

	#Another test case, this time with multi composition slab.	

	with MPRester() as m:

		Sodium = m.get_structure_by_material_id("mp-127")
		Nickel = m.get_structure_by_material_id("mp-23")


	hybrid_generator = HybridSlabGenerator(fraction_amorph, volume_scale_factor, supercell_params = [3,3,1], amorphous_matl_unit_cell=Sodium, slabgen_args=slabgen_args)
	hybrid_NaNi_slab = hybrid_generator.generate_hybrid_slab()




