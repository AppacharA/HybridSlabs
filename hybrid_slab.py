from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.sites import PeriodicSite
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


	
#		hybrid_matl_slab_sd_poscar = Poscar.from_string(Hybrid_Slab.to("poscar"))
#
#		return Hybrid_Slab, hybrid_matl_slab_sd_poscar
	#	print("A feasible packing was not found at tolerance {}A: The shortest distances between atoms are now {}. Consider adjusting your hybrid slab parameters: atomic distances less than 1A can lead to major VASP issues during amorphization.".format(tolerance, shortest_dists))
		
		#If all goes well and the structure is initialized, then print out the solution.
	#	print("A feasible packing was found at overall tolerance {}A.".format(tolerance))

