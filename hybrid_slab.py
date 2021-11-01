from pymatgen.core.structure import Structure



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
		miller_index,
		oriented_unit_cell,
		amorph_vol_expansion,
		num_amorph_sites,
		num_crystalline_sites,
		lattice, 
		species, 
		coords,
		charge,
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

		super().__init__(
		lattice,
		species,
		coords,
		charge,
		validate_proximity=validate_proximity,
		to_unit_cell=to_unit_cell,
		coords_are_cartesian=coords_are_cartesian,
		site_properties=site_properties

	)
			
