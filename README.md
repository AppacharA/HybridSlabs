A library built on Pymatgen to generate HybridSlabs: Slabs with a solid-liquid interface.

# Installation
This package requires pymatgen (and therefore python) to be installed. 


1. Clone the repository to your local machine (`git clone <HTTPS-URL-HERE>`)
2. Navigate to the folder and run `pip install -e . `

# Usage

> First you'll need to create a unit cell....

` 
#Create a basic structure 
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

lattice = Lattice.cubic(2.950)
Au = Structure(lattice, ["Au"],
               [[0,0,0], 
               ])

`

> Next you'll create the hybrid slab.

`
#Create the Hybrid Slab

from HybridSlabs.hybrid_slab import HybridSlab, HybridSlabGenerator

orientation = (1, 1, 1)
thickness = 21 #Thickness in angstroms.
slabgen_args = {"initial_structure":Au, 
                "miller_index":orientation, 
                "min_slab_size":thickness, 
                "min_vacuum_size":1, 
                "max_normal_search":1} #This last parameter is just to ensure that the c-vector is as orthogonal as possible to the a-b lattice plane

fraction_amorph = 0.66 #Proportion of slab you are amorphizing...
volume_scale_factor = 1.1435 #molten density 7790kg/m3, solid density 8908kg/m3

hybrid_generator = HybridSlabGenerator(fraction_amorph, volume_scale_factor, supercell_params = [3,3,1], slabgen_args=slabgen_args)

hybrid_Au_slab = hybrid_generator.generate_hybrid_slab()


`
