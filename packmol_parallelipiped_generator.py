#TODO: Implement a cleanup code to clean the directory.
#TODO: Maybe just pass in the lattice vectors to begin with.
import os
import numpy as np
from collections import OrderedDict
from pymatgen.core import Structure


def packmol_gen_parallelipiped_random_packing(nlayers, oriented_unit_cell, scale_factor, cleanup=True):
	#Take in as input the number of layers, the original unit cell, and the scaled c-length of the amorphous region.
	#Return a random packed Pymatgen Structure.

	#First, we must generate a species file for Packmol to pack. 
	#Get composition of elements for unit cell.
	el_dic = oriented_unit_cell.composition.as_dict()



	#Next, we get the parallelogram dimensions. This will be defined through six intersecting planes.

	#First, get lattice matrix of oriented_unit_cell. Extract the a, b, c vectors.
	
	lattice_matrix = oriented_unit_cell.lattice.matrix

	unit_a_vec = lattice_matrix[0]
	unit_b_vec = lattice_matrix[1]
	unit_c_vec = lattice_matrix[2]

	#Get the scaled c vector.
	scaled_c_vec = unit_c_vec * nlayers * scale_factor


	#Next, get the equations for the planes. Will be represented as list [i, j, k, l], where ix + jy + kz = l

	#Fundamentally we do three cross products: axb, cxa, and bxc.
	left_hand_side = np.array([unit_a_vec, unit_c_vec, unit_b_vec])
	right_hand_side = np.array([unit_b_vec, unit_a_vec, unit_c_vec])
	
	basis_planes = np.cross(left_hand_side, right_hand_side) #Gives planes ab, ac, bc

	nonzero_dot_products = -1 * (basis_planes * np.array([scaled_c_vec, unit_b_vec, unit_a_vec])).sum(axis=1) #Multiply by -1 to account for eventual flipping of normals?
	print(nonzero_dot_products)
	#This gives planes spanned, respectively, by a&b, a&c, b&c. All three plane normals point into the box. 
	basis_planes = np.cross(left_hand_side, right_hand_side)

	#Of the six boundary planes, 3 of them intersect the origin. Therefore their l values (calculated via dot product) will be zero. We need to calculate the dot products of the other 3 planes.
       
	#We additionally multiply by -1 in order to orient the plane normals to point into the box.

	#This gives planes spanned, respectively, by a&b, a&c, b&c. All three plane normals point into the box. 
	basis_planes = np.cross(left_hand_side, right_hand_side)

	#Of the six boundary planes, 3 of them intersect the origin. Therefore their l values (calculated via dot product) will be zero. We need to calculate the dot products of the other 3 planes.
       
	#We additionally multiply by -1 in order to orient the plane normals to point into the box.

	nonzero_dot_products = -1 * (basis_planes * np.array([scaled_c_vec, unit_b_vec, unit_a_vec])).sum(axis=1)
	
	#Put it all together.
	boundary_planes = np.zeros((6, 4))

	
	boundary_planes[:3, :3] = basis_planes
	boundary_planes[3:, :3] = -1 * basis_planes
	boundary_planes[3:, 3] = nonzero_dot_products

#	print(boundary_planes)
#	####################
#    
#	plane_ab1 = np.zeros(4) #The plane spanned by vectors a & b. Start with [0,0,0,0]
#	
#	#<ijk> is vector normal to plane. Get i,j,k from cross product.We cross a x b such that vector extends along c axis.
#	plane_ab1[:3] = np.cross(unit_a_vec, unit_b_vec)
#
#	#L is given by dot producting <ijk> with point in plane. AB1 contains point (0,0,0), so L is simply zero. No change is necessary.
#	 
#	
#	#Next, AB2
#	plane_ab2 = np.zeros(4)
#	
#	#AB2 is parallel to AB1. We flip the orientation such that AB2 'faces' AB1 (surface normals point to each other)
#	plane_ab2[:3] = -1 * plane_ab1[:3]
#
#	#AB2 contains point c given by scaled c vector from origin. 
#	plane_ab2[3] = np.dot(plane_ab2[:3], scaled_c_vec)
#
#
#	#Rinse and repeat for AC, BC
#
#	plane_ac1 = np.zeros(4)
#	plane_ac1[:3] = np.cross(unit_c_vec, unit_a_vec) #c x a, such that vector extends along b axis.
#
#	plane_ac2 = np.zeros(4)
#	plane_ac2[:3] = -1 * plane_ac1[:3]
#	plane_ac2[3] = np.dot(plane_ac2[:3], unit_b_vec)
#
#	plane_bc1 = np.zeros(4)
#	plane_bc1[:3] = np.cross(unit_b_vec, unit_c_vec) #b x c, such taht vector extends along a axis.
#
#	plane_bc2 = np.zeros(4)
#	plane_bc2[:3] = -1 * plane_bc1[:3] 
#	plane_bc2[3] = np.dot(plane_bc2[:3], unit_a_vec)
#
#
#	#Compile all planes into a list. 
#	
#	planes = [plane_ab1, plane_ab2, plane_ac2, plane_ac1, plane_bc1, plane_bc2]
#
#	print(planes)



	boundary_planes[:3, :3] = basis_planes
	boundary_planes[3:, :3] = -1 * basis_planes
	boundary_planes[3:, 3] = nonzero_dot_products

    
	#Now, we must write the lines for the packmol file to read.

	boundary_planes[:3, :3] = basis_planes
	boundary_planes[3:, :3] = -1 * basis_planes
	boundary_planes[3:, 3] = nonzero_dot_products

    

	lines = []

	#Add tolerance. This is distance between different materials (not elements! materials).
	lines.append("tolerance 2.0" + "\n")

	formula = oriented_unit_cell.formula

	#Strip all spaces when putting formula in output file name...
	output_file_name = "randpack_" + "".join(formula.split())

	lines.append("output " + output_file_name + ".xyz" + "\n")

	lines.append("filetype xyz" + "\n")



	#Next, we get the structure that we will be randomly packing.
	for el in el_dic:


		#First, write an XYZ file for that element, with coordinates. This code is coped directly from mpmorph.
		
		with open(el + ".xyz", "w") as f:
			f.write("1\ncomment\n" + el + " 0.0 0.0 0.0\n")


		#Then, specify that XYZ file, and the number of atoms that we need.
	
		lines.append("structure " + el + ".xyz" + "\n") 

		#Leading spaces must be here.

		lines.append(" number " + str(int(nlayers * el_dic.get(el))) + "\n" )



		#Pack into the desired region, defined by our 6 planes.. Units are PROBABLY in angstroms.

		for plane in boundary_planes:
			#Use list comprehension to convert plane into strings.
			str_plane = [str(num) for num in plane]
			
			#put it all together.

			line = " over plane " + " ".join(str_plane) + "\n"
			
			lines.append(line)
							 


		lines.append("end structure" + "\n")


	#Write the file as output.
	packmol_filename = "packmol" + "".join(formula.split()) + ".input"
	
	with open(packmol_filename, "w") as f:
		f.writelines(lines)

	
	#Next, run Packmol on the file.
	packmol_path = os.environ["PACKMOL_PATH"]

	try:
		os.system(packmol_path + "< " + "packmol" + "".join(formula.split()) + ".input")
	except:
		raise OSError("Packmol cannot be found!")

	#Then, read in the output file that Packmol just gave (it'll be in XYZ format) 
	randpack_dict = xyz_to_dict(output_file_name + ".xyz")


	#Create a lattice matrix using the vectors we specified earlier....
	lattice = [unit_a_vec, unit_b_vec, scaled_c_vec]	



	#convert dictionary + lattice to Pymatgen structure.
	randpack_struct = get_structure(randpack_dict, lattice)


	#Clean up extraneous files.
	if cleanup != True: 
	
		#Write as poscar...for debugging purposes.
		randpack_struct.to("poscar", output_file_name + "parallelogram_poscar.vasp")	
	
	else:
		os.system("rm " + packmol_filename)
	
		os.system("rm " + output_file_name + ".xyz")
	
		for el in el_dic:
			os.system("rm " + el + ".xyz")

	return randpack_struct
 

def xyz_to_dict(filename): #COPIED DIRECTLY FROM MPMORPH
	"""
	This is a generic xyz to dictionary convertor.
	Used to get the structure from packmol output.
	"""
	with open(filename, 'r') as f:
		lines = f.readlines()
		N = int(lines[0].rstrip('\n'))
		el_dict = {}
		for line in lines[2:]:
			l = line.rstrip('\n').split()
			if l[0] in el_dict:
				el_dict[l[0]].append([float(i) for i in l[1:]])
			else:
				el_dict[l[0]] = [[float(i) for i in l[1:]]]
	if N != sum([len(x) for x in el_dict.values()]):
		raise ValueError("Inconsistent number of atoms")
	el_dict = OrderedDict(el_dict)
	return el_dict

def get_structure(el_dict, lattice): #COPIED DIRECTLY FROM MPMORPH
	"""
	Args:
	    el_dict (dict): coordinates of atoms for each element type
	    e.g. {'V': [[4.969925, 8.409291, 5.462153], [9.338829, 9.638388, 9.179811], ...]
		  'Li': [[5.244308, 8.918049, 1.014577], [2.832759, 3.605796, 2.330589], ...]}
	    lattice (list): is the lattice in the form of [[x1,x2,x3],[y1,y2,y3],[z1,z2,z3]]
	Returns: pymatgen Structure
	"""
	species = []
	coords = []
	for el in el_dict.keys():
		for atom in el_dict[el]:
			species.append(el)
			coords.append(atom)
	return Structure(lattice, species, coords, coords_are_cartesian=True)

if __name__ == "__main__":
	print("I'm a dummy script!")
