#A Basic script to extract interfacial energies from a calculation.

import os
import yaml

from atomate.vasp.database import VaspCalcDb

def auto_load_db():

	#Get DB file from environment variable (assuming standard Atomate installation).
	#IS THERE A WAY TO GET THIS DIRECTLY? HAS ATOMATE ALREADY DONE THIS?
	cfg_path = os.environ.get("FW_CONFIG_FILE")

	with open(cfg_path, "r") as stream:
		yaml_dic = yaml.safe_load(stream)

	cfg_folder = yaml_dic.get("CONFIG_FILE_DIR")

	db_path = cfg_folder + "/db.json"


	#Load in Database
	db = VaspCalcDb.from_db_file(db_path) 

	return db


def calculate_interfacial_energy(crystalline_relax_id, amorph_relax_id, interfacial_relax_id):

	#First, get ground state 

	atomate_db = auto_load_db()

	#Now, we must find three energies.
	#1. Energies from interfacial slab relaxation.
	#2 Energy from bulk crystalline relaxation.
	#3. Energy from bulk amorphous relaxation. 



	#for amorphous vasp runs, the energy relaxation task name is "snap_0_static". Instead, we must search based on task_id, which is unique.

	task_ids = [crystalline_relax_id, amorph_relax_id, interfacial_relax_id)

	calc_dict = atomate_db.collection.find_one({"task_id": task_id})
	output = calc_dict.get("output")

	energy = calc_dict.get("energy_per_atom")







	#Final Calculation methods will eventually go here...
