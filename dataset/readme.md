Data features are calculated using PaDEL-Descriptor. 

You can use: java -jar ./PaDEL-Descriptor.jar -2d -fingerprint -dir(you dir) -file(you want path)

create_mol.py: Use this file to calculate the mol file required by PaDEL-Descriptor

mol_data_std:Standardize the calculated features, delete columns with too high consistency, and save them as numpy for easy training

func_nan: the calculated features may be missing and need to be processed

ECFP: calculated features of ECFP