# MIL_TCR



(1) high res data

  tumor region label (namely cell type), 0, 1, 2

  T/B label, 0, 1 defined from cell type (T/B label=1 from cell type=2, T/B label=0 from cell type=1)

  good: if cell_types[i] == 0:

  good: in_circle = [idx for idx in in_circle if cell_types[idx] != 0]

  to add: remove the cell in the center of the bag. that cell should not go into the instances. it is cell that defines the bag

  the cell in the center of the bag could have the label of cell type=1 (tumor cell, negative case) or cell type=2 (T/B cell, positive case)

  to add: remove the cell among the instances that are not defined as cell type=1 (tumor cells). namely, only tumor cells are the instances/signal sending cells and should be kept
 
(2) low res data

  tumor region label (namely cell type), only 0, 1 

  T/B label, 0, 1 defined from gene signature

  good: if cell_types[i] == 0:

  good: in_circle = [idx for idx in in_circle if cell_types[idx] != 0]

  the cell in the center of the bag could have the label of T/B label=0 (tumor cell, negative case) or T/B label=1 (T/B cell, positive case)

 ine_tune_individual_pretrain2: pretrain model from 1st all huamn gebn 2nd 8000 tumor antigen gene, tune on all human gene
fine_tuned_model_10000:  pretrain model from 1st all huamn gene , tune on 80000tumor antigen gene, using fine_tune_model1_indiviual.ipynb/fine_tune_model1_all.py, fine_tuned_model_10000/all_data_human2antigen is 1st all huamn gebn 2nd 8000 tumor antigen gene
fine_tuned_model_by8000   pretrain model from tumor antigen gene 8000, bad 
pretrained_models: train seperately on all human geen
train_indiviual: rain seperately on all human geen