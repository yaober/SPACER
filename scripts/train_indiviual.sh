#!/bin/bash
#SBATCH --job-name train_indivual

# Name of the SLURM partition that this job should run on.
#SBATCH -p 512GB    # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%_train_indivual.out
#SBATCH -e job_%_train_indivuals.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

module load python/3.8.x-anaconda
conda activate spatial_tcr
cd /project/DPDS/Wang_lab/s439765/spatial_tcr/MIL_TCR
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanOvarianCancer/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanOvarianCancer' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanOvarianCancerWholeTranscriptome/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanOvarianCancerWholeTranscriptome' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanColorectalCancerWholeTranscriptome/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanColorectalCancerWholeTranscriptome' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanOvarianCancerFFPE/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanOvarianCancerFFPE' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanLungCancer/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanLungCancer' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanBreastCancerDuctalCarcinoma/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanBreastCancerDuctalCarcinoma' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanProstateCancerAdenocarcinomaFFPE/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanProstateCancerAdenocarcinomaFFPE' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanBreastCancerA2/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanBreastCancerA2' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/InvasiveDuctalCarcinoma/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/InvasiveDuctalCarcinoma' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanBreastCancer/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanBreastCancer' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanIntestineCancerFFPE/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanIntestineCancerFFPE' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanColorectalCancerFFPE/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanColorectalCancerFFPE' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanMelanomaFFPE/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanMelanomaFFPE' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanBreastCancerWholeTranscriptome/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanBreastCancerWholeTranscriptome' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanCervicalCancerFFPE/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanCervicalCancerFFPE' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanProstateCancerAcinarFFPE/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanProstateCancerAcinarFFPE' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/Visium/HumanProstateCancerFFPE/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanProstateCancerFFPE' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/VisiumHD/HumanColorectalCancer/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanColorectalCancer' --resolution high --radius 150
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/VisiumHD/HumanLungCancer/T_cell.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/HumanLungCancerHD' --resolution high --radius 150
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/s439765/spatial_tcr/SpatialVDJ/SpatialVDJ_forZenodo/data/breast_cancer/TumE2_preprocessed.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/TumE2_preprocessed' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/s439765/spatial_tcr/SpatialVDJ/SpatialVDJ_forZenodo/data/breast_cancer/TumC2_preprocessed.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/TumC2_preprocessed' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/s439765/spatial_tcr/spatial_TCR_scRNA/p16_preprocessed.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/p16_preprocessed' --resolution low --radius 300
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/s439765/spatial_tcr/slide_tag/slide_tag_melanoma_processed.h5ad' --reference_gene_path 'data/human.csv' --pretrained_gene_path 'data/tumor_antigen_8000.csv'  --output_dir './train_indiviual/slide_tag_melanoma_processed' --resolution high --radius 300
