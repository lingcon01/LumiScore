#!/bin/bash
#SBATCH -J 1qkt               #  本次作业的名称
#SBATCH -o ./predict.out                  #  %j将被作业号替代
#SBATCH -e ./predict.error                  #  错误日志输出
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
##SBATCH -w  gpu12 # If you want to specify a computing node, you can write its name here and remove the first #


source ~/miniconda3/bin/activate
conda activate my3
module load openbabel
module unload anaconda3/2021.05
# module load cuda/11.1

#输入要执行的命令

# python -u suscore.py -p /home/suqun/tmp/GMP/data/derivate_new/Inputs_for_FEP/MCL1/complex/61/protein_pocket_5.0.pdb -l /home/suqun/data/MCL1_screen/MCL1/total5_unique2.sdf -e gt -m /home/suqun/tmp/GMP/pretrain/train_and_test/compre/SuScore_local01_rmse126.pth 

# python -u pretrain_predict.py -p /home/suqun/tmp/GMP/data/derivate_new/Inputs_for_FEP/MCL1/complex/61/protein_pocket_5.0.pdb -l /home/suqun/data/MCL1_screen/MCL1/fragment_2200_unique.sdf -e gt -m /home/suqun/tmp/GMP/pretrain/EGMDN/ET_pretrain.pth


# python -u suscore.py -p /home/suqun/tmp/GMP/data/derivate_new/Inputs_for_FEP/MCL1/complex/61/protein_pocket_5.0.pdb -l /home/suqun/data/MCL1_screen/MCL1/total5_unique2.sdf -e gt -m /home/suqun/tmp/GMP/pretrain/train_and_test/my_test/SuScore_local1_88_derivate.pth

python -u suscore.py -p /home/suqun/tmp/GMP/data/derivate_new/Inputs_for_FEP/MCL1/complex/61/protein_pocket_5.0.pdb -l /home/suqun/data/MCL1_screen/MCL1/fragment_2200_unique.sdf -e gt -m /home/suqun/tmp/GMP/pretrain/train_and_test/SuScore/SuScore_derivate_pr_mdn2_v2_local.pth
