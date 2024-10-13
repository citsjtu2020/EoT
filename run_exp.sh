#! /bin/bash

# Runs the "345M" parameter model

pip install torch
pip install regex 
pip install transformers 
pip install pybind11  
pip install six 

pip install tensorboard
pip install rouge
pip install tensorflow==2.7.4
pip install pandas==1.3.2
pip install numpy==1.23.2


DATA_PATH=/root/data/huaqin/
pip install /root/data/huaqin/bleurt/.
python /root/code/huaqin/Reasoning/EoT-ENG/exp_for_eng.py --exp="7-CON_STRUCT" --start_id=50 --proc_num=6 --evo_model_name="qwen2" --is_fact=1 --is_truth=1 --strategy="MASK" --is_super=1


