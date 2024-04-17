data_path = "/path/to/imagenet/"
checkpoint_path = "/path/to/checkpoint/"

python main.py --eval \
    --model deit_tiny \
    --dataset IMNET \
    --data_path $data_path \
    --resume /path/to/checkpoint/ \
    --use_mctf True \
    --mctf_type [16,0,0,1,1,1,20,40,1,1,0] \
    --task_type [1,0,0,0] \
    --r_evals [16] \
    --activate_layer [1,2,3,4,5,6,7,8,9,10,11]