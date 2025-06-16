
exp=moto_gpt

tasks=(
    close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap place_cups
)
data_dir=/group/ycyang/yyang-infobai/rlbench_test/
num_episodes=100
use_instruction=1
max_tries=2
verbose=1
interpolation_length=2
cameras="left_shoulder,right_shoulder,wrist,front"
seed=0
checkpoint=/home/yyang-infobai/Moto/moto_gpt/outputs/moto_gpt_trained_on_rlbench/not_paired/30000steps/saved_epoch_1_step_30000/
quaternion_format=wxyz  # IMPORTANT: change this to be the same as the training script IF you're not using our checkpoint

num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python evaluate_motogpt.py \
    --tasks ${tasks[$i]} \
    --checkpoint $checkpoint \
    --num_history 3 \
    --cameras $cameras \
    --verbose $verbose \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 1 \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/$exp/seed$seed/${tasks[$i]}.json  \
    --instructions /group/ycyang/yyang-infobai/instructions/peract/instructions.pkl \
    --variations {0..60} \
    --max_tries $max_tries \
    --max_steps 25 \
    --seed $seed \
    --interpolation_length $interpolation_length \
    --dense_interpolation 1 \
    --moto_gpt_config_path configs/moto_gpt.yaml \
    --image_size 224,224
done