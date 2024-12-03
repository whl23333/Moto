SIMPLER_ROOT="${PROJECT_ROOT}/../SimplerEnv"
gpu_id=0
# URDF variations
declare -a urdf_version_arr=(None "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" "recolor_cabinet_visual_matching_1")

# Move Near
EvalMoveNear() {
env_name=MoveNearGoogleBakedTexInScene-v0
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=${SIMPLER_ROOT}/ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png

for urdf_version in "${urdf_version_arr[@]}";
do CUDA_VISIBLE_DEVICES=${gpu_id} python -u evaluate_simpler.py --policy-model moto-gpt --moto_gpt_path ${MOTO_GPT_PATH} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
  --additional-env-build-kwargs urdf_version=${urdf_version} \
  --additional-env-save-tags baked_except_bpb_orange \
  --test_chunk_size ${TEST_CHUNK_SIZE} \
  --mask_latent_motion_probability ${MLMP} \
  --logging-dir ${EVAL_DIR};
done

echo "Done!  EvalMoveNear ${EVAL_DIR}"
}


# Pick Coke Can
EvalPickCokeCan() {
declare -a coke_can_options_arr=("laid_vertically=True" "lr_switch=True" "upright=True")
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=${SIMPLER_ROOT}/ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png

for urdf_version in "${urdf_version_arr[@]}";
do for coke_can_option in "${coke_can_options_arr[@]}";
do CUDA_VISIBLE_DEVICES=${gpu_id} python -u evaluate_simpler.py --policy-model moto-gpt --moto_gpt_path ${MOTO_GPT_PATH} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs ${coke_can_option} urdf_version=${urdf_version} \
  --test_chunk_size ${TEST_CHUNK_SIZE} \
  --mask_latent_motion_probability ${MLMP} \
  --logging-dir ${EVAL_DIR};
done
done

echo "Done!  EvalPickCokeCan ${EVAL_DIR}"
}



# Open / Close Drawer

EvalOverlay() {
EXTRA_ARGS="--enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True urdf_version=${urdf_version} \
--test_chunk_size ${TEST_CHUNK_SIZE} \
--mask_latent_motion_probability ${MLMP} \
--logging-dir ${EVAL_DIR}"

echo ${EXTRA_ARGS}

# A0
python -u evaluate_simpler.py --policy-model moto-gpt --moto_gpt_path ${MOTO_GPT_PATH} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.644 0.644 1 --robot-init-y -0.179 -0.179 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.03 -0.03 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ${SIMPLER_ROOT}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png \
  ${EXTRA_ARGS}

# A1
python -u evaluate_simpler.py --policy-model moto-gpt --moto_gpt_path ${MOTO_GPT_PATH} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.765 0.765 1 --robot-init-y -0.182 -0.182 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.02 -0.02 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ${SIMPLER_ROOT}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a1.png \
  ${EXTRA_ARGS}

# A2
python -u evaluate_simpler.py --policy-model moto-gpt --moto_gpt_path ${MOTO_GPT_PATH} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.889 0.889 1 --robot-init-y -0.203 -0.203 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.06 -0.06 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ${SIMPLER_ROOT}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a2.png \
  ${EXTRA_ARGS}

# B0
python -u evaluate_simpler.py --policy-model moto-gpt --moto_gpt_path ${MOTO_GPT_PATH} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.652 0.652 1 --robot-init-y 0.009 0.009 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ${SIMPLER_ROOT}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png \
  ${EXTRA_ARGS}

# B1
python -u evaluate_simpler.py --policy-model moto-gpt --moto_gpt_path ${MOTO_GPT_PATH} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.752 0.752 1 --robot-init-y 0.009 0.009 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ${SIMPLER_ROOT}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b1.png \
  ${EXTRA_ARGS}

# B2
python -u evaluate_simpler.py --policy-model moto-gpt --moto_gpt_path ${MOTO_GPT_PATH} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.851 0.851 1 --robot-init-y 0.035 0.035 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ${SIMPLER_ROOT}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b2.png \
  ${EXTRA_ARGS}

# C0
python -u evaluate_simpler.py --policy-model moto-gpt --moto_gpt_path ${MOTO_GPT_PATH} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.665 0.665 1 --robot-init-y 0.224 0.224 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ${SIMPLER_ROOT}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png \
  ${EXTRA_ARGS}

# C1
python -u evaluate_simpler.py --policy-model moto-gpt --moto_gpt_path ${MOTO_GPT_PATH} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.765 0.765 1 --robot-init-y 0.222 0.222 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.025 -0.025 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ${SIMPLER_ROOT}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c1.png \
  ${EXTRA_ARGS}

# C2
python -u evaluate_simpler.py --policy-model moto-gpt --moto_gpt_path ${MOTO_GPT_PATH} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.865 0.865 1 --robot-init-y 0.222 0.222 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.025 -0.025 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ${SIMPLER_ROOT}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c2.png \
  ${EXTRA_ARGS}
}



EvalDrawer() {
declare -a env_names=(
OpenTopDrawerCustomInScene-v0
OpenMiddleDrawerCustomInScene-v0
OpenBottomDrawerCustomInScene-v0
CloseTopDrawerCustomInScene-v0
CloseMiddleDrawerCustomInScene-v0
CloseBottomDrawerCustomInScene-v0
)

for urdf_version in "${urdf_version_arr[@]}"; do
for env_name in "${env_names[@]}"; do
  EvalOverlay
done
done

echo "Done! EvalDrawer ${EVAL_DIR}"
}


EvalGoogleRobot() {
cd ${PROJECT_ROOT}/moto_gpt/evaluation/robot_manipulation_benchmarks/simpler

EvalDrawer
EvalPickCokeCan
EvalMoveNear

python -u compute_metrics.py --eval_dir ${EVAL_DIR}
}



MLMP=1.0
TEST_CHUNK_SIZE=5
MOTO_GPT_PATH="${PROJECT_ROOT}/moto_gpt/checkpoints/moto_gpt_finetuned_on_rt1"
EVAL_DIR="${PROJECT_ROOT}/moto_gpt/evaluation/robot_manipulation_benchmarks/simpler/eval_results"
EvalGoogleRobot


<<COMMENT
conda activate moto_for_simpler
export PROJECT_ROOT=[your path to Moto project]
# ps aux | grep 'evaluate_moto_gpt_in_simpler' | awk '{print $2}' | xargs kill -9
# ps aux | grep 'evaluate_simpler' | awk '{print $2}' | xargs kill -9
cd ${PROJECT_ROOT}/scripts
nohup bash evaluate_moto_gpt_in_simpler.sh > evaluate_moto_gpt_in_simpler.log 2>&1 &
tail -f evaluate_moto_gpt_in_simpler.log
COMMENT