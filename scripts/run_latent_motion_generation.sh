LatentMotionGen() {
cd ${PROJECT_ROOT}/moto_gpt/evaluation/motion_prior_analysis
python -u latent_motion_generation.py \
--moto_gpt_path ${MOTO_GPT_PATH} \
--latent_motion_tokenizer_path ${LATENT_MOTION_TOKENIZER_PATH} \
--num_gen_frames ${NUM_GEN_FRAMES} \
--delta_t ${DELTA_T} \
--input_dir ${INPUT_DIR} \
--output_dir ${OUTPUT_DIR}

echo "Done!!! ${OUTPUT_DIR}"
}


# Open-X-Embodiment
MOTO_GPT_PATH="${PROJECT_ROOT}/moto_gpt/checkpoints/moto_gpt_pretrained_on_oxe"
LATENT_MOTION_TOKENIZER_PATH="${PROJECT_ROOT}/latent_motion_tokenizer/checkpoints/latent_motion_tokenizer_trained_on_oxe"
NUM_GEN_FRAMES=8
DELTA_T=3
INPUT_DIR="${PROJECT_ROOT}/moto_gpt/evaluation/motion_prior_analysis/sample_data/oxe"
OUTPUT_DIR="${PROJECT_ROOT}/moto_gpt/evaluation/motion_prior_analysis/output_motion_trajectories/oxe"
LatentMotionGen

# CALVIN
MOTO_GPT_PATH="${PROJECT_ROOT}/moto_gpt/checkpoints/moto_gpt_pretrained_on_calvin"
LATENT_MOTION_TOKENIZER_PATH="${PROJECT_ROOT}/latent_motion_tokenizer/checkpoints/latent_motion_tokenizer_trained_on_calvin"
NUM_GEN_FRAMES=8
DELTA_T=5
INPUT_DIR="${PROJECT_ROOT}/moto_gpt/evaluation/motion_prior_analysis/sample_data/calvin"
OUTPUT_DIR="${PROJECT_ROOT}/moto_gpt/evaluation/motion_prior_analysis/output_motion_trajectories/calvin"
LatentMotionGen

<<COMMENT
conda activate moto
export PROJECT_ROOT=[your path to Moto project]
# ps aux | grep 'run_latent_motion_generation' | awk '{print $2}' | xargs kill -9
# ps aux | grep 'latent_motion_generation' | awk '{print $2}' | xargs kill -9
cd ${PROJECT_ROOT}/scripts
nohup bash run_latent_motion_generation.sh > run_latent_motion_generation.log 2>&1 &
tail -f run_latent_motion_generation.log
COMMENT