export CUDA_VISIBLE_DEVICES=1
export CALVIN_ROOT=${PROJECT_ROOT}/../calvin/
export MESA_GL_VERSION_OVERRIDE=3.3



EvalCALVIN() {
# ps aux | grep "evaluate_calvin" | awk '{print $2}' | xargs kill -9
cd ${PROJECT_ROOT}/moto_gpt/evaluation/robot_manipulation_benchmarks/calvin
python evaluate_calvin.py \
    --moto_gpt_path ${MOTO_GPT_PATH} \
    --test_chunk_size ${TEST_CHUNK_SIZE} \
    --mask_latent_motion_probability ${MLMP} \
    --eval_dir ${EVAL_DIR}
echo "Done! EvalCALVIN ${EVAL_DIR}"
}

MLMP=1.0
TEST_CHUNK_SIZE=8
MOTO_GPT_PATH="${PROJECT_ROOT}/moto_gpt/outputs/moto_gpt_finetuned_on_calvin/paired_b32_gpu4_calvinABC_D_half_epoch18/pretrained_epoch_10/second_training/saved_epoch_12_step_100476"
EVAL_DIR="${PROJECT_ROOT}/moto_gpt/evaluation/robot_manipulation_benchmarks/calvin/eval_results/paired_b32_gpu4_calvinABC_D_half_epoch18/pretrained_epoch_10/second_training/saved_epoch_12_step_100476/eval2"
EvalCALVIN



<<COMMENT
conda activate moto_for_calvin
export PROJECT_ROOT=[your path to Moto project]
# ps aux | grep 'evaluate_moto_gpt_in_calvin' | awk '{print $2}' | xargs kill -9
# ps aux | grep 'evaluate_calvin' | awk '{print $2}' | xargs kill -9
cd ${PROJECT_ROOT}/scripts
nohup bash evaluate_moto_gpt_in_calvin.sh > evaluate_moto_gpt_halfABCD_12.log 2>&1 &
tail -f evaluate_moto_gpt_halfABCD_12.log