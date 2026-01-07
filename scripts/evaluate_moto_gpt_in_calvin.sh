export CUDA_VISIBLE_DEVICES=2
export CALVIN_ROOT=${PROJECT_ROOT}/../calvin/
export MESA_GL_VERSION_OVERRIDE=3.3



EvalCALVIN() {
# ps aux | grep "evaluate_calvin" | awk '{print $2}' | xargs kill -9
cd ${PROJECT_ROOT}/moto_gpt/evaluation/robot_manipulation_benchmarks/calvin
accelerate launch --main_process_port=29504 evaluate_calvin.py \
    --moto_gpt_path ${MOTO_GPT_PATH} \
    --test_chunk_size ${TEST_CHUNK_SIZE} \
    --mask_latent_motion_probability ${MLMP} \
    --eval_dir ${EVAL_DIR}
echo "Done! EvalCALVIN ${EVAL_DIR}"
}

MLMP=1.0
TEST_CHUNK_SIZE=8
MOTO_GPT_PATH="/home/hlwang/Moto/results/moto_gpt/provided_moto_gpt_3d_simple_sngl_dcdr_16/provided_pretrain_moto_gpt__3d_simple_sngl_dcdr_16epoch/saved_epoch_13_step_217698"
EVAL_DIR="${PROJECT_ROOT}/moto_gpt/evaluation/robot_manipulation_benchmarks/calvin/eval_results/$(basename $(dirname $(dirname ${MOTO_GPT_PATH})))_$(basename $(dirname ${MOTO_GPT_PATH}))_$(basename ${MOTO_GPT_PATH})_MLMP${MLMP}_TCS${TEST_CHUNK_SIZE}"
EvalCALVIN



<<COMMENT
conda activate moto_for_calvin
export PROJECT_ROOT=[your path to Moto project]
# ps aux | grep 'evaluate_moto_gpt_in_calvin' | awk '{print $2}' | xargs kill -9
# ps aux | grep 'evaluate_calvin' | awk '{print $2}' | xargs kill -9
cd ${PROJECT_ROOT}/scripts
nohup bash evaluate_moto_gpt_in_calvin.sh > evaluate_moto_gpt_finetuned_on_paired_19.log 2>&1 &
tail -f evaluate_moto_gpt_finetuned_on_paired_19.log