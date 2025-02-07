export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_HOME="/group/40007/milkcychen/cq10/cache/torch"

cd ${PROJECT_ROOT}/moto_gpt/train
accelerate launch --main_process_port 29501 train_moto_gpt.py --config_path "${PROJECT_ROOT}/moto_gpt/configs/train/${CONFIG_NAME}.yaml"

<<COMMENT
conda activate moto
export PROJECT_ROOT=[your path to Moto project]
export CONFIG_NAME="data_rtx-model_actPredFalse_motionPredTrue_visionMaeLarge_seq2_chunk3_maskProb0.5-train_lr0.0001_bs512-aug_shiftTrue_resizedCropFalse"
# ps aux | grep ${CONFIG_NAME} | awk '{print $2}' | xargs kill -9
cd ${PROJECT_ROOT}/scripts/
nohup bash pretrain_moto_gpt_on_oxe.sh > pretrain_moto_gpt_on_oxe.log 2>&1 &
tail -f pretrain_moto_gpt_on_oxe.log
COMMENT