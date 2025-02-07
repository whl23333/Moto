export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${PROJECT_ROOT}/moto_gpt/train
accelerate launch --main_process_port 29501 train_moto_gpt.py --config_path "${PROJECT_ROOT}/moto_gpt/configs/train/${CONFIG_NAME}.yaml"


<<COMMENT
conda activate moto
export PROJECT_ROOT=[your path to Moto project]
export CONFIG_NAME="data_rt1-model_actPredTrue_motionPredTrue_visionMaeLarge_seq2_chunk3_maskProb0.5-train_lr0.0001_bs512-aug_shiftTrue_resizedCropFalse-resume_from_predLatentOnly_oxe_Epoch10"
# ps aux | grep ${CONFIG_NAME} | awk '{print $2}' | xargs kill -9
cd ${PROJECT_ROOT}/scripts/
nohup bash finetune_moto_gpt_on_rt1.sh > finetune_moto_gpt_on_rt1.log 2>&1 &
tail -f finetune_moto_gpt_on_rt1.log
COMMENT