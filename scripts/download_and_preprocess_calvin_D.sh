echo "***********************************************************"
if [ ! -d ${OUTPUT_ROOT} ]; then
   mkdir -p ${OUTPUT_ROOT}
fi
echo "hi"
cd ${OUTPUT_ROOT}
dataset_path="${OUTPUT_ROOT}/task_D_D"
if [ -d ${dataset_path} ]; then
   	echo "${dataset_path} exists."
else
   echo "Downloading CALVIN task_D_D ..."
   wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
   unzip task_D_D.zip
   echo "saved folder: ${OUTPUT_ROOT}/task_D_D"
fi

echo "***********************************************************"
lmdb_path="${OUTPUT_ROOT}/lmdb_datasets/task_D_D/"
cd ${PROJECT_ROOT}/data_preprocessing
if [ -d ${lmdb_path} ]; then
   echo "${lmdb_path} exists."
else
   echo "Outputting CALVIN task_D_D lmdb dataset to ${lmdb_path} ..."
   python3 -u calvin_to_lmdb.py \
   --input_dir ${OUTPUT_ROOT}/task_D_D \
   --output_dir ${lmdb_path}
fi
