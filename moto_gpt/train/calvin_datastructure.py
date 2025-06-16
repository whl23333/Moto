import numpy as np

npy_file = np.load("/group/ycyang/yyang-infobai/task_ABC_D/validation/lang_annotations/auto_lang_ann.npy", allow_pickle=True)

npy_file = npy_file.item()
print(npy_file.keys())
print(npy_file['language'].keys())
print("len(npy_file['language']['ann']):", len(npy_file['language']['ann']))
print("npy_file['language']['ann']:", npy_file['language']['ann'])
print("len(npy_file['language']['task']):", len(npy_file['language']['task']))
print("npy_file['language']['task']:", npy_file['language']['task'])
