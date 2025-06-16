#!/bin/bash

# 查找并解压所有.zip文件
find /group/ycyang/yyang-infobai/rlbench_test -name "*.zip" -exec unzip -o {} -d /group/ycyang/yyang-infobai/rlbench_test \;

echo "All zip files have been extracted."