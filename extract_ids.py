import re

# 读取文件内容
with open('/home/yyang-infobai/Moto/google_id.txt', 'r') as file:
    content = file.read()

# 使用正则表达式提取所有ID
ids = re.findall(r'file/d/([a-zA-Z0-9_-]+)', content)

# 生成Bash脚本内容
bash_script_content = "#!/bin/bash\n\n"
for file_id in ids:
    bash_script_content += f"gdown 'https://drive.google.com/uc?id={file_id}'\n"

# 将Bash脚本内容写入文件
with open('/home/yyang-infobai/Moto/download_files.sh', 'w') as bash_file:
    bash_file.write(bash_script_content)

print("Bash script generated successfully.")