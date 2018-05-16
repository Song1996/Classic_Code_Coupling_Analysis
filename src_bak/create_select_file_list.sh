# $1 为项目名 生成的file list由上到下日期越来越旧
cd ./projects/$1
pwd
git log --pretty=format:"%H" --name-only > ../file_list/$1_file_list.txt
cd ../../
pwd
python select_file_list.py $1
