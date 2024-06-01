#!/bin/bash

output_file="output/output_data/q2(2)_sh.csv"

# 如果输出文件已存在，则删除它
if [ -f $output_file ]; then
    rm $output_file
fi

# 获取 CSV 文件列表
csv_list=$(ls data/Q2_data/*.csv)

# 从每个 CSV 文件中读取内容（不包括头部），并追加到结果文件
for file in $csv_list
do
    # 使用 awk 提取需要的列，然后使用 grep 删除 cdr3_aa 长度小于 10 或大于 100 的行
    awk -F, 'NR>2 {if (length($47)>=10 && length($47)<=100) print $14","$37","$41","$47}' $file >> temp.csv
done

# 提取第一个文件的标题行(第二行)，并写入最终的 CSV 文件
sed -n '2p' $(echo $csv_list | awk '{print $1}') | awk -F, '{print $14","$37","$41","$47}' > $output_file

# 将临时文件的内容追加到最终的 CSV 文件，并删除临时文件
cat temp.csv >> $output_file
rm temp.csv

wc $output_file


