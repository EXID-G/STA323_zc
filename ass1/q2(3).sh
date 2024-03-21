#!/bin/bash

# mkdir "output/output_data/q2(3)"

# 定义输出文件路径
output_file="output/output_data/q2(3)_sh.csv"

# 如果输出文件已存在，则删除它
if [ -f $output_file ]; then
    rm $output_file
fi

# 指定需要分割的大文件
large_file="data/Q2_data/SRR12326775_1_Light_Bulk.csv"

# 这个命令首先会使用 tail -n +3 命令从原始文件的第三行开始，将所有行输出到一个名为 temp.csv 的临时文件中。然后，它会使用 split 命令将这个临时文件分割为8个部分，每个部分的文件名都以 chunk 开头。最后，它删除临时文件 temp.csv
tail -n +3 $large_file > temp.csv
split -n l/8 -d --additional-suffix=.csv temp.csv chunk
rm temp.csv


chunk_list=$(ls chunk*.csv)
for chunk in $chunk_list; do
    file_name=${chunk:5}
    awk -F, 'NR>0 {if (length($47)>=10 && length($47)<=100) print $14","$37","$41","$47}' $chunk > temp${file_name}.csv &
done

wait

# 提取第一个文件的标题行(第二行)，并写入最终的 CSV 文件
sed -n '2p' $large_file | awk -F, '{print $14","$37","$41","$47}' > $output_file

# 将临时文件的内容追加到最终的 CSV 文件
cat temp*.csv >> $output_file

# 删除临时文件和文件块
rm chunk*.csv temp*.csv

# 显示输出文件的行数、字数和字符数
wc $output_file
