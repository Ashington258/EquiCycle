###
 # @Author: Ashington ashington258@proton.me
 # @Date: 2024-07-23 21:36:40
 # @LastEditors: Ashington ashington258@proton.me
 # @LastEditTime: 2024-07-23 21:36:46
 # @FilePath: \equicycle\overseer.bash
 # @Description: 请填写简介
 # 联系方式:921488837@qq.com
 # Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
### 
#!/bin/bash
echo "统计代码提交情况："
# 获取所有贡献者列表
authors=$(git log --format='%aN' | sort -u)
for author in $authors; do
    echo "----------------------------------------"
    echo "作者：$author"
    # 统计提交次数
    commit_count=$(git shortlog -s -n --author="$author" | awk '{print $1}')
    echo "提交次数：$commit_count"
    # 统计新增和删除行数
    line_stat=$(git log --numstat --pretty="%aN" --author="$author" | awk 'NF==3 {plus+=$1; minus+=$2} END {printf("%d %d", plus, minus)}')
    IFS=' ' read -ra stats <<< "$line_stat"
    echo "新增行数：${stats[0]}"
    echo "删除行数：${stats[1]}"
done