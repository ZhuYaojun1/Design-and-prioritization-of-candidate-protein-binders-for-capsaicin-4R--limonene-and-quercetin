import os
import shutil
import re

# ==================== 配置区 ====================
# 原始结果汇总文件路径
summary_file = r"C:\Users\Lenovo\Desktop\small_molecule_design\4_RoseTTAFold3\9IR\06_results_summary.txt"
# 注意：您提供的文件名是 "06_results_summary_CN - 副本.txt"，请根据实际情况修改上面的路径

# RF3 文件所在目录
rf3_dir = r"C:\Users\Lenovo\Desktop\small_molecule_design\4_RoseTTAFold3\9IR\out_9IR_WINDOW"
# RFD3 文件所在目录
rfd3_dir = r"C:\Users\Lenovo\Desktop\small_molecule_design\4_RoseTTAFold3\9IR\Skeletal_structure"

# 输出目录（可以自行修改）
output_rf3 = r"C:\Users\Lenovo\Desktop\small_molecule_design\4_RoseTTAFold3\9IR\RF3_cifs"
output_rfd3 = r"C:\Users\Lenovo\Desktop\small_molecule_design\4_RoseTTAFold3\9IR\RFD3_cifs"

# ==================== 函数定义 ====================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建目录: {path}")

def parse_entries(filepath):
    """读取文件，返回所有条目列表（如 buried_0_model_0_0）"""
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 按逗号分割，并去除每个条目可能的空格
            parts = [p.strip() for p in line.split(',') if p.strip()]
            entries.extend(parts)
    return entries

def extract_parts(entry):
    """从 entry 如 buried_0_model_0_0 中提取 a, b, c"""
    # 格式：buried_{a}_model_{b}_{c}
    match = re.match(r'buried_(\d+)_model_(\d+)_(\d+)', entry)
    if not match:
        raise ValueError(f"条目格式不正确: {entry}")
    a, b, c = match.groups()
    return a, b, c

def copy_files(entries):
    # 创建输出目录
    ensure_dir(output_rf3)
    ensure_dir(output_rfd3)

    for entry in entries:
        try:
            a, b, c = extract_parts(entry)
            sample_num = int(c) + 1  # 根据例子 0 -> sample1

            # RF3 文件名和完整路径
            rf3_filename = f"buried_{a}_model_{b}_sample{sample_num}_model.cif"
            rf3_src = os.path.join(rf3_dir, rf3_filename)
            rf3_dst = os.path.join(output_rf3, rf3_filename)

            # RFD3 文件名和完整路径
            rfd3_filename = f"buried_{a}_model_{b}.cif"
            rfd3_src = os.path.join(rfd3_dir, rfd3_filename)
            rfd3_dst = os.path.join(output_rfd3, rfd3_filename)

            # 复制 RF3 文件
            if os.path.exists(rf3_src):
                shutil.copy2(rf3_src, rf3_dst)
                print(f"复制 RF3: {rf3_filename}")
            else:
                print(f"警告: RF3 文件不存在 - {rf3_src}")

            # 复制 RFD3 文件
            if os.path.exists(rfd3_src):
                shutil.copy2(rfd3_src, rfd3_dst)
                print(f"复制 RFD3: {rfd3_filename}")
            else:
                print(f"警告: RFD3 文件不存在 - {rfd3_src}")

        except Exception as e:
            print(f"处理条目 {entry} 时出错: {e}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 检查汇总文件是否存在
    if not os.path.exists(summary_file):
        print(f"错误: 找不到汇总文件 {summary_file}")
        print("请确认文件路径，并将上面的 summary_file 变量修改为正确的路径。")
    else:
        entries = parse_entries(summary_file)
        print(f"共找到 {len(entries)} 个条目")
        copy_files(entries)
        print("处理完成！")