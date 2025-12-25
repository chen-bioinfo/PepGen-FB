import numpy as np
import pandas as pd
import joblib
import os
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def aac_comp(file, out):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    df1 = pd.DataFrame(file, columns=["Seq"])
    dd = []
    for j in df1['Seq']:
        cc = []
        for i in std:
            count = 0
            for k in j:
                temp1 = k
                if temp1 == i:
                    count += 1
                composition = (count/len(j))*100
            cc.append(composition)
        dd.append(cc)
    df2 = pd.DataFrame(dd)
    head = []
    for mm in std:
        head.append('AAC_'+mm)
    df2.columns = head
    df2.to_csv(out, index=None, header=False)

def dpc_comp(file, out, q=1):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    df1 = pd.DataFrame(file, columns=["Seq"])
    zz = df1.Seq
    dd = []
    for i in range(0, len(zz)):
        cc = []
        for j in std:
            for k in std:
                count = 0
                temp = j+k
                for m3 in range(0, len(zz[i])-q):
                    b = zz[i][m3:m3+q+1:q]
                    b = b.upper()
                    if b == temp:
                        count += 1
                    composition = (count/(len(zz[i])-(q)))*100
                cc.append(composition)
        dd.append(cc)
    df3 = pd.DataFrame(dd)
    head = []
    for s in std:
        for u in std:
            head.append("DPC"+str(q)+"_"+s+u)
    df3.columns = head
    df3.to_csv(out, index=None, header=False)

def prediction(inputfile1, inputfile2, model, out):
    clf = joblib.load(model)
    data_test1 = np.loadtxt(inputfile1, delimiter=',')
    data_test2 = np.loadtxt(inputfile2, delimiter=',')
    data_test3 = np.concatenate([data_test1, data_test2], axis=1)
    X_test = data_test3
    y_p_score1 = clf.predict_proba(X_test)
    return y_p_score1[:, 1]  # 返回正类（毒性）的概率

def predict_toxin_scores(clean_sequences):
    """
    计算输入序列的 ML Score（毒性预测概率）。
    
    参数:
        clean_sequences (list): 包含氨基酸序列的列表，例如 ["ACDEFG", "GHIJKL"]
    
    返回:
        list: 每个序列的 ML Score（浮点数列表）
    """
    # 验证输入序列
    valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
    clean_sequences = [re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq.upper()) for seq in clean_sequences]
    clean_sequences = [seq for seq in clean_sequences if seq]
    if not clean_sequences:
        print("错误：没有有效序列")
        return []

    # 生成临时文件
    aac_file = "temp_seq.aac"
    dpc_file = "temp_seq.dpc"
    model_file = "./toxinpred3/model/toxinpred3.0_model.pkl"

    # 计算 AAC 和 DPC 特征
    aac_comp(clean_sequences, aac_file)
    dpc_comp(clean_sequences, dpc_file)

    # 清理 CSV 文件中的末尾逗号
    os.system(f"perl -pi -e 's/,$//g' {aac_file}")
    os.system(f"perl -pi -e 's/,$//g' {dpc_file}")

    # 进行预测
    ml_scores = prediction(aac_file, dpc_file, model_file, None)

    # 清理临时文件
    for f in [aac_file, dpc_file]:
        if os.path.exists(f):
            os.remove(f)

    # 返回 ML Score 列表
    return ml_scores.tolist()

# 示例用法
if __name__ == "__main__":
    # 测试序列
    test_sequences = ["ACDEFGHIKLMNPQRSTVWY", "GHIJKL"]
    scores = predict_toxin_scores(test_sequences)
    print("序列及其 ML Score：")
    for seq, score in zip(test_sequences, scores):
        print(f"序列: {seq}, ML Score: {score:.3f}")