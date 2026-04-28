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
    return y_p_score1[:, 1]  

def predict_toxin_scores(clean_sequences):

    valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
    clean_sequences = [re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq.upper()) for seq in clean_sequences]
    clean_sequences = [seq for seq in clean_sequences if seq]
    if not clean_sequences:
        print("错误：没有有效序列")
        return []

    aac_file = "temp_seq.aac"
    dpc_file = "temp_seq.dpc"
    model_file = "./toxinpred3/model/toxinpred3.0_model.pkl"

    aac_comp(clean_sequences, aac_file)
    dpc_comp(clean_sequences, dpc_file)

    os.system(f"perl -pi -e 's/,$//g' {aac_file}")
    os.system(f"perl -pi -e 's/,$//g' {dpc_file}")

    ml_scores = prediction(aac_file, dpc_file, model_file, None)

    for f in [aac_file, dpc_file]:
        if os.path.exists(f):
            os.remove(f)

    return ml_scores.tolist()


if __name__ == "__main__":

    test_sequences = ["ACDEFGHIKLMNPQRSTVWY", "GHIJKL"]
    scores = predict_toxin_scores(test_sequences)
    print("sequence and ML Score：")
    for seq, score in zip(test_sequences, scores):
        print(f"sequence: {seq}, ML Score: {score:.3f}")
