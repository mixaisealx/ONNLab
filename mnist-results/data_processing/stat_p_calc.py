import numpy as np
from scipy import stats

FOLDER_PATH = '../raw_data'

def get_learn_stat(nm, filename, fn2 = None):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                if nm:
                    acc,f1,stp,exp = line.rstrip().split('\t')[0:4]
                    data.append([float(acc), float(f1), float(stp), float(exp)])
                else:
                    acc,f1 = line.rstrip().split('\t')[0:2]
                    data.append([float(acc), float(f1)])
            except:
                pass
    if fn2 != None:
        with open(fn2, 'r') as f:
            for line in f:
                try:
                    if nm:
                        acc,f1,stp,exp = line.rstrip().split('\t')[0:4]
                        data.append([float(acc), float(f1), float(stp), float(exp)])
                    else:
                        acc,f1 = line.rstrip().split('\t')[0:2]
                        data.append([float(acc), float(f1)])
                except:
                    pass
    return np.array(data)

def get_resist_stat(filename, nm=True):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                if nm:
                    spl = line.rstrip().split('\t')
                    data.append([float(spl[5]), float(spl[8]), float(spl[6]), float(spl[9]), float(spl[0])])
                else:
                    spl = line.rstrip().split('\t')
                    data.append([float(spl[3]), float(spl[6]), float(spl[4]), float(spl[7]), float(spl[0])])
            except:
                pass
    return np.array(data)
    
lr_relu32 = get_learn_stat(False, FOLDER_PATH+'/exp_iReLU_ReLU_mnist_cmp2_relu.log', FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__ReLU.log')
lr_nmirelueq = get_learn_stat(True,FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_cmp1_nmiReLU-EQ.log', FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmiReLU_EQ.log')

rs_relu32 = get_resist_stat(FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__ReLU.log', False)
rs_nm0relujn = get_resist_stat(FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmReLU.log')
rs_nmirelujn = get_resist_stat(FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmiReLU_JN.log')
rs_nmirelueq = get_resist_stat(FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmiReLU_EQ.log')

def CalcUP(series1, series2, metric, s1,s2):
    u_stat, p_value = stats.mannwhitneyu(series1, series2, alternative='two-sided')
    print("Mann-Whitney test results for:", metric)
    print(s1, "<=>", s2)
    print("U-value: ", u_stat)
    print("P-value: ", p_value)
    print()

CalcUP(lr_nmirelueq[:,0], lr_relu32[:,0], "Accuracy", "nmiReLU-EQ", "ReLU-32")
CalcUP(rs_nmirelueq[:,0], rs_relu32[:,0], "CWL2 success rate", "nmiReLU-EQ", "ReLU-32")
CalcUP(rs_nmirelueq[:,2], rs_relu32[:,2], "BIM success rate", "nmiReLU-EQ", "ReLU-32")
CalcUP(rs_nmirelueq[:,2], rs_nm0relujn[:,2], "BIM success rate", "nmiReLU-EQ", "nmReLU-JN")
