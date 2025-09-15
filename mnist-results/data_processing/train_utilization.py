import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, FormatStrFormatter)

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
    

lr_nm0relujn = get_learn_stat(True, FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmReLU.log')
lr_nmirelujn = get_learn_stat(True,FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_cmp1_nmiReLU-JN.log', FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmiReLU_JN.log')
lr_nmirelueq = get_learn_stat(True,FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_cmp1_nmiReLU-EQ.log', FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmiReLU_EQ.log')

def print_mean_std(mat, label):
    print("----", label, "----", " len=", mat.shape[0])
    print('Means:', list(mat.mean(axis=0)))
    print(' STDs:', list(mat.std(axis=0)))

print_mean_std(lr_nm0relujn, "nm0ReLU JN")
print_mean_std(lr_nmirelujn, "nmiReLU JN")
print_mean_std(lr_nmirelueq, "nmiReLU EQ")

fig,ax = plt.subplots(2, 1)

sns.kdeplot(lr_nm0relujn[:,2], color='blue', label='nmReLU JN', common_norm=True, ax=ax[0])
sns.kdeplot(lr_nmirelujn[:,2], color='red', label='nmiReLU JN', common_norm=True, ax=ax[0])
sns.kdeplot(lr_nmirelueq[:,2], color='green', label='nmiReLU EQ', common_norm=True, ax=ax[0])
ax[0].xaxis.set_major_locator(MultipleLocator(1))
ax[0].xaxis.set_minor_locator(AutoMinorLocator(5))
ax[0].yaxis.set_major_locator(MultipleLocator(0.05))
ax[0].yaxis.set_minor_locator(AutoMinorLocator(2))
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0].grid(which='minor', color='#CCCCCC', linestyle=':')
ax[0].grid()
ax[0].set_xlabel('Values')
ax[0].set_ylabel('Density of values')
ax[0].legend()

sns.kdeplot(lr_nm0relujn[:,3], color='blue', label='nmReLU JN', common_norm=True, ax=ax[1])
sns.kdeplot(lr_nmirelujn[:,3], color='red', label='nmiReLU JN', common_norm=True, ax=ax[1])
sns.kdeplot(lr_nmirelueq[:,3], color='green', label='nmiReLU EQ', common_norm=True, ax=ax[1])
ax[1].xaxis.set_major_locator(MultipleLocator(1))
ax[1].xaxis.set_minor_locator(AutoMinorLocator(5))
ax[1].yaxis.set_major_locator(MultipleLocator(0.05))
ax[1].yaxis.set_minor_locator(AutoMinorLocator(2))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[1].grid(which='minor', color='#CCCCCC', linestyle=':')
ax[1].grid()
ax[1].set_xlabel('Values')
ax[1].set_ylabel('Density of values')
ax[1].legend()

plt.tight_layout()
plt.show()
