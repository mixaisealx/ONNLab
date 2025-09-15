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
    

lr_relu64 = get_learn_stat(False, FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_cmp1_ReLU.log')
lr_relu32 = get_learn_stat(False, FOLDER_PATH+'/exp_iReLU_ReLU_mnist_cmp2_relu.log', FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__ReLU.log')
lr_nm0relujn = get_learn_stat(True, FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmReLU.log')
lr_irelu32 = get_learn_stat(False,FOLDER_PATH+'/exp_iReLU_ReLU_mnist_cmp2_irelu.log')
lr_nmirelujn = get_learn_stat(True,FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_cmp1_nmiReLU-JN.log', FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmiReLU_JN.log')
lr_nmirelueq = get_learn_stat(True,FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_cmp1_nmiReLU-EQ.log', FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmiReLU_EQ.log')

def print_mean_std(mat, label):
    print("----", label, "----", " len=", mat.shape[0])
    coeff = np.corrcoef(mat[:,0], mat[:,1])
    print('Accuracy<->F1 correlation:', f'{min(coeff[0,1], coeff[1,0]):.5f}')
    print('Means:', list(mat.mean(axis=0)))
    print(' STDs:', list(mat.std(axis=0)))

print_mean_std(lr_relu64, "ReLU 64")
print_mean_std(lr_relu32, "ReLU 32")
print_mean_std(lr_irelu32, "iReLU 32")
print_mean_std(lr_nm0relujn, "nmReLU JN")
print_mean_std(lr_nmirelujn, "nmiReLU JN")
print_mean_std(lr_nmirelueq, "nmiReLU EQ")

fig,ax = plt.subplots(3, 1)

sns.kdeplot(lr_relu32[:,0], color='blue', label='ReLU 32', common_norm=True, ax=ax[0])
sns.kdeplot(lr_irelu32[:,0], color='red', label='iReLU 32', common_norm=True, ax=ax[0])
ax[0].xaxis.set_major_locator(MultipleLocator(0.15))
ax[0].xaxis.set_minor_locator(AutoMinorLocator(5))
ax[0].yaxis.set_major_locator(MultipleLocator(0.20))
ax[0].yaxis.set_minor_locator(AutoMinorLocator(2))
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0].grid(which='minor', color='#CCCCCC', linestyle=':')
ax[0].grid()
ax[0].set_xlabel('Values')
ax[0].set_ylabel('Density of values')
ax[0].legend()

sns.kdeplot(lr_relu64[:,0], color='red', label='ReLU 64', common_norm=True, ax=ax[1])
sns.kdeplot(lr_nmirelueq[:,0], color='green', label='nmiReLU EQ', common_norm=True, ax=ax[1])
sns.kdeplot(lr_relu32[:,0], color='blue', label='ReLU 32', common_norm=True, ax=ax[1])
ax[1].xaxis.set_major_locator(MultipleLocator(0.15))
ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))
ax[1].yaxis.set_major_locator(MultipleLocator(0.20))
ax[1].yaxis.set_minor_locator(AutoMinorLocator(2))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[1].grid(which='minor', color='#CCCCCC', linestyle=':')
ax[1].grid()
ax[1].set_xlabel('Values')
ax[1].set_ylabel('Density of values')
ax[1].legend()

sns.kdeplot(lr_relu32[:,0], color='blue', label='ReLU 32', common_norm=True, ax=ax[2])
sns.kdeplot(lr_nm0relujn[:,0], color='red', label='nmReLU JN', common_norm=True, ax=ax[2])
sns.kdeplot(lr_nmirelujn[:,0], color='magenta', label='nmiReLU JN', common_norm=True, ax=ax[2])
sns.kdeplot(lr_nmirelueq[:,0], color='green', label='nmiReLU EQ', common_norm=True, ax=ax[2])
ax[2].xaxis.set_major_locator(MultipleLocator(0.15))
ax[2].xaxis.set_minor_locator(AutoMinorLocator(4))
ax[2].yaxis.set_major_locator(MultipleLocator(0.20))
ax[2].yaxis.set_minor_locator(AutoMinorLocator(2))
ax[2].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[2].grid(which='minor', color='#CCCCCC', linestyle=':')
ax[2].grid()
ax[2].set_xlabel('Values')
ax[2].set_ylabel('Density of values')
ax[2].legend()

plt.tight_layout()
plt.show()
