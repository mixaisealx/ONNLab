import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib.gridspec as gridspec

FOLDER_PATH = '../raw_data'

def get_resist_stat(filename, nm=True):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                if nm:
                    spl = line.rstrip().split('\t')
                    data.append([float(spl[5]), float(spl[8]), float(spl[6]), float(spl[9]), float(spl[2]), float(spl[3])])
                else:
                    spl = line.rstrip().split('\t')
                    data.append([float(spl[3]), float(spl[6]), float(spl[4]), float(spl[7])])
            except:
                pass
    return np.array(data)
    

rs_relu32 = get_resist_stat(FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__ReLU.log', False)
rs_nm0relujn = get_resist_stat(FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmReLU.log')
rs_nmirelujn = get_resist_stat(FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmiReLU_JN.log')
rs_nmirelueq = get_resist_stat(FOLDER_PATH+'/exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmiReLU_EQ.log')

def print_mean_std(mat, label):
    print("----", label, "----", " len=", mat.shape[0])
    print(list(mat.mean(axis=0)))
    print(list(mat.std(axis=0)))

print_mean_std(rs_relu32, "ReLU 32")
print_mean_std(rs_nm0relujn, "nmReLU JN")
print_mean_std(rs_nmirelujn, "nmiReLU JN")
print_mean_std(rs_nmirelueq, "nmiReLU EQ")

fig = plt.figure()
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, :])

sns.kdeplot(rs_relu32[:,0], color='red', label='ReLU 32', common_norm=True, ax=ax0)
ax0.xaxis.set_major_locator(MultipleLocator(3))
ax0.xaxis.set_minor_locator(AutoMinorLocator(4))
ax0.yaxis.set_major_locator(MultipleLocator(0.2))
ax0.yaxis.set_minor_locator(AutoMinorLocator(4))
ax0.grid(which='minor', color='#CCCCCC', linestyle=':')
ax0.grid()
ax0.set_xlabel('ReLU-32 CWL2 success rate')
ax0.set_ylabel('Density of values')

sns.kdeplot(rs_relu32[:,2], color='red', label='ReLU 32', common_norm=True, ax=ax1)
ax1.xaxis.set_major_locator(MultipleLocator(0.2))
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax1.yaxis.set_major_locator(MultipleLocator(2))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.grid(which='minor', color='#CCCCCC', linestyle=':')
ax1.grid()
ax1.set_xlabel('ReLU-32 BIM success rate')
ax1.set_ylabel('Density of values')

sns.kdeplot(rs_nm0relujn[:,0], color='red', label='nmReLU JN', common_norm=True, ax=ax2)
sns.kdeplot(rs_nmirelujn[:,0], color='blue', label='nmiReLU JN', common_norm=True, ax=ax2)
sns.kdeplot(rs_nmirelueq[:,0], color='green', label='nmiReLU EQ', common_norm=True, ax=ax2)
ax2.xaxis.set_major_locator(MultipleLocator(5))
ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
ax2.yaxis.set_major_locator(MultipleLocator(0.01))
ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
ax2.grid(which='minor', color='#CCCCCC', linestyle=':')
ax2.grid()
ax2.set_xlabel('CWL2 success rate')
ax2.set_ylabel('Density of values')
ax2.legend()

sns.kdeplot(rs_nm0relujn[:,2], color='red', label='nmReLU JN', common_norm=True, ax=ax3)
sns.kdeplot(rs_nmirelujn[:,2], color='blue', label='nmiReLU JN', common_norm=True, ax=ax3)
sns.kdeplot(rs_nmirelueq[:,2], color='green', label='nmiReLU EQ', common_norm=True, ax=ax3)
ax3.xaxis.set_major_locator(MultipleLocator(5))
ax3.xaxis.set_minor_locator(AutoMinorLocator(5))
ax3.yaxis.set_major_locator(MultipleLocator(0.005))
ax3.yaxis.set_minor_locator(AutoMinorLocator(4))
ax3.grid(which='minor', color='#CCCCCC', linestyle=':')
ax3.grid()
ax3.set_xlabel('BIM success rate')
ax3.set_ylabel('Density of values')
ax3.legend()

plt.tight_layout()
plt.show()
