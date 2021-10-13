import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
maker_ls = list(MarkerStyle.markers.values())

df = pd.read_csv("sweep_result_acc1.csv", sep=",") 
layer_names = df['name'].unique()
gamma_list = df['gamma'].unique()
noise_amp_list = df['noise_amp'].unique()[-4:]


for n in layer_names:
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    layer_idx = df['name']==n

    # gamma v.s. quant_error
    x_list = []
    quant_error_list = []
    for gamma in gamma_list:
        gamma_idx = (df['gamma']==gamma) & layer_idx
        x=gamma
        y_idx, = np.array(gamma_idx).nonzero()
        y=df.iloc[y_idx[0]]['quant error']
        # y=df[y_idx[0]:y_idx[0]+1]['quant error']
        x_list.append(x)
        quant_error_list.append(y)
    ax1.plot(x_list, quant_error_list, color="r", label=n, marker='+', markersize=4)
    
    # variation w.r.t. gamma and noise amp
    for amp in noise_amp_list:        
        amp_idx = (df['noise_amp']==amp) & layer_idx
        ax2.plot(gamma_list, df['noise error'].values[amp_idx], label=r'$\sigma={}$'.format(amp), marker='o', markersize=4)
        ax3.plot(gamma_list, df['total error'].values[amp_idx], label=r'$\sigma={}$'.format(amp), marker='v', markersize=4)
        ax4.plot(gamma_list, df['noisy acc@1'].values[amp_idx], label=r'$\sigma={}$'.format(amp), marker='v', markersize=4)

    # df['name']=="backbone.features.0" & df['']
    plt.legend(bbox_to_anchor=(0,0,1,1), loc="upper left",
            mode="expand", borderaxespad=0, ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig("test"+n+".pdf", format="pdf")
    plt.show()