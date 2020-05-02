import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

BATCH_SIZE = 1

C_in = [16, 32, 64, 128]
H_in = 256
W_in = 256
C_out = [1, 2, 4, 8, 16, 32, 64, 128, 256]
K = 3
S = 1
P = 0
D = 1
G = 1
bias = True

N_TEST = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    print(f'You are currently running with device : {DEVICE}')

    fig, axs = plt.subplots(2)
    fig.suptitle(f'Time and parameters for Conv2D forward pass (mean on {N_TEST} tests)')
    axs[0].set_ylabel('Parameters')
    axs[1].set_ylabel('Time (ms)')
    axs[1].set_xlabel('Number of out channels')
    with torch.no_grad():
        for c_in in C_in:
            img_size = (BATCH_SIZE, c_in, H_in, W_in)
            inputs = np.zeros(img_size)
            inputs = torch.from_numpy(inputs).float().to(DEVICE)
            n_params = []
            times = []
            for c_out in C_out:
                print(f'Running for C_in : {c_in} and C_out : {c_out}')
                net = nn.Conv2d(c_in, c_out, K, S, P, D, G, bias).to(DEVICE)
                n_params.append(sum(p.numel() for p in net.parameters()))
                t0 = time.time()
                for _ in range(N_TEST):
                    _ = net.forward(inputs)
                times.append((time.time() - t0) / N_TEST * 1000)

            axs[0].plot(C_out, n_params, label=f'N channel In : {c_in}')
            axs[1].plot(C_out, times, label=f'N channel In: {c_in}')

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')
        plt.show()
