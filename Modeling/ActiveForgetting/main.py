import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from custom_functions import PN_neuron, PN_KC_synapse, KC_neuron, KC_MBON_synapse, MBON_neuron, generate_connection

def network_update_for_tspan(network_tspan, danger, type, I_PN, PN_KC_conn, record_arg, PN_rec_ind, KC_rec_ind, 
                             v_PN, u_PN, S_PN_KC, v_KC, u_KC, S_KC_MBON, 
                             X_mat, Y_mat, D_mat, Da_mat, R_mat, Ra_mat, g_KC_MBON, v_MBON, u_MBON):
    v_rev = 0
    g_PN_KC = 0.25
    dt = 0.001  # ms
    dt_ms = dt * 1000
    numPN = len(v_PN)
    numKC = len(v_KC)
    numMBON = len(v_MBON)

    spike_MBON_count = np.zeros((numMBON, 1))
    DA_stim_mat = np.zeros((numKC, numMBON))

    # Recording variables
    v_PN_rec = v_PN[PN_rec_ind, :]
    v_KC_rec = v_KC[KC_rec_ind, :]
    I_MBON_rec = np.array([0, 0])
    v_MBON_rec = v_MBON
    X_rec = X_mat[KC_rec_ind, 0][:, np.newaxis]
    D_rec = D_mat[KC_rec_ind, 0][:, np.newaxis]
    R_rec = R_mat[KC_rec_ind, 0][:, np.newaxis]
    Ra_rec = Ra_mat[KC_rec_ind, 0][:, np.newaxis]
    g_rec = g_KC_MBON[KC_rec_ind, 0][:, np.newaxis]
    g_rec2 = g_KC_MBON[KC_rec_ind, 1][:, np.newaxis]

    # Main loop
    for idt in tqdm(range(int(network_tspan / dt))):

        # give punishment only to first EN
        if idt <= 1000 or idt >= 14000:
            danger = 0
        else:
            danger = 1
        
        if idt >= 9000 and idt <= 14000:
            I_PN = np.zeros((numPN, 1))
        else:
            I_PN = np.linspace(220, 340, numPN).reshape(-1, 1)

        if danger == 1:
            DA_stim_mat[:, 0] = 1
            DA_stim_mat[:, 1] = 0
        else:
            DA_stim_mat[:, 0] = 0
            DA_stim_mat[:, 1] = 0

        # Update PN neurons
        spike_PN, v_PN, u_PN = PN_neuron(dt_ms, v_PN, u_PN, I_PN)  # Placeholder for PN_neuron function
        # Update synapses from PN to KC
        spike_PN_mat = PN_KC_conn * spike_PN
        S_PN_KC = PN_KC_synapse(dt_ms, spike_PN_mat, S_PN_KC)  # Placeholder for PN_KC_synapse function
        I_KC = g_PN_KC * np.sum(S_PN_KC, axis=0)[:, np.newaxis] * (v_rev - v_KC)
        if idt == 1000:
            print(I_KC)
        spike_KC, v_KC, u_KC = KC_neuron(dt_ms, v_KC, u_KC, I_KC)  # Placeholder for KC_neuron function

        # Update synapses from KC to EN
        spike_KC_mat = np.tile(spike_KC, (1, 2))

        # Handling type_mat
        if type == "control":
            type_mat = np.ones((numKC, numMBON))
        elif type == "dRac1":
            type_mat = np.zeros((numKC, numMBON))

        # Synapse update
        S_KC_MBON, X_mat, Y_mat, D_mat, Da_mat, R_mat, Ra_mat, g_KC_MBON = KC_MBON_synapse(
            dt, dt_ms, S_KC_MBON, spike_KC_mat, DA_stim_mat, X_mat, Y_mat, D_mat, Da_mat, R_mat, Ra_mat, g_KC_MBON, type_mat)  # Placeholder for KC_EN_synapse function

        # Update EN neurons
        I_MBON = np.diag(g_KC_MBON.T @ S_KC_MBON)[:, np.newaxis] * (v_rev - v_MBON) * 0.05
        spike_MBON, v_MBON, u_MBON = MBON_neuron(dt_ms, v_MBON, u_MBON, I_MBON)  # Placeholder for EN_neuron function

        spike_MBON_count += spike_MBON

        # Recording
        if record_arg and idt % 5 == 0:
            v_MBON_rec = np.hstack([v_MBON_rec, v_MBON])
            g_rec = np.hstack([g_rec, g_KC_MBON[KC_rec_ind, 0][:, np.newaxis]])
            g_rec2 = np.hstack([g_rec2, g_KC_MBON[KC_rec_ind, 1][:, np.newaxis]])

    return spike_MBON_count, v_PN, u_PN, S_PN_KC, v_KC, u_KC, S_KC_MBON, X_mat, Y_mat, D_mat, Da_mat, R_mat, Ra_mat, g_KC_MBON, v_MBON, u_MBON, v_PN_rec, v_KC_rec, v_MBON_rec, D_rec, R_rec, g_rec, g_rec2




numPN = 49
numKC = 660
numMBON = 2

theta_IPN_50hz = 265
theta_IPN_spike = 200

stay_dur = 20
eval_dur = 2

type = "control"
#type = "control"  # or "dRac1"

record_arg = 1

num_conn = 10
PN_KC_conn = generate_connection(numPN, numKC, num_conn)  

S_PN_KC = np.zeros((numPN, numKC))
S_KC_MBON = np.zeros((numKC, numMBON))

spike_KC_mat = np.zeros((numKC, numMBON))

v_PN = np.ones((numPN, 1)) * -60
u_PN = np.zeros((numPN, 1))

v_KC = np.ones((numKC, 1)) * -85
u_KC = np.zeros((numKC, 1))

v_MBON = np.ones((numMBON, 1)) * -60
u_MBON = np.zeros((numMBON, 1))

PN_rec_ind = np.arange(0, 5)  
KC_rec_ind = np.arange(49, 650, 10)  

v_PN_rec_mat = v_PN[PN_rec_ind, :]
v_KC_rec_mat = v_KC[KC_rec_ind, :]
v_MBON_rec_mat = v_MBON
g_rec_mat = np.ones((len(KC_rec_ind), 1)) 
g_rec2_mat = np.ones((len(KC_rec_ind), 1))
D_rec_mat, R_rec_mat = np.zeros((len(KC_rec_ind), 1)), np.zeros((len(KC_rec_ind), 1))

X_mat, Y_mat, D_mat, R_mat, L_mat = np.zeros((numKC, numMBON)), np.zeros((numKC, numMBON)), np.zeros((numKC, numMBON)), np.zeros((numKC, numMBON)), np.zeros((numKC, numMBON))
Da_mat, Ra_mat, g_KC_MBON = np.ones((numKC, numMBON)), np.ones((numKC, numMBON)), np.ones((numKC, numMBON))

danger = 1

I_PN = np.linspace(220, 340, numPN).reshape(-1, 1)
#I_PN = 240* np.ones((numPN, 1))
print(I_PN)

spike_MBON_count, v_PN, u_PN, S_PN_KC, v_KC, u_KC, S_KC_MBON, X_mat, Y_mat, D_mat, Da_mat, R_mat, Ra_mat, g_KC_MBON, v_MBON, u_MBON, v_PN_rec, v_KC_rec, v_MBON_rec, D_rec, R_rec, g_rec, g_rec2 = network_update_for_tspan(stay_dur, danger, type, I_PN, PN_KC_conn, record_arg, PN_rec_ind, KC_rec_ind, v_PN, u_PN, S_PN_KC, v_KC, u_KC, S_KC_MBON, X_mat, Y_mat, D_mat, Da_mat, R_mat, Ra_mat, g_KC_MBON, v_MBON, u_MBON)

if record_arg:
    v_MBON_rec_mat = np.hstack([v_MBON_rec_mat, v_MBON_rec])
    g_rec_mat = np.hstack([g_rec_mat, g_rec])
    g_rec2_mat = np.hstack([g_rec2_mat, g_rec2])


approach_spike_count = spike_MBON_count[0]
avoid_spike_count = spike_MBON_count[1]



# Plotting
plt.subplot(211)
plt.plot(np.arange(g_rec_mat.shape[1]) * 0.001 * 5, np.average(g_rec_mat, axis=0), label='g(KC-MBON 1)', linewidth=1, color='#3C97D2')
#plt.axhline(-40, linestyle='--', label='spike threshold', color= "k")
plt.legend(loc='upper right', fontsize=10)
plt.xlabel('time (s)')
#plt.ylabel('v (mV)')
#plt.ylim([-65, -35])
plt.xlim([0, 20])
plt.title("control")

plt.subplot(212)
plt.plot(np.arange(g_rec2_mat.shape[1]) * 0.001 * 5, np.average(g_rec2_mat, axis=0), label='g(KC-MBON 2)', linewidth=1, color='#B32E30')
#plt.axhline(-40, linestyle='--', label='spike threshold', color= "k")
plt.legend(loc='upper right', fontsize=10)
plt.xlabel('time (s)')
#plt.ylabel('v (mV)')
#plt.ylim([-65, -35])
plt.xlim([0, 20])

plt.tight_layout()
plt.show()

# Plotting
plt.subplot(211)
plt.plot(np.arange(v_MBON_rec_mat.shape[1]) * 0.001 * 5, v_MBON_rec_mat[0, :], label='v(MBON 1)', linewidth=1, color='#3C97D2')
plt.axhline(-40, linestyle='--', label='spike threshold', color= "k")
plt.legend(loc='lower right', fontsize=10)
plt.xlabel('time (s)')
plt.ylabel('v (mV)')
plt.ylim([-65, -35])
plt.xlim([0, 20])
plt.title("control")

plt.subplot(212)
plt.plot(np.arange(v_MBON_rec_mat.shape[1]) * 0.001 * 5, v_MBON_rec_mat[1, :], label='v(MBON 2)', linewidth=1, color='#B32E30')
plt.axhline(-40, linestyle='--', label='spike threshold', color= "k")
plt.legend(loc='lower right', fontsize=10)
plt.xlabel('time (s)')
plt.ylabel('v (mV)')
plt.ylim([-65, -35])
plt.xlim([0, 20])

plt.tight_layout()
plt.show()