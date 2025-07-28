import numpy as np

def generate_connection(row, col, num_conn_per_col):
    connection = np.zeros((row, col))
    for icol in range(col):
        rand_conn = np.random.choice(row, num_conn_per_col, replace=False)
        connection[rand_conn, icol] = 1
    return connection


def synapse(dt, S, spikes, phi_S, tau_syn_S):
    # Update S based on spikes and decay
    S = S - S/tau_syn_S*dt + spikes*phi_S
    return S


def PN_neuron(dt, v, u, I):
    C = 100
    a = 0.3
    b = -0.2
    c = -65
    d = 8
    k = 2
    v_r = -60
    v_t = -40
    epsilon_mean = 0
    epsilon_std = 0.05

    spike = np.zeros_like(v)

    # Iterate over each element for reset condition
    for idx in range(len(v)):
        if v[idx] > v_t:
            v[idx] = c
            u[idx] += d
            spike[idx] = 1

    # Noise term
    epsilon = epsilon_mean + epsilon_std * np.random.randn(*v.shape)

    # update v and u
    v = v + dt/2 * ((k * (v - v_r) * (v - v_t) - u + I + epsilon) / C)
    v = v + dt/2 * ((k * (v - v_r) * (v - v_t) - u + I + epsilon) / C)
    u = u + dt * (a * (b * (v - v_r) - u))

    return spike, v, u


def PN_KC_synapse(dt, spikes, S):
    phi_S = 0.93
    tau_syn_S = 3.0  # [ms]

    # Call the synapse function
    S = synapse(dt, S, spikes, phi_S, tau_syn_S)

    return S



def KC_neuron(dt, v, u, I):
    C = 4
    a = 0.01
    b = -0.3
    c = -65
    d = 8
    k = 0.035
    v_r = -85
    v_t = -25
    epsilon_mean = 0
    epsilon_std = 0.05

    spike = np.zeros_like(v)

    # Iterate over each element for reset condition
    for idx in range(len(v)):
        if v[idx] > v_t:
            v[idx] = c
            u[idx] += d
            spike[idx] = 1

    # Noise term
    epsilon = epsilon_mean + epsilon_std * np.random.randn(*v.shape)

    # Update equations
    v += dt/2 * ((k * (v - v_r) * (v - v_t) - u + I + epsilon) / C)
    v += dt/2 * ((k * (v - v_r) * (v - v_t) - u + I + epsilon) / C)
    u += dt * (a * (b * (v - v_r) - u))

    return spike, v, u

def KC_MBON_synapse(dt, dt_ms, S, spikes, DA_stim, X, Y, D, Da, R, Ra, g, type_num):
    # Params
    tau_x = 0.0245
    tau_y = 0.9
    theta_da = 0.0065
    r_d = 0.2816
    r_r = 0.0887
    alpha_d = 78.92
    alpha_r = 6.9581
    A_d = 5
    A_r = 1
    X_thresh = 0.1
    k_rec = 0.005
    phi_S = 8
    tau_syn_S = 8

    # Assuming synapse function is defined elsewhere
    S = synapse(dt_ms, S, spikes, phi_S, tau_syn_S)

    # Initialize matrices
    odor = spikes
    L = np.zeros_like(X)

    # Process each element
    rows, cols = X.shape
    for i in range(rows):
        for j in range(cols):
            if j == 0:
                if DA_stim[i, j] == 1:
                    da = DA_stim[i, j]
                    L[i, j] = 0
                else: 
                    da = theta_da
                    L[i, j] = 1

                dXdt = -X[i, j] / tau_x
                dYdt = ((da - theta_da) - Y[i, j]) / tau_y
                dDdt = r_d * X[i, j] * (da - theta_da) * Da[i, j] - alpha_d * D[i, j]
                dDadt = k_rec * (1 - D[i, j] - Da[i, j]) - r_d * X[i, j] * (da - theta_da) * Da[i, j]

                if type_num[i,j] == 1:
                    L[i, j] = 0 if X[i, j] > X_thresh else 1
                         
                dRdt = r_r * L[i, j] * da * Ra[i, j] - alpha_r * R[i, j]
                dRadt = k_rec * (1 - R[i, j] - Ra[i, j]) - r_r * L[i, j] * da * Ra[i, j]
                dgdt = -A_d * alpha_d * D[i, j] + A_r * alpha_r * R[i, j]

                X[i, j] += dXdt * dt + odor[i,j]
                Y[i, j] += dYdt * dt
                D[i, j] += dDdt * dt
                Da[i, j] += dDadt * dt
                R[i, j] += dRdt * dt
                Ra[i, j] += dRadt * dt
                g[i, j] = min(1, g[i, j] + dgdt * dt)
                g[i, j] = max(0, g[i, j])
            else:
                X[i,j] = X[i, j]
                Y[i,j] = Y[i, j]
                D[i,j] = D[i, j]
                Da[i,j] = Da[i, j]
                R[i,j] = R[i, j]
                Ra[i,j] = Ra[i, j]
                g[i,j] = g[i, j]     

    return S, X, Y, D, Da, R, Ra, g

'''
def KC_MBON_synapse(dt, dt_ms, S, spikes, DA_stim, X, Y, D, Da, R, Ra, g, type_num):
    # Parameters
    tau_x = 0.0245
    tau_y = 0.9
    theta_da = 0.0065
    r_d = 0.2816
    r_r = 0.0887
    alpha_d = 78.92
    alpha_r = 6.9581
    A_d = 5
    A_r = 2
    X_thresh = 0.1
    k_rec = 0.005  # recover rate
    phi_S = 8
    tau_syn_S = 8

    # Synapse function
    S = synapse(dt_ms, S, spikes, phi_S, tau_syn_S)

    odor = spikes
    if np.any(DA_stim[:, 0] == 1):
        da = 1
        L = 0
    else:
        da = theta_da
        L = 1

    # Differential equations
    dXdt = -X/tau_x
    dYdt = ((da - theta_da) - Y)/tau_y
    dDdt = r_d * X * (da - theta_da) * Da - alpha_d * D
    dDadt = k_rec * (1 - D - Da) - r_d * X * (da - theta_da) * Da

    if np.any(type_num == 1):
        if np.any(X > X_thresh):
            L = 0
        else:
            L = 1

    dRdt = r_r * L * da * Ra - alpha_r * R
    dRadt = k_rec * (1 - R - Ra) - r_r * L * da * Ra

    # Weight update rules
    dgdt = - A_d * alpha_d * D + A_r * alpha_r * R

    # State updates
    X = X + dXdt * dt + odor
    Y = Y + dYdt * dt
    D = D + dDdt * dt
    Da = Da + dDadt * dt
    R = R + dRdt * dt
    Ra = Ra + dRadt * dt
    g = np.clip(g + dgdt * dt, 0, 1)

    return S, X, Y, D, Da, R, Ra, g
'''

def MBON_neuron(dt, v, u, I):
    C = 100
    a = 0.3
    b = -0.2
    c = -65
    d = 8
    k = 2
    v_r = -60
    v_t = -40
    epsilon_mean = 0
    epsilon_std = 0.05

    spike = np.zeros_like(v)

    # Iterate over each element for reset condition
    for idx in range(len(v)):
        if v[idx] > v_t:
            v[idx] = c
            u[idx] += d
            spike[idx] = 1

    # Noise term
    epsilon = epsilon_mean + epsilon_std * np.random.randn(*v.shape)

    # Update v and u with consideration for preventing overload
    v = np.maximum(v + dt/2 * ((k * (v - v_r) * (v - v_t) - u + I + epsilon) / C), -100)
    v = np.minimum(v, 50)
    v = np.maximum(v + dt/2 * ((k * (v - v_r) * (v - v_t) - u + I + epsilon) / C), -100)
    v = np.minimum(v, 50)
    u = u + dt * (a * (b * (v - v_r) - u))

    return spike, v, u
