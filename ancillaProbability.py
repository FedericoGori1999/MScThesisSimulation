# Here we simulate a different circuit in order to obtain a better error mitigation for the measurement of the
# photon in the cavity

import numpy as np
import matplotlib.pyplot as plt


def hadamard_one_qubit(state, position, n_qubits):
    hadamard_matrix = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    if position == 1:
        current_matrix = hadamard_matrix
        i = 1
        while i < n_qubits:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i+1
    else:
        i = 1
        current_matrix = np.identity(2)
        while i < position-1:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i+1
        current_matrix = np.kron(current_matrix, hadamard_matrix)
        i = position
        while i < n_qubits + 1:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i+1
    return current_matrix @ state


def hadamard_many_qubits(state, n_qubits):
    hadamard_matrix = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    i = 1
    current_matrix = hadamard_matrix
    while i < n_qubits:
        current_matrix = np.kron(current_matrix, hadamard_matrix)
        i = i+1
    return current_matrix @ state


def hadamard_many_qubits_except_first(state, n_qubits):
    hadamard_matrix = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    i = 0
    current_matrix = np.identity(2)
    while i < n_qubits:
        current_matrix = np.kron(current_matrix, hadamard_matrix)
        i = i+1
    return current_matrix @ state


def cnot_adjacent(state, position_first_qubit, n_qubits, direction):
    cnot = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]]
    if direction == 0:
        if position_first_qubit == 1:
            i = 2
            current_matrix = cnot
            while i < n_qubits:
                current_matrix = np.kron(current_matrix, np.identity(2))
                i = i+1
        else:
            i = 1
            current_matrix = np.identity(2)
            while i < position_first_qubit-1:
                current_matrix = np.kron(current_matrix, np.identity(2))
                i = i+1
            current_matrix = np.kron(current_matrix, cnot)
            i = 0
            while i < n_qubits - position_first_qubit - 1:
                current_matrix = np.kron(current_matrix, np.identity(2))
                i = i+1
    elif direction == 1:
        if position_first_qubit != 1:
            i = 1
            current_matrix = np.identity(2)
            while i < position_first_qubit-1:
                current_matrix = np.kron(current_matrix, np.identity(2))
                i = i+1
            current_matrix = np.kron(current_matrix, cnot)
            i = 0
            while i < n_qubits - position_first_qubit-1:
                current_matrix = np.kron(current_matrix, np.identity(2))
                i = i+1
        else:
            current_matrix = cnot
            i = 2
            while i < n_qubits:
                current_matrix = np.kron(current_matrix, np.identity(2))
                i = i+1
    return current_matrix @ state


def cnot_not_adjacent(state, first_qubit, second_qubit, n_qubits):
    zero_state_matrix = [[1, 0],
                         [0, 0]]
    one_state_matrix = [[0, 0],
                        [0, 1]]
    pauli_x = [[0, 1],
               [1, 0]]
    if first_qubit == 1:
        current_matrix = zero_state_matrix
        i = 1
    else:
        current_matrix = np.identity(2)
        i = 2
        while i < first_qubit:
            current_matrix = np.identity(2)
            i = i + 1
        current_matrix = np.kron(current_matrix, zero_state_matrix)
    while i < n_qubits:
        current_matrix = np.kron(current_matrix, np.identity(2))
        i = i + 1
    sum_matrices = current_matrix
    if first_qubit == 1:
        current_matrix = one_state_matrix
        i = 1
    else:
        current_matrix = np.identity(2)
        i = 2
        while i < first_qubit:
            current_matrix = np.identity(2)
            i = i + 1
        current_matrix = np.kron(current_matrix, one_state_matrix)
    while i < n_qubits:
        if i == second_qubit-1:
            current_matrix = np.kron(current_matrix, pauli_x)
            i = i + 1
        else:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i + 1
    sum_matrices = sum_matrices + current_matrix
    return sum_matrices @ state


def pauli_x(state, position, n_qubits):
    pauli_x = [[0, 1],
               [1, 0]]
    if position == 1:
        current_matrix = pauli_x
        i = 1
        while i < n_qubits:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i+1
    else:
        i = 1
        current_matrix = np.identity(2)
        while i < position-1:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i+1
        current_matrix = np.kron(current_matrix, pauli_x)
        i = position
        while i < n_qubits + 1:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i+1
    return current_matrix @ state


def pauli_y(state, position, n_qubits):
    pauli_y = [[0, -1j],
               [1j, 0]]
    if position == 1:
        current_matrix = pauli_y
        i = 1
        while i < n_qubits:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i+1
    else:
        i = 1
        current_matrix = np.identity(2)
        while i < position-1:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i+1
        current_matrix = np.kron(current_matrix, pauli_y)
        i = position
        while i < n_qubits + 1:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i+1
    return current_matrix @ state


def pauli_z(state, position, n_qubits):
    pauli_z = [[1, 0],
               [0, -1]]
    if position == 1:
        current_matrix = pauli_z
        i = 1
        while i < n_qubits:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i+1
    else:
        i = 1
        current_matrix = np.identity(2)
        while i < position-1:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i+1
        current_matrix = np.kron(current_matrix, pauli_z)
        i = position
        while i < n_qubits + 1:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i+1
    return current_matrix @ state


def evolution(hamiltonian, state, t):
    import scipy
    from scipy.linalg import expm
    return (scipy.linalg.expm(1j * t * hamiltonian)) @ state


def hamiltonian_qubit(delta, n_photons, n_qubits):
    pauli_z = [[1, 0],
               [0, -1]]
    sum_matrices = np.zeros((2 ** n_qubits, 2 ** n_qubits))
    j = 0
    current_matrix = pauli_z
    while j < n_qubits - 1:
        current_matrix = np.kron(current_matrix, np.identity(2))
        j = j + 1
    sum_matrices = sum_matrices + current_matrix
    i = 1
    while i < n_qubits:
        current_matrix = np.identity(2)
        j = 1
        while j < i:
            current_matrix = np.kron(current_matrix, np.identity(2))
            j = j + 1
        current_matrix = np.kron(current_matrix, pauli_z)
        j = 1
        while j < n_qubits - i:
            current_matrix = np.kron(current_matrix, np.identity(2))
            j = j + 1
        sum_matrices = sum_matrices + current_matrix
        i = i + 1
    return delta * n_photons * sum_matrices


def measurement(delta, n_photons, n_qubits):
    t = 0.0
    dt = 0.1
    zero_state = np.array([1, 0])
    time_vector = np.array([0.0])
    probability_ancilla = np.array([0+0j])

    while t <= np.pi / delta:

        # Starting state |psi_0> is |0, ..., 0>

        psi = zero_state
        i = 1
        while i < n_qubits:
            psi = np.kron(psi, zero_state)
            i = i + 1

        # Applying the Hadamard Gate to the first qubit

        psi = hadamard_one_qubit(psi, 1, n_qubits)

        # Applying the CNOT between first and second qubit, second and third, and so on... 0 as direction from
        # first qubit to the last one

        if n_qubits > 1:
            i = 1
            while i < n_qubits:
                psi = cnot_adjacent(psi, i, n_qubits, 0)
                i = i+1

        # Evolution

        psi = evolution(hamiltonian_qubit(delta, n_photons, n_qubits), psi, t)

        # Application of the Hadamard gate on the main qubits

        psi = hadamard_many_qubits(psi, n_qubits)

        # Introduction of an ancilla qubit and application of the CNOTs on the ancilla qubit

        psi = np.kron(zero_state, psi)
        i = 2
        while i <= n_qubits+1:
            psi = cnot_not_adjacent(psi, 1, i, n_qubits+1)
            i = i + 1

        # Measuring the Ancilla in the zero state. If the ancilla is initially in the zero state, then in order to
        # measure it we need to measure the first half components of the vector

        i = 0
        j = 0
        counter_one = 0
        possible_configurations = []
        chosen_configurations = []
        final_indices = []
        while i < 2**n_qubits:
            possible_configurations = np.append(possible_configurations, decimal_to_binary(i))
            while j < len(possible_configurations[i]):
                if possible_configurations[i][j] == '1':
                    counter_one = counter_one + 1
                j = j + 1
            if counter_one % 2 == 0:
                chosen_configurations = np.append(chosen_configurations, possible_configurations[i])
            counter_one = 0
            i = i + 1
            j = 0
        i = 0
        while i < len(chosen_configurations):
            final_indices = np.append(final_indices, int(chosen_configurations[i], 2))
            i = i + 1
        sum_ancilla = 0
        i = 0
        while i < len(final_indices):
            index = int(final_indices[i])
            sum_ancilla = sum_ancilla + abs(psi[index])**2
            i = i + 1
        if t == 0:
            probability_ancilla[0] = probability_ancilla[0] + sum_ancilla
        else:
            probability_ancilla = np.append(probability_ancilla, [sum_ancilla])
            time_vector = np.append(time_vector, [t])
        t = t + dt

        # Applying the Hadamard Gate on the main qubits

        # psi = hadamard_many_qubits_except_first(psi, n_qubits)
    return probability_ancilla, time_vector


def decimal_to_binary(x):
    return bin(x).replace("0b", "")

# Declaring useful parameters for problem. Here delta = 0.1 GHz, while t and dt are in units of ns


delta = 0.1
n_photons = 1
n_qubits = 5
probability_ancilla1, time_vector1 = measurement(delta, n_photons, 2)
probability_ancilla2, time_vector2 = measurement(delta, n_photons, 4)
probability_ancilla3, time_vector3 = measurement(delta, n_photons, 6)

# Plotting the probability of obtaining the state |0> from the ancilla state as a function of time

fig, axs = plt.subplots(3, sharex='all')
axs[0].plot(time_vector1, probability_ancilla1, label='Probability of ancilla in |0> state with 2 qubits',
            color='blue')
axs[1].plot(time_vector2, probability_ancilla2, label='Probability of ancilla in |0> state with 4 qubits',
            color='green')
axs[2].plot(time_vector3, probability_ancilla3, label='Probability of ancilla in |0> state with 6 qubits',
            color='red')
plt.xlabel("Time (ns)")
axs[0].set_ylabel("Probability")
axs[1].set_ylabel("Probability")
axs[2].set_ylabel("Probability")
axs[0].legend(loc='upper right')
axs[1].legend(loc='upper right')
axs[2].legend(loc='upper right')
plt.savefig('ancillaOscillations.png')
plt.show()
