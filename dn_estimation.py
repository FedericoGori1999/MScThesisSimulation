import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


# Applies the Hadamard gate to a specified qubit, through 'position'

def hadamard(rho, position, n_qubits):
    hadamard_matrix = (1 / np.sqrt(2)) * np.matrix([[1, 1], [1, -1]])
    if position == 1:
        current_matrix = hadamard_matrix
        i = 1
        while i < n_qubits:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i + 1
    else:
        i = 1
        current_matrix = np.identity(2)
        while i < position - 1:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i + 1
        current_matrix = np.kron(current_matrix, hadamard_matrix)
        i = position
        while i < n_qubits + 1:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i + 1
    return current_matrix @ rho @ current_matrix


# Applies the CNOT gate to a specified qubit, through 'position'.
# 'Direction' indicates whether the control qubit is up (0) or down (1) (?)

def cnot(rho, position_first_qubit, n_qubits):
    cnot = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    if position_first_qubit == 1:
        i = 2
        current_matrix = cnot
        while i < n_qubits:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i + 1
    else:
        i = 1
        current_matrix = np.identity(2)
        while i < position_first_qubit - 1:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i + 1
        current_matrix = np.kron(current_matrix, cnot)
        i = 0
        while i < n_qubits - position_first_qubit - 1:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i + 1
    return current_matrix @ rho @ current_matrix


# Applies the Pauli X gate to a specified qubit, through 'position'


def pauli_matrix_noise(pauli_matrix, rho, position, n_qubits):
    if position == 1:
        current_matrix = pauli_matrix
        i = 1
        while i < n_qubits:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i + 1
    else:
        i = 1
        current_matrix = np.identity(2)
        while i < position - 1:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i + 1
        current_matrix = np.kron(current_matrix, pauli_matrix)
        i = position
        while i < n_qubits:
            current_matrix = np.kron(current_matrix, np.identity(2))
            i = i + 1
    return current_matrix @ rho @ current_matrix


# Computes the Hamiltonian for the problem

def hamiltonian(delta, n_photons, n_qubits):
    pauli_z = np.matrix([[1, 0], [0, -1]])
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


# Computes the initial density matrix (initial state of the system is |0, ..., 0>)

def density_all_zero_state(n_qubits):
    zero_state = np.array([1, 0])
    if n_qubits == 1:
        return np.outer(zero_state, zero_state)
    else:
        i = 2
        while i <= n_qubits:
            if i == 2:
                psi_0 = np.kron(zero_state, zero_state)
            else:
                psi_0 = np.kron(psi_0, zero_state)
            i = i + 1
        return np.outer(psi_0, psi_0)

    # Performs the measurement process


def measurement(delta, evolve_for_short_range, n_photons, n_qubits, lambda_factor):

    # Time parameters are in ns units

    if evolve_for_short_range is True:
        time_simulation_max = 0.01 * np.pi / delta
    else:
        time_simulation_max = (6 * np.pi / delta)
    pauli_x = np.matrix([[0, 1], [1, 0]])
    pauli_y = np.matrix([[0, -1j], [1j, 0]])
    pauli_z = np.matrix([[1, 0], [0, -1]])
    rho_0 = density_all_zero_state(n_qubits)
    dt = 0.0001 * time_simulation_max
    time_vector = np.array([dt])
    t_maximum = dt

    final_rho = [rho_0]
    rho = rho_0

    # Hadamard gate on first qubit

    rho = hadamard(rho, 1, n_qubits)

    # Applying the CNOT between first and second qubit, second and third, and so on...
    # 0 as direction from first qubit to the last one

    if n_qubits > 1:
        i = 1
        while i < n_qubits:
            rho = cnot(rho, i, n_qubits)
            i = i + 1

    while t_maximum <= time_simulation_max:

        # Free evolution

        free_evolution = expm((-1j) * hamiltonian(delta, n_photons, n_qubits) * dt)
        rho = free_evolution.conj() @ rho @ free_evolution

        # Lindbladian operator

        rho = (1 - (n_qubits * 3 * (lambda_factor * dt))) * rho
        i = 1
        while i <= n_qubits:
            rho = rho + (lambda_factor * dt) * (pauli_matrix_noise(pauli_x, rho, i, n_qubits)
                                                + pauli_matrix_noise(pauli_y, rho, i, n_qubits)
                                                + pauli_matrix_noise(pauli_z, rho, i, n_qubits))
            i = i + 1

        # Applying the CNOT between nth and nth-1 qubit, ..., second and first qubit

        rho_array = np.array(rho)
        if n_qubits > 1:
            i = n_qubits
            while i > 1:
                rho_array = cnot(rho_array, i - 1, n_qubits)
                i = i - 1

        # Applying the Hadamard Gate to the first qubit

        rho_array = hadamard(rho_array, 1, n_qubits)

        # Now we apply the CNOT between first and second qubits, second and third qubits, ...,
        # in order to get the |0, ..., 0> and |1, ..., 1> states

        if n_qubits > 1:
            i = 1
            while i < n_qubits:
                rho_array = cnot(rho_array, i, n_qubits)
                i = i + 1
        final_rho.append(rho_array)
        t_maximum = t_maximum + dt
        time_vector = np.append(time_vector, [t_maximum])

    final_rho = np.array(final_rho)
    mask = build_mask(time_vector, n_photons, delta, dt)

    return final_rho, time_vector, mask


def remaining_part_of_fit(time_vector, lambda_factor, n_qubits):
    result = 0
    i = 0
    while i <= np.floor(n_qubits/2):
        result = result + math.comb(n_qubits, 2 * i) * np.exp(-8 * i * lambda_factor * time_vector)
        i = i + 1
    return result


def fit_function(n_qubits, n_photons, time_vector, lambda_factor, delta):
    return 1/2 * np.cos(2 * n_qubits * n_photons * delta * time_vector) \
        * np.exp(-4 * n_qubits * lambda_factor * time_vector) + 1/np.power(2, n_qubits) \
        * remaining_part_of_fit(time_vector, lambda_factor, n_qubits)


def build_mask(time_vector, n_photons, delta, dt):
    tolerance_time_value = 0.9 * dt
    if n_photons != 0:
        return np.isclose(np.mod(time_vector, np.pi / (2 * n_photons * delta)), 0.0,
                          atol=tolerance_time_value)
    else:
        return 0


def error_number_photons_linear(delta, time_vector, d_alpha, n_photons, error_delta, error_time):
    return d_alpha/(delta * time_vector) + (n_photons * error_delta)/delta + (n_photons * error_time)/time_vector


def error_number_photons_quadratic(delta, time_vector, d_alpha, n_photons, error_delta, error_time):
    return np.sqrt((d_alpha/(delta * time_vector)) * (d_alpha/(delta * time_vector)) +
                   ((n_photons * error_delta)/delta) * ((n_photons * error_delta)/delta) +
                   ((n_photons * error_time)/time_vector) * ((n_photons * error_time)/time_vector))


def calculate_g1(delta, time_vector, n_qubits, lambda_factor, n_photons):
    partial_term = 0
    i = 0
    while i <= np.floor(n_qubits / 2):
        partial_term = partial_term + math.comb(n_qubits, 2 * i) * np.exp(
            -8 * i * lambda_factor * time_vector) * ((8 * i * lambda_factor) / (n_photons * delta))
        i = i + 1
    return (-n_qubits * np.exp(-4 * n_qubits * lambda_factor * time_vector) *
            (np.cos(2 * n_qubits * n_photons * delta * time_vector) *
             ((2 * lambda_factor)/(n_photons * delta)) + np.sin(
                        2 * n_qubits * n_photons * delta * time_vector)) -
            1/np.power(2, n_qubits) * partial_term)


def calculate_g2(delta, time_vector, n_qubits, lambda_factor, n_photons):
    partial_term = 0
    i = 0
    while i <= np.floor(n_qubits / 2):
        partial_term = partial_term + math.comb(n_qubits, 2 * i) * np.exp(
            -8 * i * lambda_factor * time_vector) * ((32 * i * i * lambda_factor * lambda_factor) / (
                n_photons * n_photons * delta * delta))
        i = i + 1
    return (-0.5 * n_qubits * np.exp(-4 * n_qubits * lambda_factor * time_vector) *
            (((-8 * n_qubits * lambda_factor)/(n_photons * delta)) * np.sin(
                2 * n_qubits * n_photons * delta * time_vector) + 2 * n_qubits * np.cos(
                2 * n_qubits * n_photons * delta * time_vector) *
             (1 - ((4 * lambda_factor * lambda_factor)/(n_photons * n_photons * delta * delta))))
            + 1/np.power(2, n_qubits) * partial_term)


def calculate_d_alpha(delta, time_vector, dp, n_qubits, lambda_factor, n_photons, using_sign_plus_in_derivation):
    if using_sign_plus_in_derivation is True:
        coefficient = 1
    else:
        coefficient = -1
    g1 = calculate_g1(delta, time_vector, n_qubits, lambda_factor, n_photons)
    g2 = calculate_g2(delta, time_vector, n_qubits, lambda_factor, n_photons)
    return np.abs((-g1/(2 * g2)) + coefficient * np.sqrt((g1/(2 * g2)) * (g1/(2 * g2)) + dp/g2))


def comparison_error_dn(
        max_iteration, lambda_factor, comparing_runs_with_different_n_qubits,
        plot_entangled_values_comparison_different_n_qubits, using_mask, evolve_for_short_range, error_delta,
        error_time, using_linear_error_formula,
        using_sign_plus_in_derivation, plotting_error_d_alpha):
    n_qubits = 1
    while n_qubits <= max_iteration:
        final_rho, time_vector, mask = measurement(delta, evolve_for_short_range, n_photons,
                                                   n_qubits, lambda_factor)
        if n_qubits == 1:
            dp_no_ent = np.sqrt(final_rho[:, 0, 0] * (1 - final_rho[:, 0, 0]))
            time_vector_1_qubit = time_vector
        dp = np.sqrt(final_rho[:, 0, 0] * (1 - final_rho[:, 0, 0]))
        d_alpha_unentangled = (calculate_d_alpha(delta, time_vector_1_qubit, dp_no_ent, 1,
                                                 lambda_factor, n_photons,
                                                 using_sign_plus_in_derivation) / np.sqrt(n_qubits))
        d_alpha_entangled = calculate_d_alpha(delta, time_vector, dp, n_qubits,
                                              lambda_factor, n_photons, using_sign_plus_in_derivation)
        if comparing_runs_with_different_n_qubits is True:
            if plot_entangled_values_comparison_different_n_qubits is True:
                if using_mask is True:
                    if plotting_error_d_alpha is True:
                        plt.plot(time_vector[mask], d_alpha_entangled[mask],
                                 label=f'd_alphaEnt{n_qubits}qbtsMask')
                    else:
                        if using_linear_error_formula is True:
                            plt.plot(time_vector[mask], error_number_photons_linear(
                                delta, time_vector, d_alpha_entangled, n_photons, error_delta, error_time)[mask],
                                label=f'dnEnt{n_qubits}qbtsMaskLin')
                        else:
                            plt.plot(time_vector[mask], error_number_photons_quadratic(
                                delta, time_vector, d_alpha_entangled, n_photons, error_delta, error_time)[mask],
                                label=f'dnEnt{n_qubits}qbtsMaskQuad')
                else:
                    if plotting_error_d_alpha is True:
                        plt.plot(time_vector, d_alpha_entangled, label=f'd_alphaEnt{n_qubits}qbts')
                    else:
                        if using_linear_error_formula is True:
                            plt.plot(time_vector, error_number_photons_linear(
                                delta, time_vector, d_alpha_entangled, n_photons, error_delta, error_time),
                                label=f'dnEnt{n_qubits}qbtsLin')
                        else:
                            plt.plot(time_vector, error_number_photons_quadratic(
                                delta, time_vector, d_alpha_entangled, n_photons, error_delta, error_time),
                                label=f'dnEnt{n_qubits}qbtsQuad')
            else:
                if using_mask is True:
                    if plotting_error_d_alpha is True:
                        plt.plot(time_vector_1_qubit[mask], d_alpha_unentangled[mask],
                                 label=f'd_alphaUnent{n_qubits}qbtsMask')
                    else:
                        if using_linear_error_formula is True:
                            plt.plot(time_vector_1_qubit[mask], error_number_photons_linear(
                                delta, time_vector, d_alpha_unentangled, n_photons, error_delta, error_time)[mask],
                                     label=f'dnUnent{n_qubits}qbtsMaskLin')
                        else:
                            plt.plot(time_vector_1_qubit[mask], error_number_photons_quadratic(
                                delta, time_vector, d_alpha_unentangled, n_photons, error_delta, error_time)[mask],
                                label=f'dnUnent{n_qubits}qbtsMaskQuad')
                else:
                    if plotting_error_d_alpha is True:
                        plt.plot(time_vector_1_qubit, d_alpha_unentangled, label=f'd_alphaUnent{n_qubits}qbts')
                    else:
                        if using_linear_error_formula is True:
                            plt.plot(time_vector_1_qubit, error_number_photons_linear(
                                delta, time_vector, d_alpha_unentangled, n_photons, error_delta, error_time),
                                label=f'dnUnent{n_qubits}qbtsLin')
                        else:
                            plt.plot(time_vector_1_qubit, error_number_photons_quadratic(
                                delta, time_vector, d_alpha_unentangled, n_photons, error_delta, error_time),
                                label=f'dnUnent{n_qubits}qbtsQuad')
            n_qubits = n_qubits + 1
        else:
            if n_qubits == max_iteration:
                if using_mask is True:
                    if plotting_error_d_alpha is True:
                        plt.plot(time_vector[mask], d_alpha_entangled[mask],
                                 label=f'd_alphaEnt{n_qubits}qbtsMask')
                        plt.plot(time_vector_1_qubit[mask], d_alpha_unentangled[mask],
                                 label=f'd_alphaUnent{n_qubits}qbtsMask')
                    else:
                        if using_linear_error_formula is True:
                            plt.plot(time_vector[mask], error_number_photons_linear(
                                delta, time_vector, d_alpha_entangled, n_photons, error_delta, error_time)[mask],
                                label=f'Error in counting photons for {n_qubits} entangled qubits')
                            plt.plot(time_vector_1_qubit[mask], error_number_photons_linear(
                                delta, time_vector, d_alpha_unentangled, n_photons, error_delta, error_time)[mask],
                                label=f'Error in counting photons for {n_qubits} un-entangled qubits')
                        else:
                            plt.plot(time_vector[mask], error_number_photons_quadratic(
                                delta, time_vector, d_alpha_entangled, n_photons, error_delta, error_time)[mask],
                                label=f'dnEnt{n_qubits}qbtsMaskQuad')
                            plt.plot(time_vector_1_qubit[mask], error_number_photons_quadratic(
                                delta, time_vector, d_alpha_unentangled, n_photons, error_delta, error_time)[mask],
                                label=f'dnUnent{n_qubits}qbtsMaskQuad')
                else:
                    if plotting_error_d_alpha is True:
                        plt.plot(time_vector, d_alpha_entangled, label=f'd_alphaEnt{n_qubits}qbts')
                        plt.plot(time_vector_1_qubit, d_alpha_unentangled, label=f'd_alphaUnent{n_qubits}qbts')
                    else:
                        if using_linear_error_formula is True:
                            plt.plot(time_vector, error_number_photons_linear(
                                delta, time_vector, d_alpha_entangled, n_photons, error_delta, error_time),
                                label=f'dnEnt{n_qubits}qbtsLin')
                            plt.plot(time_vector_1_qubit, error_number_photons_linear(
                                delta, time_vector, d_alpha_unentangled, n_photons, error_delta, error_time),
                                label=f'dnUnent{n_qubits}qbtsLin')
                        else:
                            plt.plot(time_vector, error_number_photons_quadratic(
                                delta, time_vector, d_alpha_entangled, n_photons, error_delta, error_time),
                                label=f'dnEnt{n_qubits}qbtsQuad')
                            plt.plot(time_vector_1_qubit, error_number_photons_quadratic(
                                delta, time_vector, d_alpha_unentangled, n_photons, error_delta, error_time),
                                label=f'dnUnent{n_qubits}qbtsQuad')
                n_qubits = n_qubits + 1
            else:
                n_qubits = max_iteration
    return


def probability_plot(delta, evolve_for_short_range, n_photons, n_qubits, lambda_factor):
    final_rho, time_vector, mask = measurement(
        delta, evolve_for_short_range, n_photons, n_qubits, lambda_factor)
    plt.plot(time_vector, final_rho[:, 0, 0], label=f'Probability{n_qubits}qbts')
    return


def test_probability_formula(delta, n_photons, n_qubits, lambda_factor, evolve_for_short_range):
    final_rho, time_vector, mask = measurement(
        delta, evolve_for_short_range, n_photons, n_qubits, lambda_factor)
    plt.plot(time_vector, final_rho[:, 0, 0], label=f'{n_qubits} qubit: result of simulation')
    plt.plot(time_vector, fit_function(n_qubits, n_photons, time_vector, lambda_factor, delta),
             label=f'{n_qubits} qubit: analytical formula')
    return final_rho, time_vector

# Actual main program
# Declaring useful parameters for problem. Here delta = 0.1 GHz


delta = 0.1
error_delta = delta * 0.001
error_time = 0.000001
n_photons = 1
lambda_factor = 0.001
test_fit_function = False
assuming_presence_of_noise = True
using_mask = True
evolve_for_short_range = False
plot_the_probability = False
using_sign_plus_in_derivation = True
comparing_runs_with_different_n_qubits = False
plot_entangled_values_comparison_different_n_qubits = False
plotting_error_d_alpha = False
using_linear_error_formula = True
save_plot_comparisons = False
save_plot_different_lambda_factor = False
save_plot_error_builds_up = True
save_fit_function = False
max_iteration = 6
if evolve_for_short_range is True:
    using_mask = False
if plot_the_probability is True:
    probability_plot(delta, evolve_for_short_range, n_photons, max_iteration, lambda_factor)
if assuming_presence_of_noise is False:
    lambda_factor = 0
if n_photons == 0 or delta == 0:
    print("No evolution at all!")
elif test_fit_function is False:
    comparison_error_dn(max_iteration, lambda_factor, comparing_runs_with_different_n_qubits,
                        plot_entangled_values_comparison_different_n_qubits,
                        using_mask, evolve_for_short_range, error_delta, error_time, using_linear_error_formula,
                        using_sign_plus_in_derivation, plotting_error_d_alpha)
    # comparison_error_dn(max_iteration, 0.1 * lambda_factor, comparing_runs_with_different_n_qubits,
    #                    plot_entangled_values_comparison_different_n_qubits,
    #                    using_mask, evolve_for_short_range, error_delta, error_time, using_linear_error_formula,
    #                    using_sign_plus_in_derivation, plotting_error_d_alpha)
    # comparison_error_dn(max_iteration, 0.01 * lambda_factor, comparing_runs_with_different_n_qubits,
    #                    plot_entangled_values_comparison_different_n_qubits,
    #                    using_mask, evolve_for_short_range, error_delta, error_time, using_linear_error_formula,
    #                    using_sign_plus_in_derivation, plotting_error_d_alpha)
else:
    test_probability_formula(delta, n_photons, max_iteration, lambda_factor, evolve_for_short_range)
plt.legend(loc='upper right')
plt.xlabel('Time (ns)')
plt.ylabel('Error Δn')
if save_plot_comparisons is True and n_photons != 0 and delta != 0:
    if evolve_for_short_range is True:
        if comparing_runs_with_different_n_qubits is True:
            if plot_entangled_values_comparison_different_n_qubits is True:
                if plotting_error_d_alpha is True:
                    plt.savefig(f'{max_iteration}qubitsMaskAllEntangledShortRange_d_alpha.png')
                else:
                    plt.savefig(f'{max_iteration}qubitsMaskAllEntangledShortRange_dn.png')
            else:
                if plotting_error_d_alpha is True:
                    plt.savefig(f'{max_iteration}qubitsMaskAllUnentangledShortRange_d_alpha.png')
                else:
                    plt.savefig(f'{max_iteration}qubitsMaskAllUnentangledShortRange_dn.png')
        else:
            if plotting_error_d_alpha is True:
                plt.savefig(f'{max_iteration}qubitsMaskComparisonShortRange_d_alpha.png')
            else:
                plt.savefig(f'{max_iteration}qubitsMaskComparisonShortRange_dn.png')
    else:
        if using_mask is True:
            if comparing_runs_with_different_n_qubits is True:
                if plot_entangled_values_comparison_different_n_qubits is True:
                    if plotting_error_d_alpha is True:
                        plt.savefig(f'{max_iteration}qubitsMaskAllEntangled_d_alpha.png')
                    else:
                        plt.savefig(f'{max_iteration}qubitsMaskAllEntangled_dn.png')
                else:
                    if plotting_error_d_alpha is True:
                        plt.savefig(f'{max_iteration}qubitsMaskAllUnentangled_d_alpha.png')
                    else:
                        plt.savefig(f'{max_iteration}qubitsMaskAllUnentangled_dn.png')
            else:
                if plotting_error_d_alpha is True:
                    plt.savefig(f'{max_iteration}qubitsMaskComparison_d_alpha.png')
                else:
                    plt.savefig(f'{max_iteration}qubitsMaskComparison_dn.png')
        else:
            if comparing_runs_with_different_n_qubits is True:
                if plot_entangled_values_comparison_different_n_qubits is True:
                    if plotting_error_d_alpha is True:
                        plt.savefig(f'{max_iteration}qubitsNoMaskAllEntangled_d_alpha.png')
                    else:
                        plt.savefig(f'{max_iteration}qubitsNoMaskAllEntangled_dn.png')
                else:
                    if plotting_error_d_alpha is True:
                        plt.savefig(f'{max_iteration}qubitsNoMaskAllUnentangled_d_alpha.png')
                    else:
                        plt.savefig(f'{max_iteration}qubitsNoMaskAllUnentangled_dn.png')
            else:
                if plotting_error_d_alpha is True:
                    plt.savefig(f'{max_iteration}qubitsNoMaskComparison_d_alpha.png')
                else:
                    plt.savefig(f'{max_iteration}qubitsNoMaskComparison_dn.png')
elif save_plot_different_lambda_factor is True and n_photons != 0 and delta != 0:
    if plotting_error_d_alpha is True:
        plt.savefig(f'{max_iteration}qubits_d_alpha{lambda_factor}.png')
    else:
        plt.savefig(f'{max_iteration}qubits_dn{lambda_factor}.png')
elif save_plot_error_builds_up is True:
    if plotting_error_d_alpha is True:
        plt.savefig(f'{max_iteration}qubits_d_alphaIncrease.png')
    else:
        plt.savefig(f'{max_iteration}qubits_dnIncrease.png')
elif save_fit_function is True:
    plt.savefig(f'{max_iteration}qbtsComparison.png')
if n_photons != 0 and delta != 0:
    plt.show()
