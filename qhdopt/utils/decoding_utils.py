import numpy as np
def spin_to_bitstring(spin_list):
    # spin_list is a dict
    list_len = len(spin_list)
    bitstring = []
    for k in np.arange(list_len):
        if spin_list[k] == 1:
            bitstring.append(0)
        else:
            bitstring.append(1)

    return bitstring

def hamming_bitstring_to_vec(bitstring, d, r):
    sample = np.zeros(d)
    for i in range(d):
        sample[i] = sum(bitstring[i * r: (i + 1) * r]) / r
    return sample

def binstr_to_bitstr(s):
    return list(map(int, list(s)))

def unary_bitstring_to_vec(bitstring, d, r):
    sample = np.zeros(d)

    for i in range(d):
        x_i = bitstring[i * r: (i + 1) * r]

        in_low_energy_subspace = True
        for j in range(r - 1):
            if x_i[j] > x_i[j + 1]:
                in_low_energy_subspace = False

        if in_low_energy_subspace:
            sample[i] = np.mean(x_i)
        else:
            return None

    return sample

def unary_bitstring_to_vec_soft(bitstring, d, r):
    sample = np.zeros(d)

    for i in range(d):
        x_i = np.array(bitstring[i * r: (i + 1) * r], dtype=float)

        # 投影到单调不降（unary 子空间）
        x_i_proj = np.maximum.accumulate(x_i)

        sample[i] = np.mean(x_i_proj)

    return sample


def onehot_bitstring_to_vec(bitstring, d, r):
    sample = np.zeros(d)

    for i in range(d):
        x_i = bitstring[i * r: (i + 1) * r]
        if sum(x_i) != 1:
            return None
        else:
            slot = 0
            while slot < r and x_i[slot] == 0:
                slot += 1
            sample[i] = 1 - slot / r

    return sample

def binary_bitstring_to_vec(bitstring, d, r):
    """
    Decode a fixed-width base-2 (standard binary) encoding.

    Each dimension uses r bits b_0 (LSB) .. b_{r-1} (MSB), bit value 1 meaning
    that qubit measured in the "one" state. The decoded value is normalized
    to [0, 1]:

        x = (sum_j 2**j * b_j) / (2**r - 1)

    Unlike unary/one-hot, every bit pattern is a valid codeword (there is no
    penalty subspace to fall outside of), so this never returns None. This
    is far more qubit-efficient than unary (r bits give 2**r levels instead
    of r+1), which matters for gate-model hardware (e.g. IBM Quantum/QAOA)
    where qubit count is the binding constraint. The trade-off is that the
    corresponding problem Hamiltonian (see Backend.H_p) can only be built
    exactly for univariate/bivariate factors that are degree <= 2
    polynomials of the underlying continuous variable -- see
    Backend._fit_quadratic_from_callable.
    """
    sample = np.zeros(d)
    denom = float((1 << r) - 1) if r > 0 else 1.0
    for i in range(d):
        bits = bitstring[i * r: (i + 1) * r]
        value = sum((1 << j) * int(bits[j]) for j in range(r))
        sample[i] = value / denom if denom > 0 else 0.0
    return sample

def bitstring_to_vec(embedding_scheme, bitstring, d, r):
    if embedding_scheme == "unary":
        sample_return = unary_bitstring_to_vec(bitstring, d, r)
        if sample_return is None:
            return unary_bitstring_to_vec_soft(bitstring, d, r)
        else:
            return sample_return
    elif embedding_scheme == "onehot":
        return onehot_bitstring_to_vec(bitstring, d, r)
    elif embedding_scheme == "hamming":
        return hamming_bitstring_to_vec(bitstring, d, r)
    elif embedding_scheme == "binary":
        return binary_bitstring_to_vec(bitstring, d, r)
    else:
        raise Exception("Illegal embedding scheme.")

