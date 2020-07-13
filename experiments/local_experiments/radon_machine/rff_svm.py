import numpy as np
import scipy.fft

def feature_map(x_mat, omega):
	"""
	:param x_mat: n x d matrix (each row is a data point)
	:param omega: D x d matrix (each row is omega_1, omega_2, ... omega_D)
	:return: Randomised feature map
	"""

	# Converting lists to np arrays
	x_mat = np.array(x_mat)
	omega = np.array(omega)
	D = omega.shape[0]  # Number of iid samples (D)
	n = x_mat.shape[0]

	if x_mat.shape[1] != omega.shape[1]:
		raise RuntimeError("Dimension of omega and x_mat do not match")

	omega_x_mat = np.matmul(x_mat, omega.T)

	assert (omega_x_mat.shape == (n, D)), "assert fail: omega_x_mat dimensions incorrect"

	cos_omega_x_mat = np.cos(omega_x_mat)
	# print("cos_omega_x_mat")
	# print(cos_omega_x_mat)
	sin_omega_x_mat = np.sin(omega_x_mat)
	# print("sin_omega_x_mat")
	# print(sin_omega_x_mat)

	# return ((1 / D) ** 0.5) * (np.hstack((cos_omega_x_mat, sin_omega_x_mat))).T
	return ((1 / D) ** 0.5) * (np.hstack((cos_omega_x_mat, sin_omega_x_mat)))

# def fourier_transform(omega, D):
# 	exp_pow = - np.sum(np.square(omega)) / 2
# 	return (2 * np.pi) ** (- D / 2) * np.exp(exp_pow)

def fourier_transform(omega, D):
	scipy.fft.fftn()



def inverse_fourier_transform(p):
	a = p / ((2 * np.pi) ** (- D / 2))
	b = -2 * np.log(a)