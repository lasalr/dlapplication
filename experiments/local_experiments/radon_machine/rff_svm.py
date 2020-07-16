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


def sample_omega(x_mat, D, d):
	"""
	:param x_mat: n x d matrix (each row is a data point)
	:param D: Number of samples
	:param d: Number of dimensions of data (and for resulting omega)
	:return: iid sample from scaled normal distribution
	"""
	# Setting scale
	scale = (2 * np.pi) ** ((1 - D) * 0.5)

	# Scaled sample from the normal distribution
	omega = scale * np.random.normal(loc=0, scale=1, size=(D, d))
	return omega
