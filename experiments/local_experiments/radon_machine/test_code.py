# import numpy as np
# import rff_svm
#
# n_test = 5 # Number of samples
# print(f'n_test = {n_test}')
# d_test = 4 # Number of features
# print(f'd_test = {d_test}')
# D_test = 2 # Number of components
# print(f'D_test = {D_test}')
# np.random.seed(123)
#
# test_x = np.random.random((n_test, d_test)) * 10
# test_omega = np.random.random((D_test, d_test)) * 10
#
# # result = rff_svm.feature_map(test_x, test_omega)
# # print(test_x)
# # print(result)
# # print(test_x.shape)
# # print(result.shape)
#
# # a = np.random.random((3, 3))
# # print(a)
# # print('----------')
# # b = np.random.random((3, 3))
# # print(b)
# # print('----------')
# # c = np.hstack((a, b))
# # print(c)
#
# print(test_omega)
# print(np.square(test_omega))
# exp_pow = np.sum(np.square(test_omega))
# print(-exp_pow)


def growth(N):
    return (0.5 * (N ** 2)) + (0.5 * N) + 1

if __name__ == '__main__':
    print(growth(5))