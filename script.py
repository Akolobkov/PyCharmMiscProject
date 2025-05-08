import numpy as np

gl_eps = 0.001


def equation(x1, x2):
    return 2*x1**2 + 1 * x1*x2 + 1 * x2**2
def dfdx1(x1, x2):
    return (equation(x1+gl_eps, x2)-equation(x1, x2))/gl_eps
def dfdx2(x1, x2):
    return (equation(x1, x2+gl_eps)-equation(x1, x2))/gl_eps


def grad(x1, x2):
    grad_arr = np.array([dfdx1(x1, x2), dfdx2(x1, x2)])
    return grad_arr
def Hessian(x1, x2):
    return np.array([[4, 1], [1, 2]])
def norm(arr):
    return (arr[0]**2 + arr[1]**2)**0.5
def norm1(x1, x2):
    return (x1**2 + x2**2)**0.5
def check(X):
    if X[0][0] > 0:
        if np.linalg.det(X)>0:
            return True
    return False
def find_tk(k):
    t = 1
    xarr[2*(k+1):2*(k+1)+1] = xarr[2*k:2*k +1] +t * dk
    while(equation(xarr[2*k], xarr[2*k+1])>=equation(xarr[2*(k+1)], xarr[2*(k+1)+1])):
        xarr[2 * (k + 1):2 * (k + 1) + 1] = xarr[2 * k:2 * k + 1] + t * dk
    return t

xarr = np.zeros(100)
xarr[0:2] = [1.5, 0.5]
eps1 = 0.15
eps2 = 0.2
M = 10
print("Шаг 1. x0:", xarr[0:2], "eps1:", eps1, "eps2:", eps2, "M:", M, "Задан градиент функции и Гессиан, он постоянен и равен", Hessian(0, 0))
k = 0
print("Шаг 2: k=0")
while (True):
    gradf = grad(xarr[k * 2], xarr[k * 2 + 1])
    print("Шаг 3: вычислим градиент x", k, ", Получим\n [", round(gradf[0], 3), round(gradf[1], 3), "]")
    if norm(gradf) < eps1:
        print("Шаг 4: рассчет окончен, получили минимум в точке [", round(xarr[2 * (k + 1)], 3), round( xarr[2 * (k + 1) + 1], 3), "]", "на итерации", k)
        break
    else:
        print("Шаг 4: условие не выполнено, норма градиента равна", round(norm(gradf), 3))
        if k >= M:
            print("Шаг 5: рассчет окончен, получили", xarr[2 * k:2 * k + 1+1], "на итерации", k)
        else:
            print("Шаг 5: условие не выполнено, k = ", k, ", что меньше M")
            print("Шаг 6: Hk =", Hessian(0,0))
            H_inv = np.linalg.inv(Hessian(xarr[2*k], xarr[2*k + 1]))
            print("Шаг 7: Обратная матрица:", H_inv)
            if check(H_inv):
                dk = -1 * H_inv @ gradf
                print("Шаг 8: H-1>0, Переходим к шагу 9;\n Шаг 9: dk = [", round(dk[0], 3), round(dk[1], 3), "]")
                tk = 1
            else:
                dk = -1 * gradf
                print("Шаг 8: H-1 не больше 0, Переходим к шагу 10; dk = [", round(dk[0], 3), round(dk[1], 3), "]")
                tk = find_tk(k)
            xarr[2 * (k + 1):2 * (k + 1) + 1+1] = xarr[2 * k:2 * k + 1 + 1] + tk * dk
            print("Шаг 10: xk+1 равно:", round(xarr[2 * (k + 1)], 3), round( xarr[2 * (k + 1) + 1], 3))
            if norm(xarr[2 * (k + 1):2 * (k + 1) + 1+1] - xarr[2 * k:2 * k + 1 + 1])<eps2:
                if abs(equation(xarr[2 * (k + 1)], xarr[2 * (k + 1) + 1]) - equation(xarr[2 * k],  xarr[2 * k + 1]))< eps2:
                    if (k>0):
                        k = k-1
                        if norm(xarr[2 * (k + 1):2 * (k + 1) + 1 + 1] - xarr[2 * k:2 * k + 1 + 1]) < eps2:
                            if abs(equation(xarr[2 * (k + 1)], xarr[2 * (k + 1) + 1]) - equation(xarr[2 * k], xarr[
                                2 * k + 1])) < eps2:
                                print("Шаг 11: Рассчет окончен на итерации: ", k+1, ",получили минимум в точке:", round(xarr[2 * (k + 1)], 3), round( xarr[2 * (k + 1) + 1], 3))
                                break
                        k = k+1
                    else:
                        print("Шаг 11: Рассчет окончен на итерации: ", k, ",получили минимум в точке: [", round(xarr[2 * (k + 1)], 3), round( xarr[2 * (k + 1) + 1], 3), "]")

                        break
            print("Условие на итерации", k, "не выполнено, переходим к шагу 3")
            k = k+1

