import numpy as np
from matplotlib import pyplot as plt

class RNN(object):

    def __init__(self, n_input, n_hidden, n_output):

        # 各層の数を設定
        self.n_input = n_input # 入力層の数
        self.n_hidden = n_hidden # 隠れ層の数
        self.n_output = n_output # 出力層の数

        # 重み配列を生成し、乱数で埋める
        self.hidden_weight = np.random.randn(n_hidden, n_input + 1) # 隠れ層の重み配列は[隠れ層、　入力層+1]
        self.output_weight = np.random.randn(n_output, n_hidden + 1) # 出力層の重み配列は[出力層、　隠れ層+1]
        self.recurr_weight = np.random.randn(n_hidden, n_hidden + 1) # 再帰層の重み配列は[隠れ層、　隠れ層+1]

    def train(self, Xl, epsilon, lam, epoch):

        # 損失値を繰り返しの数の配列とする
        self.__loss = np.zeros(epoch)

        for epo in range(epoch): # 繰り返しの数だけループする
            print('epoch: {0}'.format(epo))

            for X in np.random.permutation(list(Xl)): # 学習データ配列をランダムに取り出す
                # Xは[7,1] or [10,1] or [13,1]の配列のどれか
                tau = X.shape[0] # 取り出した学習データの個数をtauとする

                zs, ys = self.__forward_seq(X) # 配列Xについて順方向計算を行い、結果として隠れ層と出力層のリストを得る

                # 逆方向計算の初期化
                hidden_delta = np.zeros(self.n_hidden) # 隠れ層の微分値配列を初期化
                output_dEdw = np.zeros(self.output_weight.shape) # 出力層の微分値の総和を初期化
                hidden_dEdw = np.zeros(self.hidden_weight.shape) # 隠れ層の微分値の総和を初期化
                recurr_dEdw = np.zeros(self.recurr_weight.shape) # 再帰層の微分値の総和を初期化

                # 逆方向計算：tau-1から0までループする  tau=13のとき11→0 教師データは1個先のデータを使用するため　11の教師データは12
                for t in range(tau - 1)[::-1]:

                    # output delta 出力層の微分計算 誤差関数を２乗誤差(E=1/2*(y-t)^2)とすると微分値は(y-T)で、
                    # 教師データtは1個先の入力値なのでy[t]-x[t+1]となる。
                    # 出力層のアクティベート関数tanhの微分は(1-tanh(x)^2)であるが、y[t]=tanh(x[t])なので(1-y[t]^2)となる。
                    # δoutを求める
                    output_delta = (ys[t] - X[t + 1, :]) * (1.0 - ys[t] ** 2)
                    # δE/δWout(t)とδE/δcを求め累積する。
                    output_dEdw += output_delta.reshape(-1, 1) * np.hstack((1.0, zs[t]))

                    # hidden delta 隠れ層の計算
                    # 隠れ層のアクティベート関数であるシグモイドの微分はy・(1-y)なので(z[t]*(1-z[t])となる。z(t)=sigmoid(u[t])
                    # δを求める
                    hidden_delta = (self.output_weight[:, 1:].T.dot(output_delta) 
                                  + self.recurr_weight[:, 1:].T.dot(hidden_delta)) * zs[t] * (1.0 - zs[t])
                    hidden_dEdw += hidden_delta.reshape(-1, 1) * np.hstack((1.0, X[t, :]))

                    # recurr delta
                    zs_prev = zs[t - 1] if t > 0 else np.zeros(self.n_hidden)
                    recurr_dEdw += hidden_delta.reshape(-1, 1) * np.hstack((1.0, zs_prev))

                    # accumulate loss
                    self.__loss[epo] += 0.5 * (ys[t] - X[t + 1]).dot((ys[t] - X[t + 1]).reshape((-1, 1))) / (tau - 1)

                # update weights
                self.output_weight -= epsilon * (output_dEdw + lam * self.output_weight)
                self.hidden_weight -= epsilon * hidden_dEdw
                self.recurr_weight -= epsilon * recurr_dEdw


    def save_param(self, fn = 'weights.npy'):
        weights = {'h': self.hidden_weight, 'o': self.output_weight, 'r': self.recurr_weight}
        np.save(fn, weights)


    def save_lossfig(self, fn = 'loss.png'):
        plt.plot(np.arange(self.__loss.size), self.__loss)
        plt.savefig(fn)


    @classmethod
    def load(cls, fn = 'weights.npy'):
        weights = np.load(fn).item()
        n_input = weights['h'].shape[1] - 1
        n_hidden = weights['h'].shape[0]
        n_output = weights['o'].shape[0]
        rnn = RNN(n_input, n_hidden, n_output)
        rnn.hidden_weight = weights['h']
        rnn.output_weight = weights['o']
        rnn.recurr_weight = weights['r']
        return rnn


    def predict(self, X):
        _, ys = self.__forward_seq(X)
        return ys


    def predict_loop(self, X, times):
        zs, ys = self.__forward_seq(X)
        y, z = ys[-1], zs[-1]
        for i in range(times):
            z, y = self.__forward(y, z)
            zs.append(z)
            ys.append(y)

        return ys


    def __sigmoid(self, arr):
        return 1.0 / (1.0 + np.exp(-arr))


    def __tanh(self, arr):
        pl = np.exp(arr)
        mn = np.exp(-arr)
        return (pl - mn) / (pl + mn)

    # 1回分の順方向計算 入力x 前回の隠れ層の出力z
    def __forward(self, x, z):
        # 各計算の掛け算は内積 [x1, x2, x3]*[y1, y2, y3]=[x1*y1+x2*y2+x3*y3]
        # 再帰層の計算 r(t)=W*z(t-1)+d
        r = self.recurr_weight.dot(np.hstack((1.0, z)))
        # 隠れ層の計算　z(t)=sigmoid(Win*x(t)+b+r(t)) 隠れ層のアクティベート関数はシグモイド
        z = self.__sigmoid(self.hidden_weight.dot(np.hstack((1.0, x))) + r)
        # 出力層の計算 y(t)=tanh(Wout*z(t)+c) 出力層のアクティベート関数はハイパブリックタンジェント
        y = self.__tanh(self.output_weight.dot(np.hstack((1.0, z))))
        return (z, y)

    # 配列の順方向の計算
    def __forward_seq(self, X):
        # 隠れ層の出力z(t)を初期化する
        z = np.zeros(self.n_hidden)
        # 隠れ層と出力層の値をリストにするための初期化
        zs, ys = ([], [])
        # Xの各データについて順方向計算を行う
        for x in X:
            z, y = self.__forward(x, z)
            # 順方向計算結果の隠れ層と出力層をリストに加える
            zs.append(z)
            ys.append(y)
        # リストを結果として返す
        return zs, ys
