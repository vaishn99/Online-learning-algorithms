from cmath import sqrt
import numpy as np
from matplotlib import pyplot as plt


def get_N_LossF(T):
    def get_one():
        return np.random.multivariate_normal(mean=np.ones(shape=(3)), cov=np.identity(3))
    return np.asarray([get_one() for i in range(T)])

class Laz_Proj_GD:
    def __init__(self,T):
        self.T=T
        self.Cum_LV=list([])*T
        self.eta=None
    def Proj(self,z):
        if np.linalg.norm(z)<=1:
            return z
        else:
            return z/np.linalg.norm(z)

    def Run(self,Loss_Fn):

        w_old_tilda=np.zeros(shape=(3,1))       # w1        # indexing according to the iterates
        sum=w_old_tilda.T[0]@Loss_Fn[0]         # First loss
        self.Cum_LV.append(sum)

        for i in range(2,self.T+1):

            # starting from :  w2 =w1-eta_1*loss_1
            # goes till..
            # last updation :  (w_t)=(w_t-1)-(eta_t-1)*(l_t-1)

            w_new_tilda=w_old_tilda-self.eta[i-2]*np.array([Loss_Fn[i-2]]).T
            w_new_hat=self.Proj(w_new_tilda)

            sum=sum+(w_new_hat.T[0]@Loss_Fn[i-1])
            w_old_tilda=w_new_tilda
            self.Cum_LV.append(sum)

        return self.Cum_LV
class Laz_Proj_GD_const(Laz_Proj_GD):
    def __init__(self,T,step_size=0.01):
        super().__init__(T)
        self.stepsize=step_size
    def init_stepsize(self):
        self.eta=list([self.stepsize])*self.T
        return
class Laz_Proj_GD_variable(Laz_Proj_GD):
    def __init__(self,T):
        super().__init__(T)
    def init_stepsize(self):
        self.eta=list([])
        for i in range(1,self.T):
            self.eta.append(1/np.real(sqrt(i)))
        return

class Act_Proj_GD:
    def __init__(self,T):

        self.T=T
        self.Cum_LV=list([])*T
        self.eta=None

    def Proj(self,z):
        if np.linalg.norm(z)<=1:
            return z
        else:
            return z/np.linalg.norm(z)

    def Run(self,Loss_Fn):

        w_old_tilda=np.zeros(shape=(3,1))       # w1        # indexing according to the iterates
        sum=w_old_tilda.T[0]@Loss_Fn[0]         # First loss
        self.Cum_LV.append(sum)

        for i in range(2,self.T+1):
            w_new_tilda=w_old_tilda-self.eta[i-2]*np.array([Loss_Fn[i-2]]).T
            w_new_hat=self.Proj(w_new_tilda)

            sum=sum+w_new_hat.T[0]@Loss_Fn[i-1]
            w_old_tilda=w_new_hat
            self.Cum_LV.append(sum)

        return self.Cum_LV
class Act_Proj_GD_const(Act_Proj_GD):
    def __init__(self,T,step_size=0.01):
        super().__init__(T)
        self.stepsize=step_size
    def init_stepsize(self):
        self.eta=list([self.stepsize])*self.T
        return
class Act_Proj_GD_variable(Act_Proj_GD):
    def __init__(self,T):
        super().__init__(T)
    def init_stepsize(self):
        self.eta=list([])
        for i in range(1,self.T):
            self.eta.append(1/np.real(sqrt(i)))
        return

def Plot(Seq_1,Seq_2,Seq_3,Seq_4):
    plt.plot(np.arange(start=0,stop=len(Seq_1)),Seq_1,label="Lazy PGD with const step",color='red')
    plt.plot(np.arange(start=0,stop=len(Seq_1)),Seq_2,label="Lazy PGD with variable step",color='blue')
    plt.plot(np.arange(start=0,stop=len(Seq_1)),Seq_3,label="Active PGD with const step",color='black')
    plt.plot(np.arange(start=0,stop=len(Seq_1)),Seq_4,label="Active PGD with variable step",color='orange')
    plt.legend()
    plt.show()




T=10000
Loss_Fs=get_N_LossF(T=T)

optmzr_1=Laz_Proj_GD_const(T=T,step_size=0.01)
optmzr_2=Laz_Proj_GD_variable(T=T)
optmzr_3=Act_Proj_GD_const(T=T,step_size=0.01)
optmzr_4=Act_Proj_GD_variable(T=T)

optmzr_1.init_stepsize()
optmzr_2.init_stepsize()
optmzr_3.init_stepsize()
optmzr_4.init_stepsize()

Loss_seq_1=optmzr_1.Run(Loss_Fs)
Loss_seq_2=optmzr_2.Run(Loss_Fs)
Loss_seq_3=optmzr_3.Run(Loss_Fs)
Loss_seq_4=optmzr_4.Run(Loss_Fs)

# print("Cum Loss seq 1.\n")
# print(Loss_seq_1[:8])
# print("\n\n")
# print("Cum Loss seq 2.\n")
# print(Loss_seq_2[0:8])
# print("\n\n")
# print("Cum Loss seq 3.\n")
# print(Loss_seq_3[:8])
# print("\n\n")
# print("Cum Loss seq 4.\n")
# print(Loss_seq_4[:8])

Plot(Loss_seq_1,Loss_seq_2,Loss_seq_3,Loss_seq_4)
# all in the same plot
 







