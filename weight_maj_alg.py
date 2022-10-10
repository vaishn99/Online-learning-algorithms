from matplotlib import pyplot as plt 
import numpy as np



class Markov_chain:
    def __init__(self,state,trans_matrix):
        self.states=state
        self.trans_prob_dict=dict()
        for x in range(len(self.states)):
            self.trans_prob_dict.update({self.states[x]:trans_matrix[x]})
        self.current_state=np.random.choice(self.states)
    def step(self):
        self.current_state=np.random.choice(a=self.states,p=self.trans_prob_dict[self.current_state])
        return self.current_state
    def reset(self):
        self.current_state=np.random.choice(self.states)
        return 
class Bernoulli_gen:
    def __init__(self,MC):
        self.MC=MC
    def step(self):
        return np.random.binomial(n=1,p=self.MC.step())
    def reset(self):
        self.MC.reset()
        return
class Expert:
    def __init__(self,id):
        self.id=id
        self.rel_past=None
    def step(self):
        return self.respective_step()
    def update(self,new_observation):
        self.relevant_update(new_observation)
        return

class Expert_one(Expert):
    def __init__(self, id=1):
        super().__init__(id)
    def respective_step(self):
        return 0
    def relevant_update(self,new_observation):
        return
    def reset(self):
        return
    def __str__(self):
        return "Expert 1"
class Expert_two(Expert):
    def __init__(self, id=2):
        super().__init__(id)
    def respective_step(self):
        return 1
    def relevant_update(self,new_observation):
        return
    def reset(self):
        return 
    def __str__(self):
        return "Expert 2"
class Expert_three(Expert):
    def __init__(self, id=3):
        super().__init__(id)
        self.rel_past=dict({0:0,1:0})
    def respective_step(self):
        if self.rel_past[0]>self.rel_past[1]:
            return 0
        return 1
    def reset(self):
        self.rel_past.update({0:0,1:0})
        return
    def relevant_update(self,new_observation):
        self.rel_past.update({new_observation:self.rel_past[new_observation]+1})
        return
    def __str__(self):
        return "Expert 3"
class Expert_four(Expert):
    def __init__(self, id=4):
        super().__init__(id)
        self.rel_past=list()
    def respective_step(self):
        if self.rel_past.count(0)>self.rel_past.count(1):
            return 0
        return 1
    def reset(self):
        self.rel_past=list([])
        return
    def relevant_update(self,new_observation):
        if len(self.rel_past)<10:
            self.rel_past.append(new_observation)
        else:
            self.rel_past.pop(0)
            self.rel_past.append(new_observation)
        return
    def __str__(self):
        return "Expert 4"

class W_MAJ:
    def __init__(self,experts,learning_rate):
        self.experts=experts
        self.learning_rate=learning_rate
        self.weights=dict()
        for x in self.experts:
            self.weights.update({x:1})
        self.current_action=None
        self.current_prediction=list()
    def step(self):

        self.current_prediction=[]
        for x in self.experts:
            self.current_prediction.append(x.step())
        w_pos=0
        w_neg=0
        for x in range(len(self.current_prediction)):
            if self.current_prediction[x]==1:
                w_pos=w_pos+self.weights[self.experts[x]]
            else:
                w_neg=w_neg+self.weights[self.experts[x]]
        if w_pos>=w_neg:
            self.current_action=1
            return 1
        else:
            self.current_action=0
            return 0
    def reset(self):
        for x in self.experts:
            self.weights.update({x:1})
        return
    def update(self,new_observation):

        for x in self.experts:
            x.update(new_observation)
        if self.current_action==new_observation:
            return
        else:
            for x in range(len(self.current_prediction)):
                if self.current_prediction[x]!=new_observation:
                    self.weights.update({self.experts[x]:self.weights[self.experts[x]]-self.learning_rate*self.weights[self.experts[x]]})

    def __str__(self):
        return "W MAJ"

def Bound_test(W_MAJ_mistakes,best_mistakes):
    m=4
    learning_rate=0.1
    return (W_MAJ_mistakes<=2*(1+learning_rate)*best_mistakes+(np.log(m)/learning_rate))
class play_ground:
    def __init__(self,Learners,ADV,num_of_instance=4,T=1000):
        self.Learners=Learners
        self.ADV=ADV
        self.N=num_of_instance
        self.N_list=list([])
        self.T=T 
    def step(self):
        container=np.zeros((self.T,len(self.Learners)))

        self.ADV.reset()
        for x in self.Learners:
            x.reset()

        learners_prediction=dict({})

        for x in self.Learners:
            learners_prediction.update({x:0})

        state=np.zeros(len((self.Learners)))
        current=np.zeros(len((self.Learners)))

        for i in range(self.T):

            for x in self.Learners:
                learners_prediction.update({x:x.step()})

            new_observation=self.ADV.step()
            

            for x in range(len(self.Learners)):
                if learners_prediction[self.Learners[x]]!=new_observation:
                    current[x]=1
                else:
                    current[x]=0
            state=state+current
            container[i]=state
            control_1=Bound_test(state[0],min(state[1:5]))
            if control_1 is False:
                print("Problem")

            for x in self.Learners:
                x.update(new_observation)
        return container            
def plot(Obj):
    fig, ax = plt.subplots(2,2)
    fig.set_size_inches(12.5, 8.5, forward=True)

    plt.setp(ax, xticks=np.linspace(0,1000,5),yticks=np.linspace(0,600,11))
    


    Mat_1=Obj.step()
    # plt.plot(range(Obj.T),Mat_1.transpose()[0], 'r',label="ALG")
    ax[0, 0].plot(range(Obj.T),Mat_1.transpose()[0], 'r',label="ALG") #row=0, col=0
    ax[0, 0].plot(range(Obj.T),Mat_1.transpose()[1], 'b',label="Expert 1")
    ax[0, 0].plot(range(Obj.T),Mat_1.transpose()[2], 'g',label="Expert 2")
    ax[0, 0].plot(range(Obj.T),Mat_1.transpose()[3], 'darkorange',label="Expert 3")
    ax[0, 0].plot(range(Obj.T),Mat_1.transpose()[4], 'k',label="Expert 4")


    Mat_1=Obj.step()
    # plt.plot(range(Obj.T),Mat_1.transpose()[0], 'r',label="ALG")
    ax[0, 1].plot(range(Obj.T),Mat_1.transpose()[0], 'r',label="ALG") #row=0, col=0
    ax[0, 1].plot(range(Obj.T),Mat_1.transpose()[1], 'b',label="Expert 1")
    ax[0, 1].plot(range(Obj.T),Mat_1.transpose()[2], 'g',label="Expert 2")
    ax[0, 1].plot(range(Obj.T),Mat_1.transpose()[3], 'darkorange',label="Expert 3")
    ax[0, 1].plot(range(Obj.T),Mat_1.transpose()[4], 'k',label="Expert 4")

    Mat_1=Obj.step()
    # plt.plot(range(Obj.T),Mat_1.transpose()[0], 'r',label="ALG")
    ax[1, 0].plot(range(Obj.T),Mat_1.transpose()[0], 'r',label="ALG") #row=0, col=0
    ax[1, 0].plot(range(Obj.T),Mat_1.transpose()[1], 'b',label="Expert 1")
    ax[1, 0].plot(range(Obj.T),Mat_1.transpose()[2], 'g',label="Expert 2")
    ax[1, 0].plot(range(Obj.T),Mat_1.transpose()[3], 'darkorange',label="Expert 3")
    ax[1, 0].plot(range(Obj.T),Mat_1.transpose()[4], 'k',label="Expert 4")

    Mat_1=Obj.step()
    # plt.plot(range(Obj.T),Mat_1.transpose()[0], 'r',label="ALG")
    ax[1, 1].plot(range(Obj.T),Mat_1.transpose()[0], 'r',label="ALG") #row=0, col=0
    ax[1, 1].plot(range(Obj.T),Mat_1.transpose()[1], 'b',label="Expert 1")
    ax[1, 1].plot(range(Obj.T),Mat_1.transpose()[2], 'g',label="Expert 2")
    ax[1, 1].plot(range(Obj.T),Mat_1.transpose()[3], 'darkorange',label="Expert 3")
    ax[1, 1].plot(range(Obj.T),Mat_1.transpose()[4], 'k',label="Expert 4")

    ax[0,0].set(xlabel="T", ylabel="Number of Mistakes")
    ax[0,1].set(xlabel="T", ylabel="Number of Mistakes")
    ax[1,0].set(xlabel="T", ylabel="Number of Mistakes")
    ax[1,1].set(xlabel="T", ylabel="Number of Mistakes")

    handles, labels = ax[1,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.suptitle("Comparison Chart")

    plt.show()
    return

def main():
    trans_matrix=np.array([[0.95,.05],[0.05,0.95]])
    state=[0.2,.8]
    learning_rate=0.1
    MC=Markov_chain(state,trans_matrix)
    ADV=Bernoulli_gen(MC)
    Expert_list=[Expert_one(),Expert_two(),Expert_three(),Expert_four()]
    EXP_1=Expert_one()
    EXP_2=Expert_two()
    EXP_3=Expert_three()
    EXP_4=Expert_four()
    ALG=W_MAJ(Expert_list,learning_rate)
    Learners=[ALG,EXP_1,EXP_2,EXP_3,EXP_4]
    Obj=play_ground(Learners,ADV=ADV)
    plot(Obj)
    return

if __name__=='__main__':
    main()
