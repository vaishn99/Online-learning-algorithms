# Online-learning-algorithms
### This repo contains the implementation of some of the online learning algorithms.   <br />
### This was done as part of Online prediction and learning (E1 245) course @ IISc   <br />
The following following problem statements are solved: <br />
## Weighted  Majority  algorithm <br />
Q1.  <br />
Implement  the **Weighted  Majority  algorithm**  for  bit  sequence  prediction  with  expert  ad-vice. Generate the (true) outcome sequencey1,y2,...as follows: each yt is a sample from aBernoulli probability distribution with parameterxt, wherex1,x2,...is a time-homogenous discrete-time Markov chain on the state space{0.2,0.8} and transition probabilities. <br /><br /><br />
  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;                  P[xt+1=0.2|xt=0.8] =P[xt+1=0.8|xt=0.2] =0.05 <br /><br />
(Basically, the Markov chain stays in a state for a long time, and depending on the state theoutcome process is mostly 0’s or mostly 1’s)
> The above Data generating process is a Hidden Markov Model(aka HMM). <br />

Code up the following experts:  
                                 (i) Expert 1 always recommends 0    <br />
                                (ii) Expert 2 always recommends 1      <br />
                                (iii) Expert 3 recommends the majority among all outcomes seen so far .                                 
                                (iv) Expert 4 recommends the majority among all outcomes seen in the most recent w=10time slots.   <br />
                                
 Run the algorithm,  with a suitable choice of learning rateεand the above experts,  for atime horizonT=1000, over 4 independent realizations of the outcome sequence. For eachrun, plot the running cumulative mistake count of all the experts together with that of thealgorithm, i.e., you should make 4 plots with each plot’s X-axis being rounds 1 to Tand 5cumulative mistake count curves (4 for the experts and 1 for the algorithm).


 ## Online Active and Lazy Projected Gradient Descent.<br />
 Q2.
<br />
Consider the decision set K:={w∈R3:∥w∥2≤1}(the unit ball for the Euclidean norm).<br />
Construct  a  sequence  of  linear  loss  functions,  each  of  whose  coefficients  are  iid  normal(Gaussian) random variables with mean 1 and variance 1. Run the following algorithms for T=10^4
time steps on this loss function sequence and plot the cumulative loss of thealgorithm over time in each case:<br />
(a)  **Lazy projected online gradient descent** with constant step sizeη=0.01.  <br />
(b)  **Lazy projected online gradient descent** with time-varying step-sizesηt=1/√t at each time slot.  <br />
(c)  **Active projected online gradient descent** with constant step sizeη=0.01.  <br />
(d)  **Active projected online gradient descent** with time-varying step-sizesηt=1/√t at each time slot.  <br />
