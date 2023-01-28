###  1. Steepest ascet
  
- Design space, design radius, center points
  **Design space:** the region inside the factorial points
  **center points:** origin (0, 0) in coded units. Model's prediction is best at the center point.
  **design radius:** measure how far away from the center point. 
- First-order response surfaces
  Approximate the response surface with a plane. 
  ![fo_response_surface.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3by4keq1j20bu0byaap.jpg )
- Finding direction of steepest ascent
  Compute partial derivatives along each factor. The rate of ascent along each direction is the effect size <img src="https://latex.codecogs.com/gif.latex?&#x5C;beta_i"/>.
- Standardized step sizes
  ![standardize_steepest_ascent.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3c2vdxgsj20o90fuwgt.jpg )
  Why standardize: 1) uniform steps give uniform differences in design radii; 2) a standardized step of 1 always defines a point on the design space boundary.
- Testing for pure error and lack of fit
  **Pure error**: standard deviation of the repeated runs at the design center. 
  **Lack of fit**:  
  ![lack_of_fit.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3caz8xxej20li0bcq4e.jpg )
  
###  2. RSM
  
- Quadratic response models
  ![quadratic_response_model.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3cgcmxlcj20kv05h0t8.jpg )
- Central composite designs (CCDs)
  ![ccd.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3ciaq8tvj20o10iagog.jpg )
- Uniform precision
  The variance at design radius 1 is the same as the center. Choose correct number of the center points to ensure uniform precision.
- Rotatable designs
  Designs where the variance only depends on the radius.
  The change in precision should be independent of the direction moving away from the center. 
  CCD with F factorial points is rotatable when <img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha=F^{1&#x2F;4}"/>
  ![ccd-rotate.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq5qrjqzehj20ok0flacg.jpg )
- Calculating factor levels
  ![factor_level.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3cpa3l52j20be052gls.jpg )
- Finding stationary points
  The argmin, argmax, or inflection point of a saddle.
  ![stationary_point.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3croiag6j20hw07v74v.jpg )
  ![stationary_point_b.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3cshpmcuj20kl05l3z1.jpg )
- Responses at the stationary point
  ![stationary_point_response.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3ctl6fgmj20ng07xt9f.jpg )
- Testing for maxima/minima/saddle points
  ![stationary_point_testing.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3cv8260ej20m004j74u.jpg )
- Alternative designs: BBD, Hoke, Koshal, Roquemore Hybrid, SCD, DSD
  ![design_summary.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3d67d2o9j20nl09pacc.jpg )
  **Box-Behnken design:**
  ![bbd.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3cxgkc6fj20nx0h3tai.jpg )
  All points are on the edges and <img src="https://latex.codecogs.com/gif.latex?&#x5C;sqrt{2}"/> away from the design center when k=3.
  It's not good at predicting response near the corners.
  
  **Hoke Design**
  D2 (10 runs) and D6 (13 runs) are most popular designs. 
  ![hoke.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3d1zaahtj20fx0b8dg2.jpg )
  
  **Requemore hybrid designs:**
  near-saturated and near-rotatable
  ![requemore.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3d2wzg7hj20mm0cl0ug.jpg )
  
  **Small composite design (SCD)**
  Resolution <img src="https://latex.codecogs.com/gif.latex?III^*"/> design: resolutoin III with no 4-letter word in the defining relation.
  High variance for main effects and TWI terms. 
  ![scd.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3d5hek2xj20670bot8r.jpg )
  
  **Definitive screening designs (DSD)**
  ![dsd.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3d8btzpoj20o40etmzm.jpg )
  
###  3. Mixtures
  
- Slack variable and Scheffe models
  Mixture: k components, each with proportion <img src="https://latex.codecogs.com/gif.latex?x_i"/>, where <img src="https://latex.codecogs.com/gif.latex?0&#x5C;le%20x_i%20&#x5C;le%201"/> and <img src="https://latex.codecogs.com/gif.latex?&#x5C;sum_{x_i}=1"/>
  ![slack_variable.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3ddbdsu2j20od0eiacg.jpg )
  
  ![scheffe.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3de8wrw8j20nr0d3wh8.jpg )
- SLD vs. SCD
  ![sld.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3dfnhxvhj20ph0k4whk.jpg )
  ![scd.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3dhtp4c0j20pt0jan09.jpg )
  SLD allows fitting an <img src="https://latex.codecogs.com/gif.latex?m^{th}"/> order model. SCD fit k-order model with up to k-way interaction term.
  In SCD, no single ingredient is run at a proportion > 1/2.
  SLD has better coverage of the boundary, while SCD has better coverage of the interior. 
  
###  4. Crossover Designs
  
- Motivation for crossover designs
  When individual experimental units are rare or expensive. The number of treatment levels and subjects is small. 
- Washout, carryover
  Carryover effect: Treatments are sequential, so the effects can persist into the next treatment. 
  2 way to deal with: 1) allow a washout period between treatments (easy but costly); 2) include carryover effects in the model (difficult)
- Blocking in crossover designs
  ![cod.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3dtt3ttfj20pb0ja78c.jpg )
  
###  5. Surrogate Optimization
  
![surrogate_optim.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3xl1ob5cj20n404fgly.jpg )
![surrogate_optim_workflow.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq426j2qmvj20o00840up.jpg )
- Global vs. local optimization
  Steepest ascent/RSM finds the optimal operating conditions in a local design space. Global optimization searches the entire design region.
- Latin Hypercube Designs
  Evenly-spaced design:
  ![even-space.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3xu2jffaj20oz0c00u4.jpg )
  Random designs are clumpy.
  Latin hypercube design:
  ![latin-design.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3xx1lgbvj20p60cu401.jpg )
  ![orthogonal_lhd.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3xze5vg9j208v08smx4.jpg )
  - placed points semi-randomly to avoid aliasing
  - avoid clumps of points
  - project well onto lower dimensions
  Orthogonal array LHD: 
- Maximin Designs
  Maximize the distance between points.
  ![maxmin.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3y1mpoyuj20ib07haax.jpg )
- Gaussian Process Regression
  GPR assumes the covariance between the data have a particular shape. The covariance function is called the kernel. 
  ![gpr_prediction.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3ycukoy2j20nh0bcgnl.jpg )
  GRP limitations:
  ![gpr_limit.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3yh93ol0j20nf07pgno.jpg )
  GPR is bayesian: the kernel (prior) is updated with data <img src="https://latex.codecogs.com/gif.latex?(X_n,y_n)"/> to compute the posterior estimates of <img src="https://latex.codecogs.com/gif.latex?(x,%20y)"/>. But it's nonparametric since the equation for <img src="https://latex.codecogs.com/gif.latex?y(x)"/> and <img src="https://latex.codecogs.com/gif.latex?&#x5C;sigma^2(x)"/> don't contain parameters.
  - Kernels
  ![kernel.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3y698lowj20ms0cx0tw.jpg )
  - Hyperparameters: scale, nugget, lengthscale
  **Scale**
  ![scale.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3yqad4n1j20nq0brtaj.jpg )
  **Nugget**
  ![nugget.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3ysc9xu6j20ng07mjsu.jpg )
  **Lengthscale**
  ![lengthscale.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq3ywsf00hj20nf0edq4l.jpg )
  Dimensions with longer lengthscales require fewer data for prediction.
  
  Putting all 3 factors together. Need to compute inverse of <img src="https://latex.codecogs.com/gif.latex?C_n"/> at each iteration. computational expensive. 
- Sequential design using mean, variance, and expected improvement
  Use the point where mean/variance/EI is biggest as next input data.
  Sequential design methods are last sample optimal. But sequential design is greedy.If <img src="https://latex.codecogs.com/gif.latex?N−2"/> of N runs are finished, two rounds of sequential design may not be optimal. It suffers limited lookahead. 
  Expected improvement: 
  ![expected_improve.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq420mijllj20nh0bbdhi.jpg )
  ![compute_ei.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4232tksrj20o70argn9.jpg )
  
- Exploration vs. exploitation
  Exploration searches areas of high uncertainty to find new regions of interest.
  Exploitation refines existing optima by adding points to known regions of interest. 
  Explore early, exploit later. 
  Alternate between batches of exploration and exploitation.
  **Exploit by maximizing the predicted GPR mean.**
  **Explore by maximizing the predicted GPR standard deviation**
  
  
###  6. Reinforcement Learning
  
Learning from trial and error. Many RL algorithms rely on random processes to generate data.  
- Markov Decision Processes: state, action, policy, reward, trajectory, episode
  Describes how agent interacts with its environment.
  **State:** the agent and environment at any time
  **Action**: the agent selects an action to move between states.
  **Reward:** every action and state produce a reward.
  The agent goal is to maximize the total reward.
  MDP has Markov property: 1) all decisions depend only on the current state. 2) each state includes all of the relevant history.
  **Policy**: a function that maps states to actions. The value <img src="https://latex.codecogs.com/gif.latex?&#x5C;pi(s,a)"/> is the probability that the agent will select action <img src="https://latex.codecogs.com/gif.latex?a"/> in state <img src="https://latex.codecogs.com/gif.latex?s"/>.
  MDP can be deterministic or stochastic. Deterministic: actions always determine the next state. Stochastic: action change the probability that any other state will be the next state.
  **Trajectory:** a single pass through a finite horizon MDP. Finite horizon MDP or episodic MDP stops after a finite number of actions.
- Value functions
  The value of a state is the expected reward from that state to the end of the trajectory. <img src="https://latex.codecogs.com/gif.latex?V(s_i)=&#x5C;mathbb{E}(&#x5C;sum%20r_k)=&#x5C;mathbb{E}(R_i)"/>. 
  A trajectory is a sequence of states, actions and rewards. Its length can vary for every trajectory. No action in terminal state, but there can be a terminal reward.  
  ![value_function.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4dv2wrn5j20pi0j8q5i.jpg )
  Every-visit vs last-visit: 
  Gridworld is deterministic and the agent should never visit the same state twice. The last-visit estimate is closest to optimal. 
  Stochastic problems can revisit the same state under the optimal policy. 
  Acting greedy w.r.t a value function is optimal. <img src="https://latex.codecogs.com/gif.latex?V(s_i)=&#x5C;mathbb{E}(r_i+..+r_T)&#x5C;rightarrow%20maxV(s_i)=max_{a_i}&#x5C;mathbb{E}{(r_i)}+V(s_{i+1})"/>. So the optimal policy at state <img src="https://latex.codecogs.com/gif.latex?s_i"/> satisfies <img src="https://latex.codecogs.com/gif.latex?max_{a_i}V(s_{i+1})"/>
  Pure Monte Carlo is inefficient. 
  ![policy_iteration.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4ea39o66j20no068q45.jpg )
  policy iteration is guaranteed to find optimal policy provided every state is visited an infinite number of times. But tabular methods require visiting every state.
  ![policy_eval.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4ef31guuj20md06lq45.jpg )
- Rollout
  V is rarely known. 
  ![noknow_v.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4eho3fdnj20n403p0t9.jpg )
  Ways to approximate value functions:
  ![approx_value_func.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4ek62j4nj20ne0aa417.jpg )
  Rollout is a Monte Carlo method, which looks ahead to estimate the value of states the agent is likely to visit next. 
  ![rollout.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4enh34ckj20o10azq43.jpg )
  policy improvement with rollout: 
  ![rollout_improve.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4eq3xb7xj20nq083jss.jpg )
  Rollout is an online method that reduces simulation by focusing on local starts. Iteration and exploration are required to find optimal policies. 
- Discount factors
  ![discounting.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4evc4gk4j20nn0dx0uy.jpg )
  ![discount-gridworld.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4exve39mj20n40du0u0.jpg )
  When to use discounting?
  - don't want to discount the future rewards, set <img src="https://latex.codecogs.com/gif.latex?&#x5C;gamma=1"/>
  - compare with greedy algorithm, set <img src="https://latex.codecogs.com/gif.latex?&#x5C;gamma=0"/>
  - want the agent to terminate the process quickly or solve non-episodic problems, set <img src="https://latex.codecogs.com/gif.latex?&#x5C;gamma&lt;1"/>
  Model-free learning: directly learn from experience, no need a model to simulate ahead when estimating value functions. Their only method of sampling is to interact with the environment, maximizing the information extracted from every trajectory. 
- TD learning for value functions
  Ideally, update estimate of value function from every trajectory, but a single trajectory is a noisy estimate of value. 
  Temporal difference (TD) learning balances new experience with previous results when updating V. 
  ![td-learning.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4fbmtdlyj20na0fgacr.jpg )
- Q-factors
  Q-factors: Learn the value of each state/action pair, because we don't know <img src="https://latex.codecogs.com/gif.latex?s_{i+1}"/> given <img src="https://latex.codecogs.com/gif.latex?s_i"/> and <img src="https://latex.codecogs.com/gif.latex?a"/>. 
  ![learn_q.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4fnhf7lqj20mi0g3q56.jpg )
  The number of Q-factors is much greater than the number of states.
- SARSA, Q-learning, and Double Q-learning
  **SARSA**: Learn Q-factors using a TD approach. 
  ![sarsa.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4frpvo8uj20lq09emye.jpg )
  The policy that generates the trajectory is not optimal, so <img src="https://latex.codecogs.com/gif.latex?a_{i+1}"/> is not the best action. 
  
  **Q-learning**: 
  ![q-learning.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4fuj6vtjj20nk0c5abz.jpg )
  Any algorithm with a max operator will drift upwards over time, even if the mean value remains fixed. 
  
  **Double Q-learning**:
  ![double_q.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4fyq09axj20n10ggmzr.jpg )
  
- Neural networks: neurons, layers, activation functions, width, depth, loss, stochastic gradient
descent
**Neuron**: connects inputs to an output. If combined input exceeds a threshold, the output fires. It's a linear classifier. 
**Activation functions**: nonlinear function. sign/step function, sigmoid function, rectified linear unit activation.
![nn.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4gd4ijlgj20i009bjs4.jpg )
**Width&Depth**:
![nn_form.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4ghtf852j20mv08o75q.jpg )
The number <img src="https://latex.codecogs.com/gif.latex?d"/> is the depth of the neural network. Deep learning means <img src="https://latex.codecogs.com/gif.latex?d&gt;2"/>.
The importance of nonlinearity: without nonlinear activation function, the network reduces to a single linear system. 
  
Universal approximation theorem states given enough neurons, a 2-layer perception can learn any reasonable function. 
Deep networks learn more efficient than wide ones. It reduces the total number of neurons needed to learn a neuron since each of the d layers needs fewer than 1/d the number of neurons. 
Why deeper networks learn better? Each layer only needs to improve the features for the next layer. 
  
**Loss**: measures how well the output of the final layer compared with known training data. 
  
Gradient descent: update weights by 1) compute the gradient of the total loss <img src="https://latex.codecogs.com/gif.latex?g(W)=&#x5C;sum&#x5C;cfrac{&#x5C;partial%20L_i(W)}{&#x5C;partial%20W}"/>; 2) update the weights using <img src="https://latex.codecogs.com/gif.latex?W^{(1)}=W^{(0)}-&#x5C;alpha%20g(W^{(0)})"/>; 3) repeat previous steps.
But it has problems in gradient calculation.
  - The gradient has one entry for each parameter, of which there are thousands!
  - These computations are repeated over all N points in the dataset for each iteration.
  
**SGD**: alternates between forward and backward passes through the model and updates parameters using the loss for a single point.
![gd.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4gzv3ga5j20ml0efgnp.jpg ) 
One pass through the entire training set is called an epoch. One epoch is not enough and the order of the training data is randomized between epochs. 
Minibatches: average a small number of training samples before updating the weights.
SGD is stochastic which is a form of regularization. 
Why neural networks learn so well?
![nn_nolocal.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4h3zelhhj20l2043gma.jpg )
How to improve nn?
![nn_improve.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4h5gq1tkj20n707wwgb.jpg )
- Deep Q-learning
  Approximate Q-factors via deep learning with artificial neural networks. 
  Define state space: onehot-encoding (16x4 matrix) vs ignore block states (12x1 trinary vector) 
  Steps of deep Q-learning: 
  ![deep-q-learning.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4h6ukfilj20ms0dq769.jpg )
  For deep Q-learning with terminal rewards, in the near-terminal state: <img src="https://latex.codecogs.com/gif.latex?&#x5C;hat{Q}(s_{T-1},%20a_{T-1})=r_{T-1}+max_a&#x5C;tilde{Q}(s_T,a)=0+&#x5C;tilde{Q}(s_T,.)=r_T"/>. The reward is bootstrapped back through the Q-factors.
  To speed up learning, set <img src="https://latex.codecogs.com/gif.latex?&#x5C;tilde{Q}(s_i,a_i)&#x5C;approx%20r_T"/>, to reward any state/action pair with a win and penalize state/action pair in losing games.
  ![q_factor.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4hfbvrqtj20mr07875i.jpg ) 
- Policy-based methods and REINFORCE
  ![policy.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4hhj4rjyj20nq09dwg6.jpg )
  To learn policy directly from experience,
  ![policy_learn.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4hkmedpij20ld0cdabi.jpg )
  
  Reinforce algorithm:
  ![reinforce.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gq4hmkqqjtj20o60b3dho.jpg )
  
  