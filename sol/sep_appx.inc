
Below, we give additional details regarding the steps and numerical methods employed in 
each of the interaction variants discussed in the main text. 
%Also, we discuss and classify other related studies w.r.t. three types of interaction to put our work into context. Afterwards,
We present details of the simulation setups for the five scenarios and give more detailed 
results for each case. Lastly, we discuss performance and list details of our neural network architectures.

% feature shift

As our experiments in the main text already demonstrate, 
deep learning algorithms that can closely interact with a differentiable 
PDE solver can yield substantially improved performance.
This illustrates how crucial it is for deep learning algorithms that co-exist 
or interact with numerical solvers in a recurrent manner
to anticipate shifts in the distributions of input 
features. We present additional results and show how 
interactions between PDE solvers and deep neural networks can be formulated. 
These interactions help to bridge the gap between distribution shifts that exist between 
different discretizations of a PDE. 
We will demonstrate that avoiding distribution shifts is essential for a model 
to infer a correction successfully. In our iterative setting, this, in turn, helps 
to keep the distributions aligned over the course of many iterations.

%---

\section{Correction Functions for PDEs}

For completeness, we provide a brief summary of our notation.
We consider reference solutions $\vrN$ of
the PDE $\pde$ that are contained in the phase space manifold
$\manifref$ with reference trajectories over time denoted by
$\{\vr{t},\vr{t+1},\cdots,\vr{t+k}\}$ % Kiwon - having parentheses looks clearer for this. Nils - these are called braces: "{}"
for $k$ steps of size $\dt$.
A more coarsely approximated solution of the same problem is denoted by $\vcN$
in the manifold $\manifsrc$ with trajectories
$\{\vc{t},\vc{t+1},\cdots,\vc{t+k}\}$. % Kiwon - same here
We typically initialize the source state from the reference version 
via a transfer operator $\project$ with $\vc{t} = \project \vr{t}$ as initial condition.
A transfer from source to reference states is denoted by $\project^T$.

The learning objective is to find the best possible correction function 
$\corr (\vcN \, | \, \theta)$ given the weights $\theta$ and a network architecture.
Without loss of generality, we assume that the correction 
function is applied additively, i.e., $\vctN = \vcN + \corr (\vcN \, | \, \theta)$, 
where the tilde in $\vctN$ indicates the corrected state. 
A new state is computed in combination with the PDE 
via $\vct{t+1} = \pdec(\vc{t}) + \corr ( \pdec(\vc{t}) \, | \, \theta)$ for which we
use the short form $(\pdec  \corr ) (\vc{t})$ below. Multiple recurrent 
evaluations of a function are denoted by $\vct{t+k} = ( \pdec \corr )^k ( \vc{t} )$ for $k$ steps
starting from an unaltered source state $\vc{t}$.

For training neural networks, we use an \new{$L^2$-based}
loss, i.e., \new{$\loss (\vct{t},\project \vr{t})=\Vert\vct{t}-\project \vr{t}\Vert^2$},
which is typically evaluated 
for $n$ steps via  $\sum_{i=t}^{t+n} \loss (\vct{i},\vr{i})$ in order to find a 
solution to the minimization problem: $\argmin_{\theta} \sum_{i=t}^{t+n} \loss (\vct{i},\vr{i})$.

%As PDEs, 
We consider constrained advection-diffusion PDEs:
%
$\partial{\vu} / \partial{t} = - \vu \cdot\nabla\vu + \nu \nabla\cdot \nabla \vu + \mathbf{g}$ subject to $\mM \vu = 0$.
%
% \begin{equation}
%   \label{eq:model-adv-diff}
%   \partial{\vu} / \partial{t}
%   = - \vu \cdot\nabla\vu +
%   \nu \nabla\cdot \nabla \vu + \mathbf{g}
%   \quad \text{subject to} \quad \mC \vu = 0 .
% \end{equation}
Here, $\vu$, $\nu$, and $\mathbf{g}$ denote velocity, diffusivity, and external forces, respectively.
The constraint matrix $\mM$ contains an additional set of equality constraints imposed on $\vu$.


% \vc{t+n} = \pdec(\pdec(\cdots \pdec( \project \vr{t}  )\cdots)) = \pdec^n ( \project \vr{t} ), 
% %
% Our learning goal is to arrive at a correction operator $\corr ( \vc{} )$ such that 
% a solution to which the correction is applied has a lower error than a unmodified solution: 
% $\loss ( \pdec( \corr (\project \vr{t_0}) ) , \project \vr{t_1}) < \loss ( \pdec( \project \vr{t_0} ), \project \vr{t_1})$. 
% The correction function 
% $\corr (\vcN | \theta)$ is represented as a deep neural network with weights $\theta$
% and receives the state $\vcN$ to infer an additive correction field with the same dimension.
% We use an exponential notation to indicate a recursive application of a function, i.e.,
%   \begin{equation}
%     \vc{t+n} = \pdec(\pdec(\cdots \pdec( \project \vr{t}  )\cdots)) = \pdec^n ( \project \vr{t} ), 
%   \end{equation}
% %As these interactions modify the 
% and, to distinguish the original phase states $\vcN$ from corrected ones, we 
% denote the latter with $\vctN$.

\subsection{Learning Without Interaction}

In the main text, we use learning via non-interacting trajectories 
as a baseline learning setup. In this case, a model is 
trained to minimize differences between states $\vcN$ and $\vrN$ in a fully supervised 
manner. These versions are denoted by NON.

Despite its simplicity, different variants of this learning setup can be considered.
In the simplest case, we initialize the source simulation from the corresponding reference
version, evaluate the PDE once, and then train a model via a large number of such cases.
In our notation, this means learning from states computed as 
$\vc{t+1} = \pdec( \project \vr{t} )$. This effectively takes into account only
a single evaluation of the source PDE, and a model can only learn from numerical 
differences that build up within this single step. Hence, a variant of this 
approach is to allow reference and target version to evolve over the course of 
multiple steps such that the errors in the source states $\vcN$ show up more clearly 
with respect to $\vrN$. Similar to the look-ahead discussed in the main text, 
we can use $\vc{t+n} = \pdec^n ( \project \vr{t} )$ as a training data set. 
We denote such versions that have no interaction but consider multiple steps of unaltered 
coarse evolution as \non{n} below. Note that the previously discussed NON version 
could be denoted by \non{1}, but we keep the label NON for consistency with 
the main text in the following.

For all choices of $n$, we obtain the following minimization problem for learning via 
non-interacting solvers:
\begin{equation} \label{eq:non-min}
\new{\argmin_{\theta} \sum_{i=0}^{n} \Vert \vc{t+i} + \corr ( \vc{t+i} \, | \, \theta) - \project \vr{t+i} \Vert^2}.
\end{equation}

Another non-interacting variant could be trained by reversing the 
setup above and initializing reference trajectories from source states,
i.e., $\vr{t+n} = \pder^n ( \project^T \vc{t} )$. 
Like before, a model could be trained in a supervised fashion from a data set of $\vcN$
and $\vrN$ states computed in this way. However, as the interesting structures that 
make up the reference solutions typically take very long time spans to form (if they are 
achievable at all), this variant is clearly sub-optimal. Hence, due to the poor performance 
of the \non{n} versions, we have not included this reversed NON variant in our experiments.

The NON models presented in the main text so far already 
allow for a first quantification of the problems caused by the distribution shifts 
of the input features: across the two-dimensional 
% (2D) - I think having this is more formal, but because we decide not to do for 3D, comment out for now
fluid flow cases, the unaltered source simulations deviate by more 
than 50\% in terms of MAE from the corrected simulations. 
%
This means that, after applying the corrections, the model receives 
inputs that strongly differ from those seen at training time.
In terms of content of the input feature vectors,
the MAE measurements show a change of over 50\%. 
Nonetheless, we expect the model to reconstruct 
the reference states despite receiving inputs that are significantly different
from the inputs seen at the time of training.
Not surprisingly, the models only have limited success achieving this goal.

% numbers from main graphs:
% karman2d: base 0.14  non .045  -> 3.18
% buoy:     base 1.60  non 1.10     1.45
% [burgers:  base 0.25  non 0.16     1.56]
% 

% ... the mean residual test errors allow for a quantification of these differences: 
% models trained with unaltered version deviate from baseline solution by X\%,
% hence at inference time on average receive inputs that differ by (N-X)\%.
% While regularization such as noise injection can stabilize models in such settings
%



% ---





\subsection{Pre-computed Interactions}

As an improvement over the non-interacting 
versions above, we consider a class of models learning 
from data generated via pre-computed interactions, denoted by PRE.
The pre-computations have the goal of reducing the gap between 
source and reference trajectories. The pre-computation
changes the source trajectories and thus provides the learning optimization 
with modified inputs that are closer to the reference at inference time. This scenario
is common practice, e.g., for weather predictions, where simulations need 
to be aligned with real-world measurements, i.e., {\em data assimilation} algorithms
\cite{jones1997latent,stephan2008assimilation,xi2011automatic}. 
As the data set has to be prepared only once, 
computationally expensive pre-computation is often still feasible
as this overhead will not influence the performance at 
inference time. However, in the context of machine learning, pre-computed corrections 
can only provide limited improvements as the correction during the pre-computation 
phase can only partially mimic the behavior of the actual, learned version.

%as these methods have to resort to manually constructed heuristics during the pre-computation phase, they compared to the non-interacting versions in our scenarios.

For PRE models, two correction functions are used: 
one for preparing the training data set denoted by $\corrPre$ and the learned 
correction $\corr$. The training data set is computed as 
$\vct{t+n} = (\pdec \corrPre)^n ( \project \vr{t} )$, where $n$ denotes 
the number of steps for independent simulation trajectories in the source and reference 
manifolds. 
%
Note that, in this context, due to the corrections being applied at 
the time of data generation, there is hope for longer unrolling periods (i.e., larger $n$)
to have a positive effect on the learning outcome (in contrast to the \non{n} versions above).
%
% A model $\corr (\vcN | \theta)$ is trained via the pre-computed data as before:
% \begin{equation}
% \argmin_{\theta} \sum_{i=0}^{n} \Vert \vct{t+i} + \corr ( \pdec(\vct{t+i}) | \theta) - \project \vr{t+i} \Vert^2
% \end{equation}
%set as outlined above.
At inference time, $\corrPre$ is no longer used, and trajectories are instead 
computed as $\vct{t+n} = ( \pdec \corr )^n ( \vc{t} )$, in line with the NON variants.
%
Hence, in total, four versions of a trajectory from a single initial phase space point $\vr{t}$
exist: a source trajectory, a source trajectory corrected by pre-computation via $\corrPre$, 
a source trajectory corrected by the learned correction function $\corr$, and the reference trajectory.

We first describe how to include a pre-computation correction 
for spatial corrections while taking into account simulation constraints
before including the temporal dimension. 
% cf eg: http://www.ws.binghamton.edu/fowler/fowler%20personal%20page/EE522_files/EECE%20522%20Notes_11%20Ch_6.pdf
For both, we adopt a constrained version of
{\em best linear unbiased estimates} \cite{henderson1975best}, which are widely used for data assimilation.


\subsubsection{Pre-computed Spatial Regularization}\label{sec:space-reg}

For a constraint-aware interpolation that can serve as a correction operator, 
%
consider two vector spaces $\mathbf{R} \in \mathbb{R}^\chi$ and
$\mathbf{S} \in \mathbb{R}^\xi$ 
with different dimensionalities $\xi,\chi \in \mathbb{N}$ with $\xi < \chi$.
Both vector spaces satisfy the constraint $\mM$,
%such as conservation of volume (following \myeqref{eq:model-ns}), 
% i.e., $\nabla\cdot\mathbf{h}=0$ for $\forall \mathbf{h} \in \mathbf{R}$, 
% and $\nabla\cdot\mathbf{l}=0$ for $\forall \mathbf{l} \in \mathbf{L}$. 
i.e., $\mM\mathbf{r}=0$ for $\forall \mathbf{r} \in \mathbf{R}$, 
and $\mM\mathbf{s}=0$ for $\forall \mathbf{s} \in \mathbf{S}$. 
%
Given a finer vector field $\mathbf{c}_R$,
e.g., containing the reference solutions,
%which contains the necessary information, 
we aim to find the closest vector
field $\mathbf{c}_S$ ($\in \mathbf{S}$) to $\mathbf{c}_R$ ($\in \mathbf{R}$). 
% We first describe how to transfer functions between both spaces in general before
% explaining how to find those that are particularly amenable for learning.
Consider an interpolation operator $\mW$ that introduces new data points
within a vector field $\mathbf{c}_S$ ($\in \mathbf{S}$), i.e.,
$\mW\mathbf{c}_S \in \mathbb{R}^\chi$. We, then, strive to minimize the
distance between $\mW\mathbf{c}_S$ and $\mathbf{c}_R$ such that
$\mathbf{c}_S$ can best represent the information of $\mathbf{c}_R$ without
violating the constraints.
%loosing its volume conserving properties. 
Thus, we aim for computing $\mathbf{c}_S$ with
\begin{equation}
  \label{eq:objective}
  \argmin_{\mathbf{c}_S} || \mW\mathbf{c}_S - \mathbf{c}_R ||^2 \quad
  \text{subject to}\quad \mM \mathbf{c}_S = 0.
\end{equation}
This represents a constrained optimization problem with equality constraints,
which we can solve via Lagrange multipliers $\mathbf{\lambda}$ as follows:
\begin{equation}
  \label{eq:lm-objective}
  \Phi = ||\mW\mathbf{c}_S - \mathbf{c}_R ||^2 +
  ( \mM \mathbf{c}_S)^{\top}\mathbf{\lambda}.
\end{equation}
This results in a system of equations:
\begin{equation}
  \label{eq:ls}
  \begin{bmatrix}
    \mW^{\top}\mW                                         &  -\mM  \\
    -\mM^{\top}                                           &  0
  \end{bmatrix}
  \begin{bmatrix}
    \mathbf{c}_S                                                    \\
    \mathbf{\lambda}
  \end{bmatrix}
  =
  \begin{bmatrix}
    \mW^{\top}\mathbf{c}_R                                         \\
    0
  \end{bmatrix}.
\end{equation}
Using the Schur complement, we can simplify this system to speed up
calculations:
\begin{eqnarray}
  \mM^{\top}(\mW^{\top}\mW)^{-1}\mM\mathbf{\lambda} & = & 
  \mM^{\top}(\mW^{\top}\mW)^{-1}\mW^{\top}\mathbf{c}_R , \\ 
  \label{eq:schur-sol}
  \mathbf{c}_S                                      & = & 
  (\mW^{\top}\mW)^{-1}(\mW^{\top}\mathbf{c}_R - \mM\mathbf{\lambda}).
\end{eqnarray}
% \Iac{nn}, then, infers the correction $\mathbf{\hat{c}}_{L}$ by minimizing the
% supervised loss,
% $L_{\text{sup}} = \sum||\mathbf{\hat{c}}_{L} - \mathbf{c}_{L}||$, for a pre-computed data set. 
%

In our setting, given source states $\vcN$ and reference states $\vrN$, 
we can thus compute a correction vector field via 
$\mathbf{c}_{t} = (\mW^{\top}\mW)^{-1}(\mW^{\top} (\vr{t} - \mW\vc{t}) - \mM\mathbf{\lambda})$,
e.g., using $\mM = (\nabla \cdot)$ for Navier-Stokes scenarios.
In order to train a model $\corr (\vcN \, | \, \theta)$ to infer the corrections, 
we can directly use the pre-computed correction vectors:
\begin{equation} \label{eq:argminPreLm}
\new{\argmin_{\theta} \sum_{i=0}^{n} \Vert \mathbf{c}_{t+i} - \corr ( \vct{t+i} | \theta)  \Vert^2}.
\end{equation}
%where $\mathbf{c}_{S,t+i}$ denotes the pre-computed correction at time $t+i$ for $\vct{t+i}$ and $\vr{t+i}$.
%We will denote versions using this pre-computation scheme as \pre{SR}.
% Kiwon - in the main text, PRE denotes "with spatiotemporal regularization"
We will denote versions using this pre-computation scheme for $\corrPre$ with
spatial regularization as \pre{SR}.

\subsubsection{Pre-computed Spatiotemporal Regularization}\label{sec:temp-reg}

The vector fields we target are obtained from a numerical simulation, 
where the underlying PDE is solved for a finite number of steps from an
initial condition. 
%
%As our aim is to find a learned representation of the corrections, 
In the context of deep learning, an important
aspect to consider is the sensitivity \cite{murphy2004quantification} of the
targeted function (i.e., the correction) with respect to the data at hand, i.e.,
in our case, the state of a source simulation. 
% Kiwon - Do we demonstrate? maybe remove or discuss why the temporal regularization can help. Nils - I hope we can show some results there...
%We will demonstrate below that the 
The pre-computation process described in the previous section is typically done on a per-time-step basis,
and hence correction vector fields can vary significantly even for smooth changes of the
source simulation. That means the correction function can have a very nonlinear and
difficult to learn relationship with the observable data in a simulation.

In order to address this difficulty, we include 
a temporal regularization by limiting the changes over time for each sample point in space.
Consequently, we regularize our correction vector fields such that
they change smoothly in time by penalizing temporal change of the correction
vector field within the Lagrange multiplier framework. We minimize $d\mathbf{c}_S/dt$
together with the constrained transfer from fine to coarse discretizations:
\begin{equation}
  \label{eq:new-objective}
  \argmin_{\mathbf{c}_S}
   \left( || \mW\mathbf{c}_S - \mathbf{c}_R ||^2 + \beta ||
    \frac{d\mathbf{c}_S}{dt} ||^2 \right)\quad
  \text{subject to}\quad \mM \mathbf{c}_S = 0.
\end{equation}
Here, $\beta$ is the temporal regularization coefficient. %The typical forward Euler 
A finite difference approximation of the temporal derivative of the correction field, 
i.e., $d\mathbf{c}_S/dt$, yields the following system of equations:
\begin{equation}
  \label{eq:ls-new}
  \begin{bmatrix}
    \mW^{\top}\mW + \beta\frac{2}{\Delta t}\mathbf{I} & -\mM \\
    -\mM^{\top}                                       & 0
  \end{bmatrix}
  \begin{bmatrix}
    \mathbf{c}_S                                             \\
    \mathbf{\lambda}
  \end{bmatrix}
  =
  \begin{bmatrix}
    \mW^{\top}\mathbf{c}_R + \beta\frac{2}{\Delta t}\mathbf{c}_S^{t-1} \\
    0
  \end{bmatrix},
\end{equation}
where $\Delta{t}$ is the time step size, $\mathbf{I}$ is the identity matrix, and
$\mathbf{c}_S^{t-1}$ denotes the correction vector field evaluated
at the previous time step.
Following \myeqref{eq:argminPreLm}, this data is pre-computed and used for training 
a neural network in a supervised manner.
Models trained with data from this spatiotemporal pre-computation as $\corrPre$ are denoted
by PRE, and we have used a coefficient of $\beta=1.0$
for all PRE models of our submission.
%our experiments and found it effective for both the correction accuracy and training.

\subsection{Solver-in-the-Loop Interactions via Differentiable Physics}

The main goal of training via differentiable physics 
is to bridge the gap that arises from changes in the input data distribution
and directly train with the environment that the learned model is supposed to work 
with at inference time. Hence, the learning process 
aims to solve the minimization problem
\begin{equation} \label{eq:diffPhysTrain}
\new{\argmin_{\theta} \sum_{i=0}^{n-1} \Vert \pdec(\vct{t+i}) + \corr ( \pdec( \vct{t+i} ) | \theta) - \project \vr{t+i+1} \Vert^2},
\end{equation}
where the phase space trajectories are computed via 
$\vct{t+k} = ( \pdec \corr )^k ( \project \vr{t} )$. This formulation 
illustrates that
a cyclic dependency between the corrected 
states $\vctN$ and the learned correction function $\corr$ exists for the ``solver-in-the-loop'' interactions
of this section. As both the deep neural network for $\corr$ and likewise the PDE $\pdec$
are potentially highly non-linear operators, the corresponding coupled minimization problem for 
calculating the weights of $\corr$ is challenging. However, our results clearly show 
that stable optimizations can be achieved in practice and that they lead to very significant 
improvements of the learned representation.

The recurrent training requires differentiable physics solvers that allow for a
back-propagation of gradients through the discretized physical simulation. 
In this work, we employ a differentiable PDE solver from the open source $\Phi_\textrm{Flow}$
library \cite{holl2020}. This solver builds on the automatic differentiation of
the underlying machine learning framework to compute analytic derivatives and augments them
with custom derivatives where necessary. For example, the pressure correction step 
of a Navier-Stokes solver is provided with a custom gradient for performance reasons.
% where built-in derivatives would be inefficient. 
This setup allows for a straightforward integration of solver functionality into 
machine learning models and enables end-to-end training in recurrent settings.
Although all of our examples use the $\Phi_\textrm{Flow}$ solver, we do not leverage 
any special functionality apart from gradients being provided for all steps 
of the PDE solve. Hence, our results should carry over to other types of differentiable 
physics solvers.

It is worth noting that, in the setup discussed so far, 
the reference solver does not need to be differentiable; i.e., the phase 
space points in $\manifref$ could be provided by a black-box approximation
without gradients as long as a differentiable solver for 
the source manifold $\manifsrc$ exists. 
%However, the source solver should be differentiable such that it can be incorporated into the training. 
We demonstrate the split setup
using an external solver for the buoyancy-driven flows below.

Our implementation directly follows \myeqref{eq:diffPhysTrain}. For each mini-batch,
we start with a collection of reference states $\vrN$ for which recurrent trajectories 
of $(\pdec \corr)^n$ are unrolled for $n$ steps. The loss with respect to corresponding 
reference states is computed over all intermediate states of the trajectory.
Back-propagation, then, unrolls
the differences through the sequence of solver steps to update the weights of the 
neural network that provides the correction function.

Under the assumption that the training process converges, this entirely 
removes the problem of distribution shift.
Once the learned correction $\corr$ converges to a steady-state, 
it is trained with exactly the phase state inputs that are produced at inference time.
The MAE of the test data samples again provides a measure of the discrepancies.
Compared to the differences of around 50\% for non-interacting variants (measured 
between source states and corrected states), the deviations grow to 75\% and above for SOL versions.
Nonetheless, even this larger difference in terms of input distributions is 
unproblematic here as the network receives the modified states at training time.
However, we noticed that, during our training runs, the 
final states typically do not fully converge, but still show smaller oscillations in terms of 
performance. While this could be prevented via learning-rate decay, we believe 
the slightly changing states provide robustness similar to dropout or
manual injections of noise \cite{sanchez2020gns}.

\new{While the error accumulates and typically
grows over the course of a full trajectory, our key hypothesis here is that a learned approach can nonetheless identify
and correct a large part of the error function based on information from a single phase-space input. 
For the PDEs we consider, a single state uniquely describes its future evolution.
%Within this assumption, our results demonstrate that a very significant portion is learnable. 
%In our pilot testing, we observed that history information did not yield significance improvements thus were dismissed from our experiments.
We have experimented with additionally providing varying numbers of previous states 
$\mathbf{s}_{t-k}, ... ,\mathbf{s}_{t-1}$ as input to our model. Our 
tests have not shown improvements from these additional states
and indicate that the components of the error function that are learned with our 
approach can be reliably inferred from a single state $\mathbf{s}_t$. 
}

% ... feature shift ... before for NON 50\% gap ...
% now the gap is even larger , 75\% and above, but 
% the model actually sees the data it should work on.
% assuming the training optimization converges, the model receives 
% the exact data it receives at inference time ... ie no more distribution 
% gap between training and inference 


%---



%additional related work, discuss in terms of classes A B C above?
% \subsection{Classification of Existing Methods}\label{app:rw}

% In addition to the discussion of related work in the main text, we provide a more 
% focused categorization of previous work with respect to the interaction 
% methodologies we focus on, i.e., learning via NON, PRE, and SOL interactions.

% \todo{TODO, write and finalize...}

% TODO, discuss nudging as B example? \cite{jones1997latent} , check other works here...?

% cite for finite difference approximations \cite{strikwerda2004finite}?

% discuss reinforcement learning as a variant for environments where no gradients exist...

% pde-net \cite{long2017pde}, "blocks" represent time steps, up to 19, but no real gains above 3 in most cases 
%   -> similar to forced-AD?

% \cite{sirignano2018dgm} 


% ... also needed for e.g.
% weather forecasting over time \cite{rasp2020weatherbench}
% two step training \cite{weyn2020improving};
% Google MetNet without time stepping \cite{sonderby2020metnet}
% %
% data assimilation and pre-computed corrections:
% %
% widely used in weather and geosciences \cite{kalnay1996ncep}
% kalman filters (sequential), ensemble kalman filters \cite{evensen1994sequential},
% general variational 4D minimization,
% reduced order models \cite{hoteit2002simplified}
% back and forth nudging \cite{auroux2005back}
% %
% medical \cite{xi2011automatic,chapelle2013fundamental}
% %
% ours , similar to BLUE, but constrained form w Lagr Mult
% \cite{henderson1975best}


% % more general, diffTaichi \cite{hu2019difftaichi}
% % JAX MD \cite{schoenholz2019jax}
% % julia / zygote ? \cite{innes2019differentiable}

% % DPI net \cite{li2018learning}
% % benjamin \cite{ummenhofer2020lagrangian}
% % more recent \cite{sanchez2020learning}

% %differentiable solvers:
% % junbang minglin, discuss diff cloth physics: \cite{liang2019differentiable}
% % florence CVPR \cite{rasheed2020learning}
% % diff MD sims \cite{wang2020differentiable}


% clarify again - we don't replace solver!

% % also cf https://arxiv.org/pdf/2004.04653.pdf , structure pres: also pendulums, but include couette flow
% % from there: Consider a system whose governing variables will be hereafter denoted by z in M Rn
% % with M the state space of these variables, which is assumed to have the structure of a differentiable manifold in Rn

% % from greydanus2019hamiltonian, distinguish
% % Learning physical laws from data. Schmidt, M. and Lipson, H. Distilling free-form natural laws from experimental data. Science, 324(5923):81–85, 2009. ;   Iten, R., Metger, T., Wilming, H., Del Rio, L., and Renner, R. Discovering physical concepts with neural networks. ; Bondesan, R. and Lamacraft, A. Learning symmetries of classical integrable systems. arXiv preprint arXiv:1906.04645, 2019.
% % Physics priors for neural networks. 
% % Modeling energy surfaces. (mode MD)

% %schroedinger eq learning \cite{mills2017deep}?


% Deep reinforcement learning (DRL) has ...
% %Existing machine learning methods that deal with similar problems have 
% ... often focused on reinforcement learning~\cite{mnih2016asynchronous,finn2016unsupervised}, but for high-dimensional environments, the computational cost of exploring the state space puts severe limits on the number of interaction parameters with which the agent can influence the physical system~\cite{lillicrap2015continuous}.
% ....
% achieved impressive results for playing games \cite{mnih2016asynchronous,silver2017mastering}, and many applications in this area involve interactions with physical environments \cite{todorov2012mujoco,denil2016learning}.  Physical knowledge and intuition were shown to benefit task performance in robotics \cite{agrawal2016learning,finn2016unsupervised} and yield challenging environments for exploration by virtual agents \cite{haber2018learning}. 
% %As we will demonstrate below, our approach has the potential to improve policies involving continuous physical systems.
% %
% %In addition, 
% ... hierarchical approaches have also been proposed for DRL \cite{kulkarni2016hierarchical,lakshminarayanan2016option}



% older list of results:
% What we have so far for buoyancy case:
% \begin{itemize}
% \item Burgers: ...
% \item Karman-2D: (a) with $n=1$, 
% \item Karman-2D: (b) with LM
% \item Karman-2D: (c) with varying $n$ using phiflow
% \item Buoyancy: (a) with $n=1$, simplest naive case, downsample, and learn differnce (naive m01: dens 0.22280 , vel 0.21461)
% \item Buoyancy: (a) with $n>1$, doing very badly...?
% \item Buoyancy: (a) with input noise, also doing quite badly (too bad?)
% \item Buoyancy: (b) with Lagr. mult. solve for $F$ , blows up (-)
% \item Buoyancy: (b) with Lagr. mult. solve for $F$ with time regularization, stable (old default, Supervised: dens 0.12924 , vel 0.10742)
% \item Buoyancy: (c) with varying $n$ using phiflow (1step: dens 0.25140 , vel 0.26875, best 64step: dens 0.28838 , vel 0.31889)
% \item CGsolve: (a) with PDE-based (DP) loss (Tompson)
% \item CGsolve: (a) pre-computed data
% \item CGsolve: (c) with PDE-based CG-solver loss 
% \item Karman-3D: (a) with $n=1$, 
% \item Karman-3D: (c) with $n=8$ using phiflow
% \end{itemize}

% % outlook?
% Additional generic variants to (perhaps) consider or discuss:
% %\begin{itemize}
% %\item A with $n>1$, diverging data, exploding?
% %\item 
% (a) with sync of $\vrN$ from $\vcN$, not feasible, interesting structures need long to develop, and then it would diverge. (discuss in appendix)
% (a) with noise (like Deepmind GNS), easy for 1step, more complex for multi step. 
% %\end{itemize}

% Ablations:
% \begin{itemize}
% \item correction approaches, show abc for each secnario
% \item no. of msteps ($n$) for (c) DP, and (a) diverging - only 1 scenario?
% \item model sizes 
% \item dropout? noise?
% \item ((previous input steps, show old graph? only discuss, shouldnt be necessary))
% \end{itemize}

% PDE details




% % ---



\section{Experiments}
\label{app:exp}

To acquire our data sets, we generate a set of simulation sequences with varying initial conditions.
These sequences are used for obtaining pairs of source and reference velocity fields for training. 
%
%For appendix: TODO, write out , 
The following PDEs typically work with a continuous 
velocity field $\vu$ with $d$ dimensions and components, i.e.,
$\vu(\vx,t): \mathbb{R}^d \rightarrow \mathbb{R}^d $.
For discretized versions below, $d_{i,j}$ will denote the dimensionality 
of a field such as the velocity with $i \in \{s,r\}$ denoting source/inference manifold 
and reference manifold, respectively.
This yields 
$\vc{} \in \mathbb{R}^{d \times d_{s,x} \times d_{s,y} \times d_{s,z} }$ and
$\vr{} \in \mathbb{R}^{d \times d_{r,x} \times d_{r,y} \times d_{r,z} }$
with domain size $d_{x},d_{y},d_{z}$ for source and reference.
Typically, $d_{r,i} > d_{s,i}$ and $d_{z}=1$ for $d=2$. 
For all PDEs, we use non-dimensional parametrizations as outlined below,
and the components of the velocity vector are denoted by $x,y,z$ subscripts, i.e.,
$\vu = (u_x,u_y,u_z)^T$ for $d=3$.
%$f(\theta,\vc{}): x \rightarrow y $ TODO 

%More on projections / mapping functions:
The mapping function $\project$
denotes a projection to the source manifold by $\project \vr{t}$,
and we assume that the transpose transforms to the reference manifold, i.e., $\project^T \vc{t}$.
The mapping function is typically neither bijective nor unique, i.e.,
$\project^T \project \vr{t} \ne \vr{t}$, however, within this work, we are primarily
concerned with retrieving projected references of the form $\project \vr{t}$.
The potential null-space of $\project^T$ is an interesting topic for super-resolution
approaches \cite{fukami2019super}. % TODO \cite{xie2018,fukami2019super}.
We found that a bi- or tri-linear
spatial downsampling from reference to source space is efficient to compute
and yields sufficient accuracy for the transfer in our experiments.
%
In order to make comparisons with the source simulations easier, we visualize 
the projected reference solution, i.e., $\project \vr{t}$, in the following.
%written as linear projection for simplicity, could be more complex...



% ---

\subsection{Unsteady Wake Flow in Two Dimensions}\label{app:expKarman2d}
% app-karman2d

For the unsteady wake flow setup, we use the incompressible Navier-Stokes equations 
for Newtonian fluids:
\begin{eqnarray}
  \label{eq:model-ns}
  \begin{aligned}
    \frac{\partial u_x}{\partial{t}} + \vu \cdot \nabla u_x =
    - \frac{1}{\rho}\nabla{p} + \nu \nabla\cdot \nabla u_x  \\
    \frac{\partial u_y}{\partial{t}} + \vu \cdot \nabla u_y =
    - \frac{1}{\rho}\nabla{p} + \nu \nabla\cdot \nabla u_y  \\
    \text{subject to} \quad \nabla \cdot \vu = 0,
  \end{aligned}
    % \frac{\partial\vu_x}{\partial{t}} + \vu \cdot \nabla \vu_x =
    % - \frac{1}{\rho}\nabla{p} + \nu \nabla\cdot \nabla \vu_x  \\
    % \frac{\partial\vu_y}{\partial{t}} + \vu \cdot \nabla \vu_y =
    % - \frac{1}{\rho}\nabla{p} + \nu \nabla\cdot \nabla \vu_y \\
  % \frac{\partial\mathbf{v}}{\partial{t}} + \mathbf{v}\cdot\nabla\mathbf{v} =  - \frac{1}{\rho}\nabla{p} + \nu \nabla\cdot \nabla \mathbf{v} + \mathbf{g} 
  % \frac{\partial\vu}{\partial{t}} + \vu \cdot \nabla \vu =
  % - \frac{1}{\rho}\nabla{p} + \nu \nabla\cdot \nabla \vu + \mathbf{g} 
\end{eqnarray}
where $\rho$, $p$, $\nu$, and $g$ denote density, pressure, viscosity, and
external forces, respectively. The constraint, $\nabla \cdot \vu = 0$, is
particularly important and introduces additional complexity
as it restricts motions to the space of divergence-free (i.e., volume preserving) motions.
The flow is integrated over time with operator splitting, and pressure is solved 
implicitly with a Chorin projection \cite{chorin1967numerical}.
%
The domain $\Omega$ has an extent of $1 \times 2$ with open boundary conditions
and a velocity inflow $\vu_{\text{in}}=(0,1)^T$ at the bottom face of the domain.
A circular obstacle with diameter of $0.1$ is located at position $(1/2,1/2)^T$.
For reference simulations, the domain is discretized with 
$d_{r,x}=128$ and $d_{r,y}=256$ cells using a staggered layout for the velocity components. 
The source domain instead contains 
$d_{s,x}=32$ and $d_{r,y}=64$ cells. Data sets from both contain sequences of 500 time steps
each. For the training data, the viscosity coefficient $\nu$ is chosen
to yield Reynolds numbers %Re$_{\text{train}} \in \{97.65625, 195.3125, 390.625, 781.25, 1562.5, 3125\}$,
Re$_{\text{train}} \in \{97.7, 195.3, 390.6, 781.3, 1562.5, 3125.0\}$;
i.e., there is a factor of more than 30 between smallest and largest Reynolds numbers
in the training data.
The test data set instead contains the Reynolds numbers 
% Re$_{\text{test}} \in \{146.484375 , 292.96875 , 585.9375 , 1171.875, 2343.75\}$,
Re$_{\text{test}} \in \{146.5 , 293.0 , 585.9 , 1171.9, 2343.8\}$,
which are denoted as $\times1$, $\times2$, $\times4$, $\times8$, and $\times16$ below, respectively.
%

%discuss? critical Re around 180, -> onset of instability. 
%transition-in-wake (TrW) regime , up to Re 400. currently not present in 2D data.}

%\todo{mention number of vortex shedding cycles per test case?}


\paragraph{Training Procedure}

% \begin{figure}[htb]
%   \centering
%   \subcaptionbox{Regular model}{\includegraphics[width=0.45\columnwidth,page=2]{figs/summary-karman-plot}}
%   \subcaptionbox{Smaller model}{\includegraphics[width=0.45\columnwidth,page=1]{figs/summary-karman-plot-small}}
%   \caption{\ku{For five test simulations with different Reynolds numbers, the
%       velocity error is measured and averaged over 500 simulation steps. Two NN
%       models of different size are evaluated.}}
%   \label{fig:appx:karman-plot}
% \end{figure}

The neural network of $\corr$ is fully convolutional. It consists of 
five ResBlocks \cite{he2015} with 5$\times$5 kernels. 
The convolutional layers have two times 32 features per block 
(details of the architecture are given in \myappref{app:models}). 
Overall, the model has around 260k trainable parameters.
\new{In addition to the velocity, the model receives a constant field containing 
the Reynolds number in order to distinguish the different physical regimes.}

With the Reynolds number range above, we generate 500 time steps as 
training data, which contain temporal dynamics with ca. eight vortex shedding cycles for each case,
% large eddy turnover time is similar:
\new{i.e., they cover a similar number of eddy turnover times.}
This leads to roughly 98 million cells of data in the reference trajectories, 
which are down-sampled to 6.1 million cells with lower resolution of the source data.
Example flow fields are shown in \myfigref{fig:appx:karman-images-train}.

All SOL models are trained with the differentiable physics solver for 99.8k iterations 
with a batch size of 3 and a learning rate of $10^{-4}$.
The NON model uses the same training modalities replacing the differentiable PDE solver
with the supervised loss of \myeqref{eq:non-min}.
On the other hand, all PRE models are trained in a supervised manner for 36k iterations 
with a batch size of 32 and initial learning rate of $10^{-3}$ that is lowered to 
$5\times10^{-7}$ over the course of the training. 
%
Here, we augment the training data via randomized horizontal flipping and 
use 5\% of the training data as validation samples.
To show the stability of training, we train three models for each case 
below with different random seeds.

%... for the Non-SOL models, trained 400 epochs with a batch size of 32 and the adaptive learning rate started from 0.001, used 5\% of the data for the validation data set

\paragraph{Results}

%For the unsteady wake flow scenario, 
% Below,
We present results for the unsteady wake flow scenario
using models trained via different interaction methodologies %, evaluated
and evaluate each model on the test set of Reynolds numbers Re$_{\text{test}}$. 
Each simulation is computed for 500 time steps using the source solver in combination with 
a correction from a trained neural network. Mean errors are computed 
in comparison to reference phase space states, i.e., $\project \vrN$.
We compute the errors over the three trained models for each variant.

In this scenario, the NON model already leads
to a significant reduction of the overall velocity error. While the regular source 
simulation (SRC) shows a MAE of 0.146 with respect to the projected reference 
states $\project \vrN$, the NON model reduces this error to 0.049. These errors (and the 
following measurements) are mean values for all five test Reynolds numbers, which 
were not seen at training time. The results are visualized in 
\myfigref{fig:appx:karman-plot}, and corresponding numeric values are given in \mytabref{tab:karman}.

The pre-computed variants improve on this behavior,
roughly halving the remaining error.
The pre-computed variant without temporal regularization (\pre{SR}) gives a worse performance
than the one with spatiotemporal regularization (PRE) but, nonetheless, fares better than the 
NON version.

\myfigref{fig:appx:karman-plot} additionally shows results for different 
SOL versions trained with the solver-in-the-loop interaction. While the \sol{4} version 
fares better than NON, it is only roughly on par with \pre{SR}. Increasing the number 
of look-ahead steps, however, increases the performance substantially
with the \sol{32} model exhibiting a final MAE of only 0.013.
Several visual examples of simulated flows from the five 
test cases used in these evaluations are shown in \myfigref{fig:appx:karman-images-test}.
It is visible that the SOL version matches the behavior of the reference 
solution much more closely.

We additionally break down the errors with respect to the different Reynolds 
numbers of the five cases in \myfigref{fig:appx:karman-plot-re}. Despite a factor of
16 between the Reynolds numbers, there is no significant decrease in performance across 
the different cases. 
Only the NON version exhibits slightly larger errors for higher Reynolds numbers.
On the other hand, the performance is largely uniform for the SOL versions.
% \mynote{kiwon}{NON version? did you mean PRE version here? need to double-check this claim}
% \mynote{nils}{I meant NON, it has an increasing error across the Reynolds numbers for the first few,
%   that is something you could expect from the underlying physics (larger Re being more difficult)}
% \mynote{kiwon}{Okay, now I see your point. I overlooked it ... it sounded like ``lower than other models''...
%   could it be safer to write like this?
% \ku{While the NON version exhibits errors proportional to the Reynolds number for the four lower Reynolds cases,}
% }

Due to the distinct vortex shedding characteristics of the flow, 
it is interesting to evaluate the flow field in terms of its frequency spectrum.
%
As an example, \myfigref{fig:appx:karman-ucen-step} shows the $u_x$ velocity component 
over the course of 500 simulation steps at the center of domain, i.e., behind the
obstacle, for one of our test data sets.
%
We show the corresponding evaluation in \myfigref{fig:appx:karman-plot-freq}.
Interestingly, especially the PRE versions fare better in terms of frequency 
errors. Here the relatively expensive
pre-computation step shows a performance gain. Nonetheless, the models trained via differentiable 
physics likewise learn to control the frequency behavior when training 
with a sufficient number of look-ahead steps %, with the \sol{32} model yielding
as the \sol{32} model yields
a substantially lower frequency error than the PRE model.

%\todo{optionally - show and discuss std dev over the three models?}
% improvements per Re , p41 - (move x start to 0.4?)

We additionally show results for a smaller model for 
a simpler sequential convolutional network with
57k trainable parameters in \myfigref{fig:appx:karman-plot-small}.
The overall relative ordering of the interaction methods remains the same.
The non-interacting method
performs worse than pre-computation, which in turn is outperformed 
by the differentiable physics interaction. However, the overall 
performance is reduced, e.g., the NON model only reduces the error by ca. 30\%.
%
The \sol{16} version still outperforms the other versions.
Overall, not surprisingly, the reduced weight count significantly
reduces the representational capabilities of the neural networks and leads
to a deteriorated performance. Nonetheless, training via interactions with differentiable 
physics is beneficial for inference performance. % no "the", general statement here

To conclude, approximate solutions of the unsteady wake flow case can be corrected substantially by learned
models, and especially training with differentiable physics in the loop yields significantly 
reduced errors in long simulated sequences. The \sol{32} version with a larger model reduces the MAE 
with respect to the reference solution to less than 9\% (on average) of the error induced by the source 
simulation.

\newcommand{\myFigHKarman}{0.30\columnwidth}
\begin{figure}[htb]
  \centering
  \subcaptionbox{ Velocity error for different models                          }{\includegraphics[height=\myFigHKarman,page=3]{figs/summary-karman-plot}}
  \subcaptionbox{ Velocity improvement (relative to SRC) for different models  }{\includegraphics[height=\myFigHKarman,page=7]{figs/summary-karman-plot}}
  \caption{
    Different models applied to five test cases over 500 time steps for the unsteady wake flow scenario.
    The \sol{32} reduces the error introduced by SRC by a factor of 11.2 on average.
  }
  \label{fig:appx:karman-plot}
\end{figure}

\begin{figure}[htb]
  \centering
  \subcaptionbox{ Velocity error per Reynolds number                         }{\includegraphics[height=\myFigHKarman,page=2]{figs/summary-karman-plot}}
  \subcaptionbox{ Velocity improvement (relative to SRC) per Reynolds number }{\includegraphics[height=\myFigHKarman,page=5]{figs/summary-karman-plot}}
  \caption{
    Separate evaluations for five different test cases of the unsteady wake flow scenario.
  }
  \label{fig:appx:karman-plot-re}
\end{figure}

\newcommand{\tweakSpaceKtwo}{\vspace{-0.1cm}}
\begin{figure}[htb]
  \centering
  \includegraphics[height=\myFigHKarman,page=1]{figs/summary-karman-plot-appx}
  \tweakSpaceKtwo\caption{
    $u_x$-velocity at the center of domain for one test data set (Re = $\times4$).
  }
  \label{fig:appx:karman-ucen-step}
\end{figure}

\begin{figure}[htb]
  \centering
  \subcaptionbox{ Frequency error for different models }{\includegraphics[height=\myFigHKarman,page=9]{figs/summary-karman-plot}}
  \subcaptionbox{ Frequency error per Reynolds number  }{\includegraphics[height=\myFigHKarman,page=8]{figs/summary-karman-plot}}
  \tweakSpaceKtwo\caption{
    Frequency-domain evaluation for the unsteady wake flow scenario. 
    Shown for the five test cases over 500 time steps.
  }
  \label{fig:appx:karman-plot-freq}
\end{figure}

\begin{figure}[htb]
  \centering
  \subcaptionbox{ Velocity error for different smaller models      }{\includegraphics[height=\myFigHKarman,page=3]{figs/summary-karman-plot-small}}
  \subcaptionbox{ Smaller models per Reynolds number               }{\includegraphics[height=\myFigHKarman,page=2]{figs/summary-karman-plot-small}}
  \tweakSpaceKtwo\caption{
    Different models with a smaller network size (57k trainable weights) applied to five test cases over 500 time steps for the unsteady wake flow scenario.
  }
  \label{fig:appx:karman-plot-small}
\end{figure}

\begin{figure}[tb]
  \centering
  % 1875 x 400 image = 100 x 21.3 overpic unit
  \begin{overpic}[width=0.96\linewidth]{figs/summary-karman-training-data.jpg}
    \put(-1.6,0)    {\scriptsize\rotatebox{90}{\makebox[10.66\unitlength]{Reference}}}
    \put(-1.6,10.66){\scriptsize\rotatebox{90}{\makebox[10.66\unitlength]{SRC}}}
    \put(100.5,0){\includegraphics[height=10.66\unitlength]{figs/summary-legend-karman}}
  \end{overpic}
  \caption{An example sequence of the unsteady wake flow from the training
      data set for time steps $t \in \{50, 60, \cdots, 200\}$.
  }
  \label{fig:appx:karman-images-train}
\end{figure}

\begin{figure}[tb]
  \centering
  % 1875 x 600 image = 100 x 32 overpic unit
  \begin{overpic}[width=0.96\linewidth]{figs/summary-karman-test-data-b.jpg}
    \put(0,30)      {(a)}
    \put(-1.6,0)    {\scriptsize\rotatebox{90}{\makebox[10.66\unitlength]{Reference}}}
    \put(-1.6,10.66){\scriptsize\rotatebox{90}{\makebox[10.66\unitlength]{\sol{32}}}}
    \put(-1.6,21.33){\scriptsize\rotatebox{90}{\makebox[10.66\unitlength]{SRC}}}
    \put(100.5,0){\includegraphics[height=10.66\unitlength]{figs/summary-legend-karman}}
  \end{overpic}
  \\\vspace{0.5em}
  \begin{overpic}[width=0.96\linewidth]{figs/summary-karman-test-data.jpg}
    \put(0,30)      {(b)}
    \put(-1.6,0)    {\scriptsize\rotatebox{90}{\makebox[10.66\unitlength]{Reference}}}
    \put(-1.6,10.66){\scriptsize\rotatebox{90}{\makebox[10.66\unitlength]{\sol{32}}}}
    \put(-1.6,21.33){\scriptsize\rotatebox{90}{\makebox[10.66\unitlength]{SRC}}}
    \put(100.5,0){\includegraphics[height=10.66\unitlength]{figs/summary-legend-karman}}
  \end{overpic}
  \caption{Time steps of test cases for the unsteady wake flow for $t \in \{50, 60, \cdots, 200\}$: (a) Re = $\times1$ and (b) Re = $\times16$. 
    }
  \label{fig:appx:karman-images-test}
\end{figure}

\begin{table}[tb]
  \caption{Quantitative evaluation of different models for the unsteady wake flow scenario.}
  \label{tab:karman}
  \begin{center}
    \begin{tabular}{ccccccccc}
      \toprule
      \textbf{Model} & \multicolumn{8}{c}{\textbf{MAE Velocity}, Mean (std. dev.)}                      \\
      \cmidrule{2-9}
                     & SRC     & NON     & \pre{SR} & PRE     & \sol{4} & \sol{8} & \sol{16} & \sol{32} \\
      \midrule                                                            
      Regular        & 0.146   & 0.049   & 0.036    & 0.031   & 0.041   & 0.031   & 0.023    & 0.013    \\
                     & (0.004) & (0.012) & (0.009)  & (0.010) & (0.009) & (0.012) & (0.004)  & (0.003)  \\
      Smaller        & 0.146   & 0.092   & 0.083    & 0.059   & -       & 0.042   & 0.035    & -        \\
                     & (0.004) & (0.028) & (0.025)  & (0.015) & -       & (0.011) & (0.010)  & -        \\
      \midrule
                     & \multicolumn{8}{c}{\textbf{\new{$L^2$ Error} of $u_x$ Velocity in Frequency Domain}} \\
      \cmidrule{2-9}
                     & SRC     & NON     & \pre{SR} & PRE     & \sol{4} & \sol{8} & \sol{16} & \sol{32} \\
      \midrule                                                           
      Regular        & 0.557   & 0.202   & 0.106    & 0.087   & 0.194   & 0.128   & 0.101    & 0.051    \\
      Smaller        & 0.557   & 0.275   & 0.244    & 0.158   & -       & 0.093   & 0.155    & -        \\
      \bottomrule
    \end{tabular}
  \end{center}
\end{table}

% --- 

\clearpage

\subsection{Buoyancy-driven Fluid Flow}\label{app:expBuoy}
% app-buoyancy 

This scenario encompasses a volume of hot smoke rising in a 
closed container. The motion of the smoke volume is driven by buoyancy forces
computed via a marker field that is passively advected in the flow, and 
which marks a region of fluid with lower density.
Assuming a small relative change of density between the marker and the 
bulk, we compute the resulting forces with a Boussinesq model. 
Hence, this scenario is likewise based on the Navier-Stokes equations, 
but due to the additional coupled system, it leads to significantly 
more chaotic and complex behavior than the unsteady wake flow.
%
In order to target solutions with complex motions, we do not explicitly solve for
viscosity effects, but rely on the numerical viscosity inherent in the discretization.
%
%note: buoyancy case actually has 2 coupled sets of equations, velocity and density, ie, increased complexity, write out both equations} numerical viscosity only
%
This yields the following PDE:
\begin{eqnarray}
  \label{eq:boussinesq}
  \frac{\partial u_x}{\partial{t}} + \vu \cdot \nabla u_x =
  - \frac{1}{\rho}\nabla{p},
  %\nonumber
  %\\
  \quad 
  %
  \frac{\partial u_y}{\partial{t}} + \vu \cdot \nabla u_y =
  - \frac{1}{\rho}\nabla{p}  + \eta d
  \nonumber
  \\
  \text{subject to} \quad \nabla \cdot \vu = 0,
  %\nonumber \\
  \quad
  \frac{\partial d}{\partial{t}} + \vu \cdot \nabla d = 0 \ , 
  %\\  \nonumber
\end{eqnarray}
where $\eta$ denotes the buoyancy factor for the Boussinesq model.

We also use this scenario to demonstrate that the reference data can be computed by 
a discretization or algorithm that differs from the one used to compute the source trajectories.
More specifically, we use second-order pressure projection 
scheme for the reference trajectory solutions \cite{zehnder2018},
which was shown to lead to an improved conservation of energy \cite{hairer2006geometric}.
In addition, we use a less dissipative advection scheme for the source 
and reference solvers \cite{selle2008}.

%give dimensions $d_{s,x}$ etc. , params ... 2nd order advection \cite{selle2008} , for reference instead uses \cite{zehnder2018}

The domain has an extend of $1 \times 2$ units, where the marker density is injected 
in the lower quadrant.
%
The reference simulations use a staggered discretization with 
$d_{r,x}=128$ and $d_{r,y}=256$, while the source simulations 
use a domain with $d_{s,x}=32$ and $d_{r,y}=64$.
%
We randomize the initial size of the marker volumes with circular shapes with a radius 
$r \sim \mathcal{U}(0.1,0.25)$, where $\mathcal{U}$ denotes a uniform distribution.
The training data set consists of 
48 different initial conditions simulated for 1000 steps each.
Several examples are shown in \myfigref{fig:appx:smoke-images-train}.
% note: 50 were generated, but due to batchsize 4 ,2 scenes were never used...
%
For the test scenes, we change the initial marker distribution $d$
to obtain five simulations containing two circles with
$r \sim \mathcal{U}(0.05,0.1)$ and another five simulations with 
$r \sim \mathcal{U}(0.2,0.3)$. Thus, we obtain ten test scenes, half of which 
have a reduced marker quantity compared to the training data and five with 
an increased quantity. As the $d$ determines the forces induced by the Boussinesq
model, this leads to simulations that are slower and faster, respectively, 
than those in the training set.

\paragraph{Training Procedure}

The neural network architecture for $\corr$ follows the one described above, 
but instead uses four ResBlocks with 16 features each and contains ca. 36k trainable weights.
As both velocity $\vu$ and marker $d$ determine the dynamics of the flow, the network 
receives both fields as input, but still only infers a correction for the velocity;
%i.e., $d$ is not directly modified by $\corr$, only via advection through $\vu$.
i.e., $d$ is modified only via advection through $\vu$, not directly by $\corr$.
%
% unsup / phiflow:
% 5 vali scenes (index 300+), with 300 time steps
% 500 iters, 12 batches w 49 steps each -> 294k iters
All SOL and NON models are trained for 294k iterations 
with a batch size of 4 and a learning rate of $10^{-4}$.
We evaluate the models on validation set with 5 simulations and 300 time steps 
drawn from the same initial marker distribution as the training data,
and keep the model with the lowest validation loss.

% sup:
% 200 epos * 48*1000 simsteps / 32 bs -> 300k
To speed up the pre-computations, we only compute $\corrPre$ 
for cells $i,j$ in the domain with $d_{i,j}>10^{-4}$ (we validate this choice below).
The PRE variants of $\corr$ are then trained on the resulting, regularized 
data for 300k iterations with a batch size of 32 using horizontal flipping as data augmentation.

\paragraph{Results}
%
We evaluate different models which are applied to 300 time steps of ten 
test conditions. Errors with respect to the reference solutions are computed and averaged 
across the resulting 3k phase field states.
Numeric error values for the following tests can be found in \mytabref{tab:smoke}.

We evaluate the different baseline versions (NON and PRE) in comparison to the source 
simulation (which underlies all other variants) and compare them to SOL versions 
with increasing look-ahead.
The resulting errors and relative improvements 
are shown in \myfigref{fig:appx:smoke-plot} and given numerically in \mytabref{tab:smoke}.
It is apparent that the SOL versions yield very significant improvements over the other learned variants.
Besides the velocity errors, we also provide an evaluation of the passively 
advected marker density $d$. This quantity is crucial for the dynamics of the flow,
but cannot be influenced directly by the neural networks. Hence, it provides 
an additional view on how well the inferred corrections manage to reduce the numerical 
errors of the source simulation. %
The corresponding evaluation highlights 
that both velocity and density improvements increase consistently with SOL variants that 
were trained with larger look-aheads.
%
We also evaluate the different
models in terms of kinetic energy of the flows. As the kinetic energy is agnostic 
to the direction of the flow, the residual errors of the different variants do not show up as clearly 
as in the other evaluations.
%
However, while the density and kinetic energy improvements are 
smaller than those for the velocity fields, the \sol{128} model nonetheless
clearly outperforms the other variants.

Visualized evolutions of several test simulations 
are shown in \myfigref{fig:appx:smoke-images-test}. Here, the bi-modal nature 
of the test data with smaller (b) and larger (a,c) initial marker density configurations
is shown. The different initial conditions lead to smaller and larger average velocities
and, hence, highlight that the trained model generalizes very well.

%\todo{comment on generalization again? point out in images?}
%Note that the standard deviations of the measurements for the test cases of the buoyancy-driven flow 

\paragraph{Ablations}
%
An evaluation of different neural network architectures for the buoyancy-driven 
flows with \sol{2} interaction illustrates how improvements stagnate beyond a certain network size and depth.
For example, a model with more than 100k weights and almost three times the size of the regular 
model only yields an improvement of 3.6\%. 
Another increase by a factor of four only gives 0.3\% improvement.
The corresponding graphs can be found in \myfigref{fig:appx:smoke-sizes}.
Decreasing the network size, on the other hand, yields a performance that is
8.7\% lower or even more for the smallest model. This motivates our choice 
to focus on the architecture with 36k trainable parameters, which was 
used for all other test with the buoyancy-driven flows.
%with numeric values provided in \mytabref{tab:smoke}.

As discussed in the main text, we also evaluated a method proposed by 
Sanchez et al. \cite{sanchez2020gns} to perturb inputs to network with noise 
in order to stabilize predictions. This approach shares our goal to 
reduce the shift of distributions for the input data such that the trained 
networks can produce more reliable estimates % when inputs they encounter new inputs
as they encounter new inputs
at inference time. However, in contrast to the Lagrangian graph-based physics
predictions, the added noise did not lead to large gains in our context.
We test a variety of trained \sol{2} networks for which noise was injected
into the input features, i.e., cell-wise samples of velocity and
marker density, from a component-wise normal distribution $\mathcal{N}(0,\sigma)$
with standard deviation $\sigma$.

Details of the results are visualized in \myfigref{fig:appx:smoke-noise}.
As can be seen in the results,
there is only a slight positive effect across a wide range of different noise strengths.
The networks with $\sigma \sim 10^{-4}$ show the best results. 
However, the improvements of up to 34.6\% via noise perturbations
are surpassed by the \sol{n} models, where the best one yields an improvement of 59.8\%.
%
% We think that this lack of clear gains from injecting noise comes from 
We think that the gains of our interacting model compared to injecting noise come from
the systematic improvements of the SOL training, which potentially 
provides more reliable inputs at training time than stochastic perturbations.
The fully convolutional nature of the networks additionally provides 
regularization at training time.
% Kiwon - the former version didn't sound like the reason of the lack of gain of injecting noise...

We have also evaluated how sub-optimal choices for solver interactions 
affect the inference performance.
We train several NON models that are allowed % without interaction
to evolve for $n$ time steps without interaction,
while computing a regular \new{$L^2$ loss} via \myeqref{eq:non-min}. These versions 
are denoted with \non{dn} for $n$ steps of diverging evolution. 
%
In addition, we evaluate a model \pre{\text{SR}} using a pre-computed interaction without 
temporal regularization (i.e., only spatial) and one version (\pre{\text{F}}) that uses the full spatiotemporal 
regularization without a density threshold; i.e., it requires several times more pre-computation 
by solving the Lagrange-multiplier minimization for the full spatial domains.
Especially, the \non{dn} variants perform badly and exhibit large errors, with \non{d8} significantly distorting 
the flow behavior, instead of improving it. 
The corresponding evaluations %can be found in \mytabref{tab:smoke} and 
are visualized in \myfigref{fig:appx:smoke-plot-diverging}.
It is likewise apparent that the additional PRE variants deteriorate the ability of the ANNs to 
correct the numerical errors of the source simulations.
%diverging, and PRE-SR , PRE-F without cutoff in \myfigref{fig:appx:smoke-plot-diverging}

To summarize, despite the complexity of the buoyancy-driven flows and 
the difficult reference trajectories produced by a higher-order PDE solver,
the numerical errors of the source simulation can be reduced very successfully 
by training with the solver in the training loop.


% \newcommand{\tweakSpaceSmoke}{\vspace{-0.1cm}}
\newcommand{\myFigHSmoke}{0.25\columnwidth}
% \begin{figure}[htb]
%   \centering
%   \subcaptionbox{Velocity error\label{fig:appx:smoke-plot:models-vel}}{\includegraphics[height=\myFigHSmoke,page=2]{figs/out-vgf-jmsmaller-steps-smoke-uns-jmsmaller-msteps}}
%   \subcaptionbox{Marker advection error\label{fig:appx:smoke-plot:models-dens}}{\includegraphics[height=\myFigHSmoke,page=1]{figs/out-vgf-jmsmaller-steps-smoke-uns-jmsmaller-msteps}}
%   \\
%   \subcaptionbox{Velocity improvement\label{fig:appx:smoke-plot:models-vel-impr}}{\includegraphics[height=\myFigHSmoke,page=6]{figs/out-vgf-jmsmaller-steps-smoke-uns-jmsmaller-msteps}}
%   \subcaptionbox{Improvement of marker advection\label{fig:appx:smoke-plot:models-ke-impr}}{\includegraphics[height=\myFigHSmoke,page=5]{figs/out-vgf-jmsmaller-steps-smoke-uns-jmsmaller-msteps}}
%   \caption{
%     Density and marker advection errors for different models, especially different SOL versions with increasing look-ahead.
%     In the second row, we show improvements relative to the source version SRC.
%     %\ku{For each simulation, the density error is measured and averaged over 300 simulation steps.}
%   }
%   \label{fig:appx:smoke-plot}
% \end{figure}

\newcommand{\myFigHSmokeS}{0.175\columnwidth}
\begin{figure}[htb]
  \centering
  \subcaptionbox{Velocity error\label{fig:appx:smoke-plot:models-vel}}{\includegraphics[height=\myFigHSmokeS,page=2]{figs/out-vgf-jmsmaller-steps-smoke-uns-jmsmaller-msteps}}
  \subcaptionbox{Marker advection error\label{fig:appx:smoke-plot:models-dens}}{\includegraphics[height=\myFigHSmokeS,page=1]{figs/out-vgf-jmsmaller-steps-smoke-uns-jmsmaller-msteps}}
  \subcaptionbox{Kinetic energy error\label{fig:appx:smoke-plot:models-ke}}{\includegraphics[height=\myFigHSmokeS,page=4]{figs/out-vgf-jmsmaller-steps-smoke-uns-jmsmaller-msteps}}
  \\
  \subcaptionbox{Velocity improvement\label{fig:appx:smoke-plot:models-vel-impr}}{\includegraphics[height=\myFigHSmokeS,page=6]{figs/out-vgf-jmsmaller-steps-smoke-uns-jmsmaller-msteps}}
  \subcaptionbox{Impr. of marker advection\label{fig:appx:smoke-plot:models-den-impr}}{\includegraphics[height=\myFigHSmokeS,page=5]{figs/out-vgf-jmsmaller-steps-smoke-uns-jmsmaller-msteps}}
  \subcaptionbox{Impr. of kin. energy\label{fig:appx:smoke-plot:models-ke-impr}}{\includegraphics[height=\myFigHSmokeS,page=7]{figs/out-vgf-jmsmaller-steps-smoke-uns-jmsmaller-msteps}}
  \caption{Velocity, marker advection, and kinetic energy errors for different
    models, especially for different SOL versions with increasing look-ahead. In
    the second row, we show improvements relative to the source version SRC.}
  \label{fig:appx:smoke-plot}
\end{figure}

%  \subcaptionbox{Different setups\label{fig:appx:smoke-plot:setups}}{\includegraphics[height=\myFigHSmoke,page=2]{figs/out-vgf-jmsmaller-steps-m01versions}}

\begin{figure}[htb]
  \centering
  \subcaptionbox{Trainable weights\label{fig:appx:smoke-plot:sizesw}}{\includegraphics[height=\myFigHSmoke,page=9]{figs/out-vgf-jmsmaller-steps-smoke-uns-varymodel}}
  \subcaptionbox{Velocity error\label{fig:appx:smoke-plot:sizesv}}{\includegraphics[height=\myFigHSmoke,page=2]{figs/out-vgf-jmsmaller-steps-smoke-uns-varymodel}}
  \subcaptionbox{Marker error\label{fig:appx:smoke-plot:sizesd}}{\includegraphics[height=\myFigHSmoke,page=1]{figs/out-vgf-jmsmaller-steps-smoke-uns-varymodel}}
  \subcaptionbox{Velocity improvement\label{fig:appx:smoke-plot:sizesvi}}{\includegraphics[height=\myFigHSmoke,page=5]{figs/out-vgf-jmsmaller-steps-smoke-uns-varymodel}}
  \caption{\sol{2} training with different architectures that strongly vary the
    number of trainable parameters (a). While the smaller two models lead to a
    clear drop in accuracy, the larger two architectures yield small gains
    despite the increased weight count.}
  \label{fig:appx:smoke-sizes}
\end{figure}

\newcommand{\myFigHSmokeN}{0.22\columnwidth}
\begin{figure}[htb]
  \centering
  \subcaptionbox{Velocity improvement \label{fig:appx:smoke-plot:noisev}}{\includegraphics[height=\myFigHSmokeN,page=5]{figs/out-vgf-jmsmaller-noise}}
  \subcaptionbox{Improvement of marker advection  \label{fig:appx:smoke-plot:noised}}{\includegraphics[height=\myFigHSmokeN,page=4]{figs/out-vgf-jmsmaller-noise}}
  \caption{Varying levels of noise injected into the input features for \sol{2}
    at training time. While values around $10^{-4}$ lead to slight positive
    effects, the improvements are negligible compared to those achievable by the
    SOL variants.}
  \label{fig:appx:smoke-noise}
\end{figure}

\newcommand{\myFigHSmokeB}{0.22\columnwidth}

\begin{figure}[tb]
  \centering
  % 2979 x 400 image = 100 x 13.43 overpic unit
  \begin{overpic}[width=0.96\linewidth]{figs/summary-smoke-training-data}
    \put(-1.6,0)   {\scriptsize\rotatebox{90}{\makebox[6.71\unitlength]{Ref.}}}
    \put(-1.6,6.71){\scriptsize\rotatebox{90}{\makebox[6.71\unitlength]{SRC}}}
    \put(100.5,0){\includegraphics[height=6.71\unitlength]{figs/summary-legend-buoy}}
  \end{overpic}
  \caption{An example sequence of the buoyancy scenario from the training data
    set for time steps $t \in \{0, 25, \cdots, 375\}$. % for SRC (top) and reference versions (bottom).
  }
  \label{fig:appx:smoke-images-train}
\end{figure}

\begin{figure}[tb]
  \centering
  % 1986 x 400 image = 100 x 20.14 overpic unit
  \begin{overpic}[width=0.96\linewidth]{figs/summary-smoke-test-data}
    \put(0,18)      {{\color{white}(a)}}
    \put(-1.6,0)    {\scriptsize\rotatebox{90}{\makebox[6.71\unitlength]{Ref.}}}
    \put(-1.6,6.71){\scriptsize\rotatebox{90}{\makebox[6.71\unitlength]{\sol{128}}}}
    \put(-1.6,13.43){\scriptsize\rotatebox{90}{\makebox[6.71\unitlength]{SRC}}}
    \put(100.5,0){\includegraphics[height=6.71\unitlength]{figs/summary-legend-buoy}}
  \end{overpic}
  \\\vspace{0.5em}
  \begin{overpic}[width=0.96\linewidth]{figs/summary-smoke-test-data-b}
    \put(0,18)      {{\color{white}(b)}}
    \put(-1.6,0)    {\scriptsize\rotatebox{90}{\makebox[6.71\unitlength]{Ref.}}}
    \put(-1.6,6.71){\scriptsize\rotatebox{90}{\makebox[6.71\unitlength]{\sol{128}}}}
    \put(-1.6,13.43){\scriptsize\rotatebox{90}{\makebox[6.71\unitlength]{SRC}}}
    \put(100.5,0){\includegraphics[height=6.71\unitlength]{figs/summary-legend-buoy}}
  \end{overpic}
  \\\vspace{0.5em}
  \begin{overpic}[width=0.96\linewidth]{figs/summary-smoke-test-data-c}
    \put(0,18)      {{\color{white}(c)}}
    \put(-1.6,0)    {\scriptsize\rotatebox{90}{\makebox[6.71\unitlength]{Ref.}}}
    \put(-1.6,6.71){\scriptsize\rotatebox{90}{\makebox[6.71\unitlength]{\sol{128}}}}
    \put(-1.6,13.43){\scriptsize\rotatebox{90}{\makebox[6.71\unitlength]{SRC}}}
    \put(100.5,0){\includegraphics[height=6.71\unitlength]{figs/summary-legend-buoy}}
  \end{overpic}
  \caption{Several time steps $t \in \{50, 60, \cdots, 200\}$ of three buoyancy-driven fluid flow test cases
    (a)-(c).}
  \label{fig:appx:smoke-images-test}
\end{figure}


\begin{figure}[htb]
  \centering
  \subcaptionbox{Velocity error \label{fig:appx:smoke-plot:vel}}{\includegraphics[height=\myFigHSmokeB,page=2]{figs/out-vgf-jmsmaller-diverging-beta0}}
  \subcaptionbox{Density error \label{fig:appx:smoke-plot:den}}{\includegraphics[height=\myFigHSmokeB,page=1]{figs/out-vgf-jmsmaller-diverging-beta0}} 
  \subcaptionbox{Velocity improvement \label{fig:appx:smoke-plot:imp}}{\includegraphics[height=\myFigHSmokeB,page=5]{figs/out-vgf-jmsmaller-diverging-beta0}}
  % \subcaptionbox{ KE impr \label{fig:appx:smoke-plot:x}}{\includegraphics[height=\myFigHSmokeB,page=6]{figs/out-vgf-jmsmaller-diverging-beta0}}
  \caption{A comparison of models trained with a variety of sub-optimal
    interaction schemes for the buoyancy scenario. \non{dn} allows
    non-interacting models to evolve and diverge over $n$ steps, while \pre{SR}
    employs only spatial regularization in the pre-computation. \pre{F}
    resembles PRE, but was trained without a density threshold. Especially, the
    changes relative to SRC in (c) highlight that the \non{dn} variants have a
    negative effect.}
  \label{fig:appx:smoke-plot-diverging}
\end{figure}

% 8 ?
% 4 models , 3x noise?  , NON_d4?
\begin{table}[tb]
  \caption{Quantitative evaluation of models for the buoyancy-driven flow
    scenario. M$_{XS,S,L,XL}$ denote different model sizes, while
    $\sigma_{1,2,3}$ denote models trained with noise of
    $\sigma=10^{-3,-4,-5}$.}
  \label{tab:smoke}
  \begin{center}
    \begin{tabular}{ccccccccc}
      \toprule
      \textbf{Quantity} & \multicolumn{8}{c}{\textbf{MAE Velocity}, Mean (std. dev.)}                      \\
      \cmidrule{2-9}
                     & SRC     & NON     & PRE      & \sol{2} & \sol{16} & \sol{32} & \sol{64} & \sol{128} \\
      \midrule                                                            
      Velocity       &  1.590  &  1.079  &  1.373  &  1.027  &  0.859  &  0.775  &  0.695  &  0.620  \\
                     & (1.032) & (0.658) & (0.985) & (0.656) & (0.539) & (0.482) & (0.420) & (0.389) \\
      Marker $d$     &  0.677  &  0.499  &  0.579  &  0.484  &  0.430  &  0.419  &  0.401  &  0.391  \\
                     & (0.473) & (0.336) & (0.409) & (0.325) & (0.281) & (0.277) & (0.262) & (0.253) \\
      \midrule
%                     & \multicolumn{8}{c}{\textbf{ \todo{noise? others...} }} \\
%      \cmidrule{2-9}
                     &M$_{XS}$ &M$_{S}$  &  M$_{L}$& M$_{XL}$&$\sigma_1$&$\sigma_2$&$\sigma_3$& \non{d4} \\
      \midrule                                                           
      Velocity       &  1.228  &  1.193  &  0.982  &  0.969  &  1.070  &  1.056  &  1.078  &  3.196  \\
                     & (0.746) & (0.826) & (0.646) & (0.626) & (0.683) & (0.700) & (0.706) & (1.404) \\
      Marker $d$     &  0.521  &  0.494  &  0.461  &  0.466  &  0.503  &  0.496  &  0.503  &  0.656  \\
                     & (0.352) & (0.349) & (0.313) & (0.318) & (0.341) & (0.339) & (0.345) & (0.426) \\
      % Velocity       & 0.146   & 0.049   & 0.036    & 0.031   & 0.041   & 0.031   & 0.023    & 0.013    \\
      %                & (0.004) & (0.012) & (0.009)  & (0.010) & (0.009) & (0.012) & (0.004)  & (0.003)  \\
      % Marker $d$     & 0.146   & 0.049   & 0.036    & 0.031   & 0.041   & 0.031   & 0.023    & 0.013    \\
      %                & (0.004) & (0.012) & (0.009)  & (0.010) & (0.009) & (0.012) & (0.004)  & (0.003)  \\
      % \midrule
      %                & \multicolumn{8}{c}{\textbf{ \todo{noise? others...} }} \\
      % \cmidrule{2-9}
      %                & M_{XS}  &  M_{S}  &  M_{L}   & M_{XL}  &$\sigma_1$&$\sigma_2$&$\sigma_3$& \non{d4} \\
      % \midrule                                                           
      % Velocity       & 0.557   & 0.202   & 0.106    & 0.087   & 0.194   & 0.128   & 0.101    & 0.051    \\
      %                & (0.004) & (0.012) & (0.009)  & (0.010) & (0.009) & (0.012) & (0.004)  & (0.003)  \\
      % Marker $d$     & 0.557   & 0.202   & 0.106    & 0.087   & 0.194   & 0.128   & 0.101    & 0.051    \\
      %                & (0.004) & (0.012) & (0.009)  & (0.010) & (0.009) & (0.012) & (0.004)  & (0.003)  \\
      \bottomrule
    \end{tabular}
  \end{center}
\end{table}






% --- Burgers

\clearpage

\subsection{Forced Advection-Diffusion}\label{app:expBurgers}

In the forced advection-diffusion scenario, we target a PDE environment 
with a constant, randomized forcing term. 
This forcing continuously injects energy into the dissipative system
and takes the form of a spectrum of parametrized bands of sine waves.
In this scenario, we target Burgers' equation. It represents
a well-studied advection-diffusion PDE:
%... forced burgers in 2D ...
\begin{eqnarray}
  \label{eq:burgers}
  \new{\frac{\partial u_x}{\partial{t}} + \vu \cdot \nabla u_x =
  \nu \nabla\cdot \nabla u_x + g_x(t)}, 
  \quad 
  \new{\frac{\partial u_y}{\partial{t}} + \vu \cdot \nabla u_y =
  \nu \nabla\cdot \nabla u_y + g_y(t)},
\end{eqnarray}
where $\nu$ and $\mathbf{g}$ denote diffusion constant and external forces, respectively.
Our setup resembles a 2D variant of the tests employed
by the work on learning data-driven discretizations \cite{barsinai2019data};
correspondingly, we extend the forcing terms described there to 2D.
We generate the forces from 20 overlapping sine functions each with a random direction, amplitude, and phase shift:
%
\begin{equation}
    g_x(t) = \sum_{i=1}^{20} \cos(\alpha_i) a_i \sin(\omega_i t - k x + \phi_i),
    \quad
    g_y(t) = \sum_{i=1}^{20} \sin(\alpha_i) a_i \sin(\omega_i t - k x + \phi_i).
\end{equation}
%
This PDE scenario does not involve any equality constraints, i.e., $\mM=0$.

Similar to the previous scenarios, we discretize the system 
on a staggered grid and compute the
advection operator with a semi-Lagrangian scheme \cite{stam1999}.
The domain has a square, normalized size of $1 \times 1$
with reference trajectories computed via a resolution of
$d_{r,x}\!=\!d_{r,y}\!=\!128$. The source domain correspondingly uses 
$d_{s,x}\!=\!d_{s,y}\!=\!32$.

\paragraph{Training Procedure and Results}

As training data, ten simulations of 200 steps each are used.
An example sequence of the data is shown in \myfigref{fig:appx:burgers-images-train}.
The SOL and NON models are trained for 38.4k steps with a batch size of \new{five}
with a learning rate of $10^{-4}$,
while the %NON and PRE models are 
PRE model is trained for 25k steps with a batch size of 32
using an initial learning rate of $10^{-3}$ that was lowered to 
$5\times10^{-7}$ over the course of the training. %These variants 
The PRE model additionally %use
uses 5\% of the training data set for validation.
%
The test data set contains five cases with different initial conditions 
and force fields over the course of 200 time steps.
All models use a neural network architecture 
with five ResBlocks with 32 features each.
% leave out for now: "that contains ca. 261k trainable weights."

As summarized in the main text, the learned correction functions can 
significantly decrease the numerical errors of the source simulation. 
Across the different test cases (partly shown in \myfigref{fig:appx:burgers-images-test}),
the best models achieve a 
reduction by over 67\%. The corresponding MAE measurements 
are given in \mytabref{tab:burgers}, 
and \myfigref{fig:appx:burgers-plot}
provides an overview of the performance per test case.
While the PRE model shows a lower performance, most likely due to an overly 
strong temporal regularization, the NON model is close to the best 
SOL model in this case with an MAE of 0.159 compared to 0.148 for \sol{2}.
%
Interestingly, this behavior matches the results of Bar-Sinai et al. \cite{barsinai2019data}.
They experimented with up to \new{eight} recurrent steps 
of a 1D Burgers' simulation, but did not report significant advantages from training 
with the 1D solver in the loop.

\new{
  In contrast, we found that more interactions show their advantage in a
  deterministic scenario, where we exclude the external forces from the Burgers'
  equation above, i.e., \myeqref{eq:burgers}. 
  As this versions exhibits less chaotic behavior,
  the SRC version generally shows smaller errors
  compared to the SRC version in the forced scenario. 
  The SOL versions now yield 
  further improvements when trained with more look-ahead:
  \sol{4} yields an improvement of 10\% over SRC, \sol{16} yields 12\%,
  while the \sol{32} version reduces the error by 17\%. % (MAE_SRC - MAE_SOL32)/MAE_SRC
  \mytabref{tab:burgers} shows the corresponding MAE measurements.
}

Our results highlights that deep learning via physical 
simulations works particularly well when the ANNs can actually learn 
to predict the behavior of the dynamics and, thus, compensate for the 
numerical errors that will occur. If, on the other hand, external and 
unpredictable influences such as the randomized forcing terms dominate the behavior,
the model has a reduced chance to predict the right correction function.

%\todo{optional - show errors for non-forced version if there's time...}

\begin{table}[htb]
  \caption{Quantitative evaluation of different models for the forced
    advection-diffusion scenario. \new{MAE values without forcing are given with
      a $\times 100$ factor.}}
  \label{tab:burgers}
  \begin{center}
    \begin{tabular}{ccccccc}
      \toprule
      \multicolumn{7}{c}{\textbf{MAE Velocity}, Mean (std. dev.)}                       \\
      \midrule
      With forcing    & SRC     & PRE     & NON     & SOL$_2$ & SOL$_4$    & SOL$_8$    \\
      \cmidrule{2-7}
                      & 0.248   & 0.218   & 0.159   & 0.148   & 0.152      & 0.158      \\
                      & (0.019) & (0.017) & (0.015) & (0.016) & (0.015)    & (0.017)    \\
      \midrule
      Without forcing & SRC     & NON     & SOL$_4$ & SOL$_8$ & SOL$_{16}$ & SOL$_{32}$ \\      
      \cmidrule{2-7}
      $(\times 100)$  & 0.306   & 0.272   & 0.276   & 0.277   & 0.268      & 0.253      \\
                      & (0.020) & (0.028) & (0.037) & (0.040) & (0.030)    & (0.020)    \\
      \bottomrule
    \end{tabular}
  \end{center}
\end{table}

\begin{figure}[tb]
  \centering
  % 1400 x 200 image = 100 x 14.3 overpic unit
  \begin{overpic}[width=0.96\linewidth]{figs/summary-burgers-training-data.jpg}
    \put(-1.6,0)   {\scriptsize\rotatebox{90}{\makebox[7.14\unitlength]{Ref.}}}
    \put(-1.6,7.14){\scriptsize\rotatebox{90}{\makebox[7.14\unitlength]{SRC}}}
    \put(100.5,0){\includegraphics[height=7.14\unitlength]{figs/summary-legend-burgers}}
  \end{overpic}
  \caption{An example sequence from the training data set of the forced
    advection-diffusion test case.}
  \label{fig:appx:burgers-images-train}
\end{figure}

\begin{figure}[tb]
  \centering
  % 1400 x 300 image = 100 x 21.43 overpic unit
  \begin{overpic}[width=0.96\linewidth]{figs/summary-burgers-test-data.jpg}
    \put(100,19)    {(a)}
    \put(-1.6,0)    {\scriptsize\rotatebox{90}{\makebox[7.14\unitlength]{Ref.}}}
    \put(-1.6,7.14) {\scriptsize\rotatebox{90}{\makebox[7.14\unitlength]{\sol{2}}}}
    \put(-1.6,14.28){\scriptsize\rotatebox{90}{\makebox[7.14\unitlength]{SRC}}}
    \put(100.5,0){\includegraphics[height=7.14\unitlength]{figs/summary-legend-burgers}}
  \end{overpic}
  \\\vspace{0.5em}
  \begin{overpic}[width=0.96\linewidth]{figs/summary-burgers-test-data-b.jpg}
    \put(100,19)    {(b)}
    \put(-1.6,0)    {\scriptsize\rotatebox{90}{\makebox[7.14\unitlength]{Ref.}}}
    \put(-1.6,7.14) {\scriptsize\rotatebox{90}{\makebox[7.14\unitlength]{\sol{2}}}}
    \put(-1.6,14.28){\scriptsize\rotatebox{90}{\makebox[7.14\unitlength]{SRC}}}
    \put(100.5,0){\includegraphics[height=7.14\unitlength]{figs/summary-legend-burgers}}
  \end{overpic}
  \\\vspace{0.5em}
  \begin{overpic}[width=0.96\linewidth]{figs/summary-burgers-test-data-c.jpg}
    \put(100,19)    {(c)}
    \put(-1.6,0)    {\scriptsize\rotatebox{90}{\makebox[7.14\unitlength]{Ref.}}}
    \put(-1.6,7.14) {\scriptsize\rotatebox{90}{\makebox[7.14\unitlength]{\sol{2}}}}
    \put(-1.6,14.28){\scriptsize\rotatebox{90}{\makebox[7.14\unitlength]{SRC}}}
    \put(100.5,0){\includegraphics[height=7.14\unitlength]{figs/summary-legend-burgers}}
  \end{overpic}
  \caption{
    % \todo{placeholder! insert burgers}
    % Kiwon - done
    Time steps of three test cases (a)-(c) from the forced advection-diffusion scenario.
      % From top to bottom: SRC, \sol{2}, and reference.
    }
  \label{fig:appx:burgers-images-test}
\end{figure}

\newcommand{\myFigHBurgers}{0.35\columnwidth}
\begin{figure}[htb]
  \centering
  \subcaptionbox{Velocity error per test case}{\includegraphics[height=\myFigHBurgers,page=1]{figs/summary-burgers-plot}}
  \subcaptionbox{Velocity improvement (relative to SRC) per test case}{\includegraphics[height=\myFigHBurgers,page=4]{figs/summary-burgers-plot}}
  % \includegraphics[width=0.45\columnwidth,page=1]{figs/summary-burgers-plot}
  \caption{Separate evaluations for five different test cases of the forced advection-diffusion scenario.}
  \label{fig:appx:burgers-plot}
\end{figure}

%... results for Burgers ...

% \paragraph{Discussion}
% ... optionally discuss ... probably not for Burgers




% --- CG Solver

\clearpage

%\subsection{Comparison of Neural-Network-generated initial guesses for CG-Solve}
\subsection{Inference of Initial Guesses for Conjugate Gradient Solvers}\label{app:expCgSolver}
% app-cgsolver

%\newcommand{\res}{\mathbf{r}} % \vr is taken for the reference states
% makig all scalar values non-bold
\newcommand{\res}{\new{r}} % \vr is taken for the reference states
\newcommand{\press}{\new{p}} 
\newcommand{\pressh}{\new{\hat{\press}}} 

In this section, we investigate the interaction of learning models with conjugate gradient (CG) solvers \cite{hestenes1952cg}. 
%As a sample problem, we used two-dimensional Navier-Stokes flow.
We target Poisson problems, 
which often arise many PDEs, e.g., in electrostatics or % to compute the pressure $\press$
in fluid flow problems where the pressure $\press$ is computed
via $\nabla \cdot \nabla \press = \nabla \cdot \vu$.
Specifically, we explore the iteration behavior of the CG solver given an initial state predicted by a trained model.
To this end, we compare three main methods: A solver-in-the-loop (\sol{n}) approach, a non-interacting supervised approach (NON),
and a differentiable physics-based (\sol{\text{DIV}}), which is trained to directly minimize the PDE residual.
%approach that directly aims for the same physical goal as the CG solver.
%
In general, the CG solver iterations converge toward a reference pressure field $\press$ such that $\mA \press = \nabla \cdot \vu$ with
$\mA=\nabla \cdot \nabla$. For an intermediate solution $\pressh$, the residual $\res = \nabla \cdot \vu - \mA \pressh$ measures how far away the approximated pressure $\pressh$ is from the true solution.
Thus, as the solver converges, $\res$ decreases and the difference $\pressh - \press$ converges to zero. In the following, we employ the neural network $\corr$ to infer 
a pressure field given a velocity sample $\vu$, i.e., $\pressh=\corr(\vu)$.
%
We focus on 2D cases, i.e., $\vu{} \in \mathbb{R}^{2 \times d_{x} \times d_{y}}$ and
$\press,\res \in \mathbb{R}^{d_{x} \times d_{y}}$.

\paragraph{Loss Functions}
%\myeqref{eq:cg_loss1} (NON) 
The NON version employs a regular supervised loss, i.e., the difference of the predicted pressure $\hat{p}$ from the pre-computed reference pressure $p$ for $j$ different samples:
\begin{equation} \label{eq:cg_loss1}
    \new{\loss_{\text{NON}} = \Vert \corr(\vu) - \press \Vert^2}.
\end{equation}

% \begin{equation}
%     \loss_{\text{NON}} = \Vert \hat{p} - p \Vert^2.
% \end{equation}
%\myeqref{eq:cg_loss2} 

We additionally compare to a variant that is often referred to as \emph{unsupervised} in previous work,
and which is in line with other physics-based or physics-informed loss constructions \cite{raissi2018hiddenfluid,sirignano2018dgm}.
Specifically, the \sol{\text{DIV}} version
replicates the setup described in
\cite{tompson2017} and uses the PDE $\nabla^2 p - \nabla \cdot \vu = 0$ as loss for the training of a neural network.
%
Given an input velocity $\vu^*$, the goal is to infer a 
pressure function $\hat{p}(\nabla \cdot \vu^*)$ such that the PDE residual is minimized:
\begin{equation}
    \begin{aligned}
    \new{\loss_{\text{SOL}_{\text{DIV}}} 
        = \Vert \nabla \cdot \vu^* - \nabla\cdot\nabla \corr(\vu)  \Vert^2}.
    \end{aligned}
    \label{eq:cg_loss2}
\end{equation}
This version represents a different form of
differentiable PDE solvers, namely including them in the loss formulation, and hence we denote it with \sol{\text{DIV}}. However, due to a lack of iterating calculations for this variant,
a more appropriate name would be \emph{``solver-in-the-loss''} rather than \emph{``solver-in-the-loop''}.

%        &= \sum_{j} \Vert \nabla (\vu_j^* - \nabla \corr(\vu_j))           \Vert^2\\
%        &= \sum_{j} \Vert \nabla \vu_j^* - \Delta \corr(\vu_j)             \Vert^2\\
% \begin{equation}
%     \loss_{\text{SOL}_{\text{DIV}}} =\Vert \nabla^2 \hat p - \nabla \cdot \vu \Vert^2
% \end{equation}

As a third variant, we employ a solver-in-the-loop interaction that 
employs a differentiable CG solver and uses a learning objective 
to minimize the PDE residual after $n$ iterations of the CG solver. In this scenario,
$\pdec$ represents a linear operator, i.e., one step of the CG method 
to approximate 
%$\press = (\nabla \cdot \nabla)^{-1} \nabla \cdot \vu$,
$\nabla^{-2} \, (\nabla \cdot \vu)$,
and the loss function is given by:
\begin{equation} 
  %\loss_{SOL_{k}} = \sum_{j} \{\hat{p}_{+k} - \hat{p}\}_{j}^2 \label{eq:cg_loss3}
  \new{\loss_{\text{SOL}_{n}} = \Vert \pdec^n( \corr(\vu) ) - \corr(\vu) \Vert^2}.
  \label{eq:cg_loss3}
\end{equation}
%\myeqref{eq:cg_loss3} (\sol{n}) involves the CG solver directly in the loss function. 
%
% The solver converges toward $p$, such that $Ap = \nabla u$. For an intermediate solution $p_i$, the residual $r_i = \nabla u - Ap_i$ measures how far away the current pressure $p_i$ is from the true solution. So, as the solver converges, $r_i$ and the difference $p_{i+k} - p_i$ diminish.
%
%The residual $r_i$ is the finite difference estimate of $\nabla u^*_{t+1}$, the divergence of the velocity in the next time step.
Thus, the \sol{n} and \sol{\text{DIV}} both minimize the same residual divergence $\res$; while the \sol{\text{DIV}} loss aims to do so directly, the \sol{n} version instead sees how the iterative solver performs.
At training time, the \sol{n} variant receives gradients through $n$ iterations of
the iterative solver via back-propagation.

\paragraph{Training Procedure} 
The trained models in this section all use the same convolutional U-net architecture \cite{unet2015} with 22 layers of strided convolutions and 5$\times$5 kernels, containing around 127k trainable parameters (see \myappref{app:models} for details).
%
The training data set was generated using the conjugate gradient solver from the $\Phi_\textrm{Flow}$ framework \cite{holl2020}. 
It is comprised of 3k fluid simulations on a domain with $d_{x} = d_{y} = 64$ and closed boundaries. Each simulation consists of a randomly generated density and velocity field, which are integrated over time for 16 steps. 
%A buoyancy factor of 0.1 and incompressible physics scheme are used.
%
Each model was trained for 300k steps with a learning rate of $2 \times 10^{-4}$ and training batch size of 16. 
The reference solutions were pre-computed with a CG solver using an accuracy threshold of $10^{-6}$ for the residual norm.
%validation batch size of 32 and on the same data set. \todo{what fraction for validation samples?}

\paragraph{Results}
%
We now compare the different loss functions by their performance in conjunction with the CG solver.
We compute averages for 100 test cases each time, i.e., samples that were not seen at training time.
As baseline, we denote a CG solving process that starts from a zero guess as SRC.

We first compare how many CG iterations are required to reach a certain target accuracy given the inferred solutions by the three different types of models.
The results are shown in \mytabref{tab:cgresults} and visualized in \myfigref{fig:cgresults}.
%
Initially, \sol{\text{DIV}} reaches an accuracy of almost $10^{-2}$, closely followed by \sol{5}.
While the supervised NON version produces pressure predictions that seem quite close to the reference, its initial accuracy is only slightly better than the zero guess employed by SRC.
This is due to the error being measured locally per grid point, while the correctness of larger structures becomes more important after in interactions with the CG solvers.
%
Over the first five to ten CG iterations, the accuracy of \sol{5} improves very quickly, overtaking the other methods.
To reach an accuracy of $10^{-2}$, the CG solver requires an average of around two steps in conjunction with \sol{5}, nine steps with NON, 28 steps with \sol{\text{DIV}} and 78 steps 
starting from zero.
When running the CG solver for more iterations, the accuracy increases similarly for all methods, with \sol{5} retaining its advantage. 

% I dont think we need to discuss the NON version too much:
%NON also rapidly gains in accuracy since the CG iterations can correct its small-scale errors very quickly.
% with \sol{\text{DIV}} catching up to NON after around 150 steps in our tests.
%
%
%The \sol{\text{DIV}} approach requires the most iterations for higher accuracies, followed by the supervised version (NON), while the solver-based approach (\sol{5}) outperforms both.
%For lower accuracies ($10^{-1}, 10^{-2}$), the direct PDE loss of \sol{\text{DIV}} does better than the supervised one, not requiring any additional solver iterations.
%Across the different accuracy thresholds, the \sol{5} version requires fewer solver iterations than all other methods.

%Figure \ref{fig:cgresults} b) shows how the residual divergence develops on average over the course of CG iterations given initial states predicted by the different models.
%The \sol{5} and NON curves both show a steep initial decline, allowing the solver to converge to a higher accuracy much more quickly.  
%The \sol{\text{DIV}} version actually perform best initially, but the convergence completely stagnates over the course of the first ca. twenty iterations.

%leads to a slight increase over the course of the first iterations, in this way loosing its advantage. 
%causes a slight increase in the residual in the beginning, implying the solver has to undo some errors in the prediction. 
%However, looking at the start of the residual curves, the \sol{\text{DIV}} guess itself has the lowest residual divergence.
%These findings suggests that the prediction of the Tompson-based network itself is better than the other networks' if it were directly used to correct the input. However, once fed into the solver to further improve the guess, it does not speed up the convergence behavior as well as the other methods. 
%This is likely due to the fact that it cannot account for the CG Solver's boundary conditions given only the divergence. Tompson et. al. therefore also gave their network geometry data as an additional input \cite{tompson2017}.
%The \sol{n} and NON versions however learn this without the need for any extra modifications, as they see solver-generated data during training.

Comparing \sol{5} to \sol{\text{DIV}} shows the importance of training with the solver in the loop: the \sol{\text{DIV}} model does not 
receive any feedback regarding the behavior of the solver.
It predicts solutions that satisfy the loss -- measured per grid point -- but do not match the large-scale structures of the true solution.
Consequently, this task is left to the CG solver, which requires many iterations to work out the correct global solution.
The \sol{5} model, however, sees the corrections performed by the CG solver at training time and can learn to adjust its guess accordingly.

When investigating the inferred pressure fields themselves (\myfigref{fig:cg_img_results}), we see that the guesses of the \sol{5} model come closest to the reference, followed by those of the NON variant. 
The \sol{\text{DIV}} differs more strongly,
%version's predictions show a relatively strong difference from the reference data. 
and the residual divergence, shown in \myfigref{fig:cg_residual_img}, highlights that it % the \sol{\text{DIV}} approach 
has a noticeable error pattern near the outer border of the domain. This provides an explanation for the poor 
behavior of the \sol{\text{DIV}} model for the initial CG solver iterations:
while it minimizes the PDE-based loss in an absolute sense, it does not receive information about 
how different parts of the solution influence the future iterations of the solver. This ambiguity
is alleviated to some extent by the pre-computed reference solutions for NON, but especially 
the \sol{5} version receives this feedback in terms of gradient from the differentiable solver
and, in this way, can best adapt to the requirements for future iterations.

%\paragraph{Look-Ahead}
%
We also experimented with varying the number of look-ahead steps for \sol{n} models in the loss function of \myeqref{eq:cg_loss3}. 
This ablation study (\myfigref{fig:cg_it_stepsizes}) shows how too few iterations clearly deteriorate the performance, while more than 5 iterations 
lead to a slight increase in the required iterations. We assume that this behavior is potentially caused by evaluating the loss 
only for the final output of the $n$ iterations.
%There is a natural incentive to use a look-ahead that is as small as possible, so that training requires less computation. 
%However, our ablation shows that too a look-ahead that is too small leads to significantly worse iteration behavior. Larger look-ahead also stop yielding an improvement at some point and even performed worse in some cases. This last observation could however be due to the fact that our \sol{n} loss only penalizes the difference between $\hat{p}$ and $\hat{p}_{+k}$, disregarding the effect the guess has on the intermediate solutions in between.

\paragraph{Discussion}
%
Our results highlight the advantages of training with the solver in the loop for fully 
implicit PDE solvers. Likewise, it shows that a physics-informed loss formulation alone 
yields only a partial view of the problem.
%even compared to physics-informed loss functions.
While a loss-based residual cannot adapt to iterative algorithms, the solver-in-the-loop 
models directly receive gradient-based feedback at training time.

The combination of an inferred initial guess with a traditional solver represents a particularly interesting
hybrid algorithm, as it gives convergence guarantees that a learned approach alone would not be able to provide.
Even if a trained model generates a sub-optimal solution, the solver can improve the solution 
until it matches the desired accuracy threshold. On the other hand, pre-training a model for a known problem domain can significantly reduce the required number of iterations and, consequently, reduce the workload in scenarios where PDEs from the same problem domain need to be solved repeatedly and in large numbers.
Here, the current hardware developments provide an additional promise: the advances in terms of highly specialized hardware for evaluating neural networks
can provide a substantial future speed-up even for a fixed, pre-trained model.

% The \sol{\text{DIV}} version we investigated directly aims for the physical goal of our sample problem by including it in the loss function via differentiable physics. This results in a better solution if the network's output itself is used. However, in terms of being used as an initial state for the CG solver to iterate upon, it fares noticeably worse. A basic supervised approach (NON) on the other hand, displays good iteration behaviour yet does not achieve a very low-divergence prediction as easily. Our Solver-in-the-Loop (\sol{5}) approach consistently outperforms the supervised approach, shows a relatively low-divergence initial prediction and also provides the best iteration behavior. It also does not require special handling of boundary conditions.
% %
% As such, involving differentiable fluid physics directly in the training seems promising, as it could lead to a speed up while still obtaining any desired accuracy by feeding the network's prediction back into a solver.

\begin{figure}[htb]
  \centering
  \begin{overpic}[width=0.45\linewidth]{figs/cg-iterations}
    \put(-5, 55) {(a)}
  \end{overpic}
  \hspace{0.05\linewidth}
  \begin{overpic}[width=0.45\linewidth]{figs/cg-residuum-iterations}
    \put(-5, 55) {(b)}
  \end{overpic}
  \caption{(a) Iterations needed to reach target accuracy and (b) comparison of
    maximum residual error over iterations.}
  \label{fig:cgresults}
\end{figure}

\begin{figure}[htb]
  \centering
  \begin{overpic}[width=0.56\linewidth]{figs/cg-pressure-images}
    \put(-1, 55) {(a)}
  \end{overpic}
  \begin{overpic}[width=0.43\linewidth]{figs/cg-pressure-differences}
    \put(-5, 73) {(b)}
  \end{overpic}
  \caption{(a) Sample outputs of the models and (b) difference of output from
    reference.}
  \label{fig:cg_img_results}
\end{figure}

\begin{figure}[htb]
  \centering
  \includegraphics[width=0.49\columnwidth]{figs/cg-residuum-img1}
  \includegraphics[width=0.49\columnwidth]{figs/cg-residuum-img2}
  \caption{Residual error after one CG solver iteration.}
  \label{fig:cg_residual_img}
\end{figure}

\begin{figure}[htb]
  \centering
  \includegraphics[width=0.49\columnwidth]{figs/cg-iterations-stepsizes}
  \includegraphics[width=0.49\columnwidth]{figs/cg-residuum-stepsizes}
  \caption{Comparison of % \sol{n} networks with different look-ahead $n$.
    SOL models with different look-ahead steps.}
  \label{fig:cg_it_stepsizes}
\end{figure}

\begin{table}[htb]
  \caption{Evaluation of the CG solver performance for different models.}
  \label{tab:cgresults}
  \begin{center}
    \begin{tabular}{ccccccc}
      \toprule
      \textbf{Model}   & \multicolumn{6}{c}{\textbf{Iterations for Accuracy}, Mean (std. dev.)}     \\
      \cmidrule{2-7}
                       & $10^{-1}$ & $10^{-2}$ & $10^{-3}$ & $10^{-4}$ & $10^{-5}$ & $10^{-6}$ \\
      \midrule
      NON              & 1.67      & 9.33      & 52.16     & 109.12    & 155.37    & 186.12    \\
                       & (1.010)   & (5.428)   & (17.540)  & (15.875)  & (10.155)  & (5.719)   \\
      \sol{\text{DIV}} & 0.0       & 27.79     & 79.06     & 117.97    & 155.76    & 181.07    \\
                       & (0.0)     & (15.255)  & (10.042)  & (13.234)  & (9.403)   & (6.052)   \\
      \sol{5}          & 0.03      & 1.97      & 29.59     & 88.37     & 133.59    & 167.37    \\
                       & (0.171)   & (1.118)   & (14.832)  & (13.465)  & (11.605)  & (8.549)   \\
      \bottomrule
    \end{tabular}
  \end{center}
\end{table}


% end CG solver part

% ---

% \subsection{Performance}
% \label{app:performance}

% update?}
% Here, we give details about the performance of our method to illustrate the
% gains that can be achieved by evaluating our trained model.
% \mytabref{tab:performance} shows the timings measured from different simulations
% of each example. All experiments were performed on an Intel Xeon E5-1620 3.70
% GHz processor with 128 GB memory. The trained \ac{nn} models were evaluated with
% the CUDA support and TensorFlow on an 
% NVIDIA GeForce GTX 960 GPU.  % or cube?
% The reference is a full simulation with four times higher resolution
% evaluated with an optimized CPU-based solver (the same one using to generate the
% input for our method).

% \vspace{1em}
% \begin{table}[htb]
%   \caption{Performance comparison. Simulation timings (in seconds) are measured
%     for 1,000 simulation steps. Here, {\em un.} and {\em sup.} denote the unsupervised and
%     supervised learning models, respectively.}
%   \label{tab:performance}
%   \begin{center}
%     \begin{tabular}{ccrrr}
%       \toprule
%       {\bf Example}                                & {\bf Test} & \multicolumn{1}{c}{\bf Basic} & \multicolumn{1}{c}{\bf Reference} & \multicolumn{1}{c}{\bf Ours} \\
%       \midrule                                                                                                                  
%       Rising smoke (\myfigref{fig:phif-cplxsmoke}) & A          & 9.40                          & 104.46                            & 17.56 (un.) 19.91 (sup.)     \\
%                                                    & B          & 9.87                          & 117.21                            & 17.12 (un.) 18.96 (sup.)     \\
%       Liquid stream (\myfigref{fig:tf-stream})     & A          & 18.00                         & 85.49                             & 29.86 (sup.)                 \\
%                                                    & B          & 19.58                         & 84.10                             & 31.82 (sup.)                 \\
%       \bottomrule
%     \end{tabular}
%   \end{center}
% \end{table}


% ---

\clearpage

\subsection{Three-dimensional Unsteady Wake Flow}\label{app:expKarman3d}
% app-karman3d

As a final scenario, we target a three-dimensional %\ku{(3D)} - I dont think that's necessary
fluid flow problem.
The third spatial dimension leads to a large increase in terms of 
degrees of freedom, especially in the finer reference manifold. Additionally,
the three axes of rotation lead to significantly more complicated flow 
structures. 

Overall, we target a setup that represents an extension of the 2D %two-dimensional
unsteady wake flow case of \myappref{app:expKarman2d}. Instead of a circular 
obstacle, the flow now faces a cylindrical obstacle in a 3D domain with 
extent of $1 \times 1 \times 2$. The cylinder 
with diameter $0.1$ is located at position $(1/2,1/2,0)^T$ and has an extent 
of 1 unit along the z-axis. We use the incompressible Navier-Stokes 
equations in three dimensions as underlying PDE:  
\begin{eqnarray}
  \label{eq:model-ns3d}
  \frac{\partial u_x}{\partial{t}} + \vu \cdot \nabla u_x =
  - \frac{1}{\rho}\nabla{p} + \nu \nabla\cdot \nabla u_x 
  \nonumber
  \\
  \frac{\partial u_y}{\partial{t}} + \vu \cdot \nabla u_y =
  - \frac{1}{\rho}\nabla{p} + \nu \nabla\cdot \nabla u_y 
  \nonumber
  \\
  \frac{\partial u_z}{\partial{t}} + \vu \cdot \nabla u_z =
  - \frac{1}{\rho}\nabla{p} + \nu \nabla\cdot \nabla u_z 
  \nonumber
  \\
  \text{subject to} \quad \nabla \cdot \vu = 0.
\end{eqnarray}

For reference simulations, the domain is discretized with 
$d_{r,x}\!=\!d_{r,y}\!=\!128$ and $d_{r,z}\!=256$ cells using a staggered layout for the velocity components. 
The source domain has a resolution of
$d_{s,x}\!=\!d_{s,y}\!=\!32$ and $d_{r,z}\!=64$ cells. Data sets from both domains contain 
phase space trajectories of 500 time steps.
For the training data, the viscosity coefficient $\nu$ is chosen
to yield Reynolds numbers 
Re$_{\text{train}} \in \{ 58.6, 78.1, 117.2, 156.3, 234.4, 312.5, 468.8, 625.0 \}$.
%.i.e., there is a factor of more than 30 between smallest, and largest Reynolds numbers in the training data.
% Re$_{\text{test}} \in \{146.484375 , 292.96875 , 585.9375 , 1171.875, 2343.75\}$,
% which are denoted as $\times1$, $\times2$, $\times4$, $\times8$, and $\times16$ below, respectively.
%still factor larger than 10, and much higher complexity due to 3D vortex structures;
While the range of Reynolds numbers covers a slightly reduced range 
compared to the 2D case, there is still a factor of more than ten between 
largest and smallest ones, and the 3D nature of the flow introduces
a significant amount of complexity. The example visualizations of 
a training data set in \myfigref{fig:appx:karman3d-images-train} highlight the complexity 
of the flows.

For the test set, we use different Reynolds numbers, namely 
Re$_{\text{test}} \in \{$68.4, 97.7, 195.3, 136.7, 273.4, 390.6, 546.9$\}$.
The following test evaluations were computed for the seven Reynolds numbers in Re$_{\text{test}}$
over 300 time steps. Numeric values are given in \mytabref{tab:karman3d}.


\paragraph{Training Procedure} 

For the 3D case, we use a ResNet that largely follows the architecture of the 2D 
cases, but employs 3D convolutions instead. The ResNet contains six blocks 
with kernel sizes of 5$\times$5$\times$5 and 3$\times$3$\times$3 for the two convolutional layers per block.
The number of filters is increases to 48 in the center of the network, 
yielding 1002k trainable parameters (also see \myappref{app:models}). 
\new{As for the 2D case, the inputs for the 3D models contain a constant field indicating 
the targeted Reynolds number.}
%
All models were trained for 300k iterations using a
learning rate of $10^{-4}$ and a batch-size of four.
We then use three validation simulations with
Re$_{\text{val}} \in \{$61.0 , 305.2 , 470.0$\}$ to select the best performing model.
%    Re = [100000,500000,770000] -> convert! 10/(128*128/(Re)) = 61.0 , 305.175 , 469.97
% 100 epochs = 2*10 steps only!
% 750 for m01! but epochs written only every 20 steps! -> 750*20*20 iters, 300k!

Due to the increased computational workload to train the 3D models,
we focus on a NON variant and a \sol{16} version, which 
uses the same differentiable Navier-Stokes solver for producing gradient 
information over the course of up to 16 unrolled simulation and inference steps 
for each iteration at training time. This version was trained 
with \sol{8} for 200k iterations and then for an additional 100k iterations 
as \sol{16}.


\paragraph{Results} 

The 3D flow represents a significant increase in terms 
of complexity for the deep learning models. Among others, we were not 
able to train a stable NON version despite numerous tests. While 
the models performed well for ca. 100 to 150 time steps, small scale oscillations
induced by the corrections accumulate and start to strongly distort the flow.
This is a good example of the undesirable shift of distributions for the inputs:
once the phase space trajectories produced by the hybrid method leave the 
distribution of the regular source states seen at training time, the 
model fails to infer reasonable corrections.

In contrast, the \sol{16} version retains its stability over the course 
of long simulations with several hundred steps. This is reflected in the MAE measurements 
of the velocity fields over the test cases: the regular source simulation induces 
an error of 0.167, which the NON version reduces to 0.143. The \sol{16}
reduces the error to 0.130 instead, which however only gives a partial view of 
the overall behavior of the different versions. 
The graphs over time shown in \myfigref{fig:appx:karman3d-freq:mae}
illustrate the diverging behavior of the NON version. While it does very well initially, 
even slightly surpassing \sol{16} around frame 100, the errors quickly grow 
afterwards, eventually leading to a performance that is worse than the source simulation.

The frequency graphs of the kinetic energy in \myfigref{fig:appx:karman3d-freq:ke},
measured for an array of $5^3$ sample points at the center of the domain,
instead show that the \sol{16}
simulations closely match the frequency distribution of the reference simulations. 
% maybe it's enough in the caption below...
% While this version shows an increased offset in terms of the mean
% (initial points of SRC and \sol{16} curves in \myfigref{fig:appx:karman3d-freq:ke} are colocated 
% above the reference}), 
It succeeds in restoring the change of frequencies across the different temporal scales of the flow
significantly better than the SRC and NON models.
%\mynote{kiwon}{I couldn't get the point of what is written in the parentheses.}
The source simulation instead underestimates larger frequencies and over-estimates 
smaller ones.

\myfigref{fig:appx:karman3d-images-test} visualizes the vorticity magnitude of 
several test cases with Reynolds numbers not seen during training.
% with Re=\todo{X} from test set.
%\ku{from the test set when Re=\todo{X}.}
The \sol{16} model manages to correct the vortex shedding behavior of the source simulation 
and closely matches the reference. As we visualize in the supplemental video,
the NON version starts to oscillate, injecting undesirable 
distortions into the velocity field.

%\todo{include m08 model?}
%\todo{show and discuss time error / time frequency plots}
%large scale freq errors in k3d - not accounted for? larger kernel?

\begin{figure}[htb]
  \centering
  % 1717 x 400 image = 100 x 23.3 overpic unit
  \begin{overpic}[width=1.0\linewidth]{figs/karman3d-train01}
    \put(100,17)      {(a)}
    \put(-1.6,0)    {\scriptsize\rotatebox{90}{\makebox[11.65\unitlength]{Reference}}}
    \put(-1.6,11.65){\scriptsize\rotatebox{90}{\makebox[11.65\unitlength]{SRC}}}
    \put(100.5,0){\includegraphics[height=11.65\unitlength]{figs/summary-legend-karman3d}}
  \end{overpic}\vspace{10pt}
  \begin{overpic}[width=1.0\linewidth]{figs/karman3d-train04}
    \put(100,17)      {(b)}
    \put(-1.6,0)    {\scriptsize\rotatebox{90}{\makebox[11.65\unitlength]{Reference}}}
    \put(-1.6,11.65){\scriptsize\rotatebox{90}{\makebox[11.65\unitlength]{SRC}}}
    \put(100.5,0){\includegraphics[height=11.65\unitlength]{figs/summary-legend-karman3d}}
  \end{overpic}
  \caption{Two example sequences with (a) Re=117.2 and (b) Re=273.4 of the
    three-dimensional wake flow from the training data set. 
    Each row shows 200 time steps for SRC (top) and
    reference versions (bottom) in terms of vorticity magnitude.
    }
  \label{fig:appx:karman3d-images-train}
\end{figure}

\newcommand{\myFigHKthree}{0.37\columnwidth}
\begin{figure}[htb]
  \centering
  \subcaptionbox{Velocity error over time\label{fig:appx:karman3d-freq:mae}}{\includegraphics[height=\myFigHKthree,page=15]{figs/out-vgf-karman3d-time}}
  \subcaptionbox{Frequency error\label{fig:appx:karman3d-freq:ke}}{\includegraphics[height=\myFigHKthree,page=16]{figs/out-vgf-karman3d-time}}
  \caption{Evolutions of velocity MAE and frequency errors over the course of 300 time steps 
    averaged for the seven test cases of the three-dimensional wake flow. (a) The NON versions perform well initially,
    but strongly diverges for later frames. (b) The \sol{16} shows a clearly improvement in terms of the frequency 
    distribution of the kinetic energies. The overall curve of \sol{16} closely follows the reference with an initial 
    offset over the reference, % that it 
    which inherits from the source simulation.
    }
  \label{fig:appx:karman3d-freq}
\end{figure}

\begin{figure}[tb]
  \centering
  % 1717 x 600 image = 100 x 34.94 overpic unit
  %\vspace{105pt} % acts after!
  \begin{overpic}[width=1.0\linewidth]{figs/karman3d-test00}
    \put(100,22)   {(a)}
    \put(-1.6,0)   {\scriptsize\rotatebox{90}{\makebox[11.65\unitlength]{Ref.}}}
    \put(-1.6,8.0) {\scriptsize\rotatebox{90}{\makebox[11.65\unitlength]{\sol{16}}}}
    \put(-1.6,16.0){\scriptsize\rotatebox{90}{\makebox[11.65\unitlength]{SRC}}}
    \put(100.5,0)  {\includegraphics[height=11.65\unitlength]{figs/summary-legend-karman3d}}
  \end{overpic}\vspace{10pt}
  \begin{overpic}[width=1.00\linewidth]{figs/karman3d-test03}
    \put(100,22)   {(b)}
    \put(-1.6,0)   {\scriptsize\rotatebox{90}{\makebox[11.65\unitlength]{Ref.}}}
    \put(-1.6,8.0) {\scriptsize\rotatebox{90}{\makebox[11.65\unitlength]{\sol{16}}}}
    \put(-1.6,16.0){\scriptsize\rotatebox{90}{\makebox[11.65\unitlength]{SRC}}}
    \put(100.5,0)  {\includegraphics[height=11.65\unitlength]{figs/summary-legend-karman3d}}
  \end{overpic}\vspace{10pt}
  \begin{overpic}[width=1.00\linewidth]{figs/karman3d-test06}
    \put(100,22)   {(c)}
    \put(-1.6,0)   {\scriptsize\rotatebox{90}{\makebox[11.65\unitlength]{Ref.}}}
    \put(-1.6,8.0) {\scriptsize\rotatebox{90}{\makebox[11.65\unitlength]{\sol{16}}}}
    \put(-1.6,16.0){\scriptsize\rotatebox{90}{\makebox[11.65\unitlength]{SRC}}}
    \put(100.5,0)  {\includegraphics[height=11.65\unitlength]{figs/summary-legend-karman3d}}
  \end{overpic}
  \caption{Three test cases with (a) Re=68.4, (b) Re=136.7, and (c) Re=546.9.
    Each row shows time steps over the course of 200 time steps for SRC,
    \sol{16}, and the reference (top to bottom). The \sol{16} model interacting
    with the source solver successfully preserves the complex rotating motions
    behind the cylindrical obstacle (middle), which the regular source solver
    cannot resolve (top).}
  \label{fig:appx:karman3d-images-test}
\end{figure}

\begin{table}[htb]
  \caption{Quantitative evaluation of different models for the three-dimensional 
    wake flow scenario.}
  \label{tab:karman3d}
  \begin{center}
  \small
    \begin{tabular}{cccccc}
      \toprule
      \multicolumn{3}{c}{\textbf{MAE Velocity}, Mean (std. dev.)} & \multicolumn{3}{c}{\textbf{Freq. MAE Kinetic Energy}, Mean (std. dev.)}   \\
      \cmidrule(r){1-3} \cmidrule(l){4-6}
      SRC             & NON           & SOL$_{16}$    & SRC            & NON           & SOL$_{16}$    \\
      \cmidrule(r){1-3} \cmidrule(l){4-6}
      0.167 (0.035)   & 0.143 (0.070) & 0.130 (0.024) & 0.0614 (0.133) & 0.074 (0.209) & 0.058 (0.088) \\
      \bottomrule
    \end{tabular}
  \end{center}
\end{table}




% ---

%\subsection{Data Sets}\label{sec:data}
%... add table to list sizes of training, validation and test samples for all 5 scenarios ...

%\clearpage

\section{Performance}\label{app:perf}

%  new measurements from june 23:
%   2d:  5.85s for 100 steps manta CPU , vs old model eval: 1.10s -> fac x 5.3
%       fixed model eval (only NN!): 0.43
%   3d: 913.2s per 100 steps for CPU   vs   13.3s for SOL$_16$ version  -> fac x 68.66

We measure the computational performance of our models 
in comparison to a reference simulation 
on a workstation with an
Intel Xeon E5-1650 CPU with 12 virtual cores at 3.60GHz and
an NVIDIA GeForce GTX 1080 Ti GPU.
As reference solver, we employ a CPU-based simulator 
using OpenMP parallelization.
%that was hand-optimized for OpenMP. 
We compare this with our (relatively un-optimized)
differentiable physics framework, which evaluates the 
PDE and the trained model within \emph{TensorFlow} on the GPU.

For the buoyancy-driven flow simulation, the CPU-based reference simulation 
requires 5.79 seconds on average for 100 time steps. 
Instead, 
%the source solver with the trained model requires 0.37 seconds on average.
\new{evaluating the \sol{128} neural network model itself requires an accumulated 0.43 seconds. 
For comparison, computing 100 time steps of the source solver takes 0.476 seconds.}
%
\new{
In comparison to the inference for forward simulations with a pre-trained model, each iteration during training is significantly more expensive: for the \sol{8}, \sol{16}, and \sol{32} models of the 2D wake flow case, a training iteration took 0.6, 1.3, and 2.5 seconds on average, respectively. As this is a one-time, pre-processing cost, the gains in performance of the resulting hybrid solver can quickly offset the computational expense for training a model.}

The computational workload for PDE solvers typically rises super-linearly 
with the number of degrees of freedom. Hence, the gap is even more 
pronounced when considering the 3D wake flow case.
\new{Here, the reference simulation requires 913.2 seconds for 100 time steps, 
while the \sol{16} version requires 13.3 seconds on average. Thus, the 
source simulation with  learned corrections is more than 68 times %$\times$
faster than 
the reference simulation.}

Despite the substantial reduction in terms of runtime, we believe these performance 
results are preliminary, and far from the speed-up that could be achieved 
in optimal settings with a learning-augmented PDE solver. An inherent advantage of combining 
an approximate PDE solver with a deep-learning-based corrector ANN 
stems from the fact that a relatively simple solver suffices as a basis. Hence, 
while existing reference solvers in scientific computing fields might come 
with vast existing code-bases, the source solver could encompass only a small subset 
of the full solver and introduce the residual dynamics via a learned component.
This would also reduce the work to provide gradients for the source solver, 
which many existing simulation frameworks do not readily offer.
Due to its reduced scope, the source solver would also be significantly easier to optimize. 
Additionally, the learned corrector component would trivially benefit from all 
future hardware advances for efficient evaluations of neural networks.
Hence, we believe that, in practice, a much more substantial speed-up will be achievable
than the ones we have measured for the two- and three-dimensional simulation
scenarios of this work.

 % from main: involving the trained model took 0.37s on average for 100 time steps, whereas its corresponding reference counterpart required 5.79s.

%... karman 3d in mantaflow: avg 9.5725 over 576 steps
%hybrid ML instead: avg 0.1247 over 600 steps
%-> 12.47s for 100 time steps for \sol{16} versus 957.25s for full simulation 

\section{Neural Network Architectures}\label{app:models}

Below, we give additional details of the network architectures
used for the five different scenarios. We intentionally 
slightly vary the architecture to demonstrate that our solver-in-the-loop 
approach does not rely on a single, specific architecture.
We employ ResNets for the large majority of the PDE interaction models
as the correction task resembles a translation from phase space input 
quantities to a field of localized corrections.
The CG solver scenario, on the other hand, requires a more global view,
which motivates our choice of a U-net architecture.
The overall structure with kernel sizes and feature maps 
of both types of networks is illustrated in \myfigref{fig:nn_architecture}.
%We provide details of both architecture types and list kernel sizes as well as the number of feature maps for each layer.
We additionally list hyperparameters for each architecture 
in \mytabref{tab:nn_architecture}.

%The neural network architecture for CG solver scenario of see figure \ref{fig:nn_architecture}).

\begin{figure}[htb]
\centering
$\vcenter{\hbox{\includegraphics[width=0.49\columnwidth]{figs/networkArchitectureKarman2d}}}$
$\vcenter{\hbox{\includegraphics[width=0.49\columnwidth]{figs/networkArchitecture}}}$
\caption{A visual summary of the two main architectures of the neural networks used for
  Sect.~\ref{app:expKarman2d} to \ref{app:expBurgers} (left), % no "in" here
  and Sect.~\ref{app:expCgSolver} (right).
}
\label{fig:nn_architecture}
\end{figure}


%
% Name , Architecture, Layers , Convolutions, Trainable Weights 
\begin{table}[htb]
  \caption{Hyperparameters of neural network architectures.}
  \label{tab:nn_architecture}
  \begin{center}
    \small
    \begin{tabular}{lccccr}
      \toprule
      Experiment                                & Arch.      & Layers & Features     & Conv. Kernels & Train. Weights \\
      \midrule
      2D Wake Flow \ref{app:expKarman2d}        & Res-Net    & 12     & 32           & $5^2$         & 260,354        \\
      2D Wake Flow \ref{app:expKarman2d}, Small & Sequential & 3      & 32, 64       & $5^2$         & 56,898         \\
      Buoyancy \ref{app:expBuoy}, M$_{XS}$      & Res-Net    & 6      & 4            & $5^2,3^2$     & 1,310          \\
      Buoyancy \ref{app:expBuoy}, M$_{S}$       & Res-Net    & 8      & 8            & $5^2,3^2$     & 5,114          \\
      Buoyancy \ref{app:expBuoy}, Regular       & Res-Net    & 10     & 16           & $5^2,3^2$     & 35,954         \\
      Buoyancy \ref{app:expBuoy}, M$_{L}$       & Res-Net    & 14     & 16, 32       & $5^2,3^2$     & 100,114        \\
      Buoyancy \ref{app:expBuoy}, M$_{XL}$      & Res-Net    & 14     & 32, 64       & $5^2,3^2$     & 400,930        \\
      Forced Adv.-diff. \ref{app:expBurgers}    & Res-Net    & 12     & 32           & $5^2$         & 261,154        \\
      CG Solver \myappref{app:expCgSolver}      & U-Net      & 22     & 4, 8, 16, 32 & $5^2$         & 127,265        \\
      3D Wake Flow \ref{app:expKarman3d}        & Res-Net    & 14     & 24,48        & $5^3,3^3$     & 1,002,411      \\
      \bottomrule
    \end{tabular}
  \end{center}
\end{table}

%%% Local Variables:
%%% TeX-master: "neurips_2020_appx"
%%% End:
