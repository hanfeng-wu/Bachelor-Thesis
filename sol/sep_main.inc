
\begin{abstract}
  Finding accurate solutions to partial differential equations (PDEs) is a
  crucial task in all scientific and engineering disciplines. It has recently
  been shown that machine learning methods can improve the solution accuracy by
  correcting for effects not captured by the discretized PDE. We target the
  problem of reducing numerical errors of iterative PDE solvers and compare
  different learning approaches for finding complex correction functions. We
  find that previously used learning approaches are significantly outperformed
  by methods that integrate the solver into the training loop and thereby allow
  the model to interact with the PDE during training. This provides the model
  with realistic input distributions that take previous corrections into
  account, yielding improvements in accuracy with stable rollouts of several
  hundred recurrent evaluation steps and surpassing even tailored supervised
  variants. We highlight the performance of the differentiable physics networks
  for a wide variety of PDEs, from non-linear advection-diffusion systems to
  three-dimensional Navier-Stokes flows.
\end{abstract}

\begin{figure}[b!]
  \includegraphics[width=0.95\columnwidth]{figs/sol-teaser}
  \caption{A 3D fluid problem (shown in terms of vorticity) for which the
    regular simulation introduces numerical errors that deteriorate the resolved
    dynamics (a). Combining the same solver with a learned corrector trained via
    differentiable physics (b) significantly reduces errors w.r.t. the reference
    (c).}
  \label{fig:teaser}
\end{figure}

\myspacesec{}
\section{Introduction}
\myspacesec{}

Numerical methods are prevalent in science to improve the understanding of our
world, with applications ranging from climate modeling
\cite{taylor2012overview,stocker2013climate} over simulating the efficiency of
airplane wings \cite{rhie1983numerical} to analyzing blood flow in a human body
\cite{johnston2004non}. These applications are extremely costly to compute due
to the fine spatial and temporal resolutions required in real-world scenarios.
In this context, deep learning methods are receiving strongly growing attention
\cite{morton2018deep,barsinai2019data,greydanus2019hamiltonian} and show promise
to account for those components of the solutions that are difficult to resolve
or are not well captured by our physical models. Physical models typically come
in the form of PDEs and are discretized in order to be processed by computers.
This step inevitably introduces numerical errors. Despite a vast amount of work
\cite{ghosal1996analysis,arnold2012ode} and experimental evaluations
\cite{brachet1983small,pang1997error}, analytic descriptions of these errors
remain elusive for most real-world applications of simulations.

% A good example is cloud formation, which is a highly complex process that is
% only partially understood, but crucial for climate predictions \cite{}, and
% potentially a good candidate for data-driven models \cite{}.

In our work, we specifically target the numerical errors that arise in the
discretization of PDEs. We show that, despite the lack of closed-form
descriptions, discretization errors can be seen as functions with regular and
repeating structures and, thus, can be learned by neural networks. Once trained,
such a network can be evaluated locally to improve the solution of a PDE-solver,
i.e., to reduce its numerical error.

The core of most numerical methods contains some form of iterative process --
either in the form of repeated updates over time for explicit solvers or even
within a single update step for implicit solvers. Hence, we focus on iterative
PDE solving algorithms \cite{golub2012matrix}. \new{We show that neural networks can
achieve excellent performance if they take the reaction of the solver into
account.} This interaction is not possible with supervised learning on
pre-computed data alone. Even small inference errors of a supervised model can
quickly accumulate over time \cite{tompson2017,kim2019}, leading to a data
distribution that differs from the distribution of the pre-computed data. For
supervised learning methods, this causes deteriorated inference at best and
solver explosions at worst.

% regular repeating structures, mention manifold hypothesis investigate
% interplay, key insight: crucial to train together context of differentiable
% programming

We demonstrate that neural networks can be successfully trained if they can
\emph{interact} with the respective PDE solver during training. To achieve this,
we leverage differentiable simulations
\cite{amosKolter2017,toussaint2018differentiable}. Differentiable simulations
allow a trained model to autonomously explore and experience the physical
environment and receive directed feedback regarding its interactions throughout
the solver iterations. Hence, our work fits into the broader context of machine
learning as differentiable programming, and we specifically target recurrent
interactions of highly non-linear PDEs with deep neural networks. This
combination bears particular promise: it improves generalizing capabilities of
the trained models by letting the PDE-solver handle large-scale changes to the
data distribution such that the learned model can focus on localized structures
not captured by the discretization. While physical models generalize very well,
learned models often specialize in data distributions seen at training time.
However, we will show that, by combining PDE-based solvers with a learned model,
we can arrive at hybrid methods that yield improved accuracy while handling
solution manifolds with significant amounts of varying physical behavior.


% contributions:
We show the advantages of training via differentiable physics for explicit and
implicit solvers applied to a broad class of canonical PDEs. For explicit and
semi-implicit solvers, we consider advection-diffusion systems as well as
different types of Navier-Stokes variants. We showcase models trained with up to
128 steps of a differentiable simulator and apply our model to complex
three-dimensional (3D) flows, as shown in \myfigref{fig:teaser}. Additionally,
we present a detailed empirical study of different approaches for training
neural networks in conjunction with iterative PDE-solvers for recurrent rollouts
of several hundred time steps. On the side of implicit solvers, we consider the
Poisson problem \cite{mathews1970mathematical}, which is an essential component
of many PDE models. Here, our method outperforms existing techniques on
predicting initial guesses for a conjugate gradient (CG) solver by receiving
feedback from the solver at training time.
%
\new{The source code for this project is available at \url{https://github.com/tum-pbs/Solver-in-the-Loop}.}

% ------------------------------------------------------------------------------

% start with neurips topics? ML with models, model-based RL, physical
% understanding, PDE models ...
\myspacepara{} \myspacepara{}
\paragraph{Previous Work}
%
Combining machine learning techniques with PDE models has a long history in
machine learning
\cite{crutchfield1987equations,kevrekidis2003equation,brunton2016discovering}.
% , and is a topic of growing interest within the deep learning community.
%
% Data-driven approaches were shown to be highly effective for inferring unknown
% \acp{pde} and coefficients of corresponding terms
% \cite{brunton2016discovering}.
%
% skip: the first is similar to brunton2016discovering , the second has no DL
% Their capabilities were investigated for different problems, among others the
% Navier-Stokes equations
% \cite{rudye2017,schaeffer2017}. % PNAS brunton& kurz -> fluids, schaeffer - ceofficients only, turbulence? no deep learning
%
More recently, deep-learning-based methods were successfully applied to infer
stencils of advection-diffusion problems \cite{barsinai2019data}, to discover
PDE formulations \cite{long2017pde,raissi2018hiddenphys,sirignano2018dgm}, and
to analyze families of Poisson equations \cite{magill2018}. While identifying
governing equations represents an interesting and challenging task, we instead
focus on a general method to improve the solutions of chosen spaces of
solutions.

Other studies have investigated the similarities of dynamical systems and deep
learning methods \cite{weinan2017proposal} and employed conservation laws to
learn systems described by Hamiltonian mechanics
\cite{greydanus2019hamiltonian,cranmer2020lagrangian}.
% ... connect physical conservation laws and deep learning
% \cite{weinan2017proposal}, or learn conservation laws via Hamiltonian
% mechanics \cite{greydanus2019hamiltonian}, or target symmetries in Lagrangian
% formulations \cite{cranmer2020lagrangian}. % even newer
% all reversible, i.e. non-dissipative, though?
Existing studies have also identified discontinuities in finite-difference
solutions with deep learning \cite{ray2018} and focused on improving the
iterative behavior of linear solvers \cite{hsieh2019learning}. So-called Koopman
operators likewise represent an interesting opportunity for deep learning
algorithms \cite{morton2018deep,li2019learning}. While these methods replace the
PDE-based time integration with a learned version, our models rely on and
interact with a PDE-solver that provides a coarse approximation to the problem.
Hence, our models always alternate between inference via an artificial neural
network (ANN) and a solver step. This distinguishes our work from studies of
recurrent ANN architectures
\cite{connor1994recurrent,sutskever2014sequence,vaswani2017attention} as the
PDE-solver can introduce significant non-linearities in-between evaluations of
the ANN.

% point out, recurrent NNs alone studied in detail, we instead alternate highly
% non-linear PDE solver and NN}
%
% % However, we investigate the problem for challenging two-dimensional flow
% % problems and propose a new approach for learning the correction bby taking
% % into account temporal dynamics.
% ... \cite{hsieh2019learning} focuses on linear models, trained for 1..20
% iterations

% Due to the complexity and nonlinear nature of In the following,
We focus on chaotic systems for which fluid flow represents an exciting
and %\ku{a} % not necessary to repeat
challenging problem domain that is highly relevant for industrial applications.
% fluid dynamics, they remain a which remain highly challenging and relevant for
% many industrial applications. and accordingly data-driven approaches including
% deep learning have
Deep learning methods have received significant amounts of attention in this
area \cite{kutz2017}.
% to reach an efficient solution for such problems \cite{kutz2017}. E.g.,
For example, both steady \cite{guo2016} and unsteady \cite{morton2018deep}, as
well as multi-phase flows \cite{gibou2018} have been investigated with deep
learning based approaches. Turbulence closure modeling has been an area of
particular focus \cite{tracey2015machine,maulik2017,beck2018}. Additionally,
\ac{cnns} were studied for stochastic sub-grid modeling \cite{um2018},
airfoil flow problems \cite{thuerey2020deep,zhang2018},
and as part of generative networks to leverage the fast inference of
pre-trained models \cite{chu2017,xie2018,kim2019}. Other studies have targeted the
unsupervised learning of divergence-free corrections \cite{tompson2017} or
incorporated PDE-based loss functions to represent individual flow solutions via
ANNs \cite{raissi2018hiddenfluid,sirignano2018dgm}. In addition to temporal
predictions of turbulent flows
\cite{mohan2019compressed}, 
similar algorithms were more recently also employed for classification problems
\cite{he2020advectivenet}. 
However, to the best of our knowledge, the existing methods do not let ANNs
interact with solver in a recurrent manner. As we will demonstrate below, this
combination yields significant improvements in terms of inference accuracy.

While we focus on Eulerian, i.e., grid-based discretizations, the Lagrangian
viewpoint is a popular alternative. While a variety of studies has investigated
graph-based simulators, e.g., for rigid-body physics in the context of human
reasoning \cite{battaglia2013simulation,watters2017visual,bapst2019structured}
or weather predictions \cite{seo2020graphphysics}, particles are also a popular
basis for fluid flow problems
\cite{li2018learning,ummenhofer2020lagrangian,sanchez2020gns}. Despite our
Eulerian focus, Lagrangian methods could likewise benefit from incorporating
differentiable solvers into the training process.

Our work shares the motivation of previous work to use differentiable components
at training time
\cite{amosKolter2017,de2018end,toussaint2018differentiable,chen2018neural} and
frameworks for differentiable programming
\cite{schoenholz2019jax,hu2019difftaichi,innes2019differentiable,holl2020}.
Differentiable physics solvers were proposed for inverse problems in the context
of liquids \cite{schenck2018spnets}, cloth \cite{liang2019differentiable}, soft
robots \cite{hu2019difftaichi}, and molecular dynamics
\cite{wang2020differentiable}.
While these studies typically focus on optimization
problems or replace solvers with learned components, we focus on the interaction between
the two. Hence, in contrast to previous work, we always rely on a PDE-solver to
yield a coarse approximate solution and improve its performance via a trained
ANN.

% JAX MD \cite{schoenholz2019jax} ... particle-based / graph fluids DPI net
% \cite{li2018learning} benjamin \cite{ummenhofer2020lagrangian} more recent
% \cite{sanchez2020learning} julia / zygote ? \cite{innes2019differentiable}

% farimani 2017, still not published? follow up in 2018 \cite{sharma2018weakly}
% https://scholar.google.de/citations?hl=en&user=aH52nxkAAAAJ&view_op=list_works&sortby=pubdate
% , https://sites.google.com/view/barati/team

% schroedinger eq learning \cite{mills2017deep}? differentiable solvers: junbang
% minglin, discuss diff cloth physics: \cite{liang2019differentiable} florence
% CVPR \cite{rasheed2020learning} diff MD sims \cite{wang2020differentiable}
% PBDL Differentiable Molecular Simulations for Control and Learning ,
% https://arxiv.org/pdf/2003.00868.pdf more general, diffTaichi
% \cite{hu2019difftaichi}
%
% ICLR physics-aware graph networks \cite{seo2020graphphysics}
% https://github.com/USC-Melady/ICLR2020-PADGN Physics-aware Difference Graph
% Networks for Sparsely-Observed Dynamics graph nets \cite{sanchez2018graph}
% also schenck \cite{schenck2018spnets}

% also needed for e.g. weather forecasting over time \cite{rasp2020weatherbench}
% two step training \cite{weyn2020improving}; Google MetNet without time
% stepping \cite{sonderby2020metnet}

% categorize previous work w.r.t. correction methodology in appendix? many
% others use simple euler integration as PDE, and learn most of the physics on
% top...}

% ... remove?} While the references above provide a rough positioning of our
% work in the context of existing research, a more in-depth discussion of
% previous work is provided in the appendix.

% clarify - we don't replace solver!}

% reduce own cites For fluid simulation, particularly in the field of computer
% graphics, data-driven approaches have been considered as an efficient
% alternative to replace computationally expensive steps of numerical processes
% \cite{ladicky2015,tompson2017}. Moreover, deep learning, particularly with
% generative adversarial models \cite{goodfellow2016a}, has been used for
% efficiently synthesizing details within coarsely resolved simulations
% \cite{chu2017,xie2018}. In addition to scalar transport and smoke flows,
% liquid droplets have been efficiently tackled by learning stochastic models
% \cite{um2018}. Due to their capabilities for transforming space-time data
% into reduced latent spaces, fluid simulations were processed and re-simulated
% with \ac{nn} models \cite{kim2019,prantl2018}. Motivated by the success of
% previous studies regarding data-driven approaches in numerical simulations, we
% aim for enabling the learning of general correction functions. Our aim differs
% from methods that synthesize details for low-resolution input data.
%
% As such, our method is orthogonal to super-resolution methods and shares
% similarities with methods to ...

% also cf https://arxiv.org/pdf/2004.04653.pdf , structure pres: also pendulums,
% but include couette flow from there: Consider a system whose governing
% variables will be hereafter denoted by z in M Rn with M the state space of
% these variables, which is assumed to have the structure of a differentiable
% manifold in Rn

% from greydanus2019hamiltonian, distinguish Learning physical laws from data.
% Schmidt, M. and Lipson, H. Distilling free-form natural laws from experimental
% data. Science, 324(5923):81–85, 2009. ; Iten, R., Metger, T., Wilming, H., Del
% Rio, L., and Renner, R. Discovering physical concepts with neural networks. ;
% Bondesan, R. and Lamacraft, A. Learning symmetries of classical integrable
% systems. arXiv preprint arXiv:1906.04645, 2019. Physics priors for neural
% networks. Modeling energy surfaces. (mode MD)

% ------------------------------------------------------------------------------

\myspacesec{}
\section{Learning to Reduce Numerical Errors}\label{sec:flow-corr}
\myspacesec{}

Numerical methods yield approximations of a smooth function $\vu$ in a discrete
setting and invariably introduce errors. These errors can be measured in terms
of the deviation from the exact analytical solution. For discrete simulations of
PDEs, they are typically expressed as a function of the truncation, $O(\dt^k)$.
Higher-order methods, with large $k$, % for a given step size $\dt$
are preferable but difficult to arrive at in practice. For practical schemes, no
closed-form expression exists for truncation errors, and the errors often grow
exponentially as solutions are integrated over time. We investigate methods that
solve a discretized PDE $\pde$ by performing discrete time steps $\dt$. Each
subsequent step can depend on any number of previous steps,
$\vu(\vx,t+\dt) = \pde(\vu(\vx,t),\vu(\vx,t-\dt),...)$, where
$\vx \in \Omega \subseteq \mathbb{R}^d$ for the domain $\Omega$ in $d$
dimensions, and $t \in \mathbb{R}^{+}$.

\myspacepara{}
\paragraph{Problem Statement:}
%
We consider two different discrete versions of the same PDE $\pde$, with $\pder$
denoting a more accurate discretization with solutions $\vrN \in \manifref$ from
the \emph{reference manifold}, and an approximate version $\pdec$ with solutions
$\vcN \in \manifsrc$ from the \emph{source manifold}. We consider $\vrN$ and
$\vcN$ to be states at a certain instance in time, i.e., they represent phase
space points, and evolutions over time are given by a trajectory in each
solution manifold. As we focus on the discrete setting, a solution over time
consists of a \emph{reference sequence}
\new{$\{\vr{t}, \vr{t+\dt}, \cdots, \vr{t+k \dt}\}$} in the solution manifold
$\manifref$, and correspondingly, a more coarsely approximated \emph{source
  sequence} \new{$\{\vc{t},\vc{t+\dt},\cdots,\vc{t+k \dt}\}$} exists in the
solution manifold $\manifsrc$. We also employ a mapping operator $\project$ that
transforms a phase space point from one solution manifold to a suitable point in
the other manifold, e.g., for the initial conditions of the sequences above, we
typically choose $\vc{t} = \project \vr{t}$. We discuss the choice of $\project$
in more detail in the appendix, but in the simplest case, it can be obtained via
filtering and re-sampling operations.

By evaluating $\pder$ for $\manifref$, we can compute the points of the phase
space sequences, e.g., $\vr{t+\dt} = \pder(\vr{t})$ for an update scheme that
only depends on time $t$. Without loss of generality, we assume a fixed $\dt$
and denote a state $\vr{t+k \dt}$ after $k$ steps of size $\dt$ with $\vr{t+k}$.
Due to the inherently different numerical approximations,
$\pdec(\project \vr{t}) \ne \project \vr{t+1}$ for the vast majority of states.
In chaotic systems, such differences typically grow exponentially over time
until they saturate at the level of mean difference between solutions in the two
manifolds. We use an \new{$L^2$-norm} in the following to quantify the
deviations, i.e.,
\new{$\loss (\vc{t},\project \vr{t})=\Vert\vc{t}-\project \vr{t}\Vert_2$}. Our
learning goal is to arrive at a correction operator $\corr ( \vc{} )$ such that
a solution to which the correction is applied has a lower error than an
unmodified solution:
$\loss ( \pdec( \corr (\project \vr{t_0}) ) , \project \vr{t_1}) < \loss (
\pdec( \project \vr{t_0} ), \project \vr{t_1})$. The correction function
$\corr (\vcN | \theta)$ is represented as a deep neural network with weights
$\theta$ and receives the state $\vcN$ to infer an additive correction field
with the same dimension. To distinguish the original phase states $\vcN$ from
corrected ones, we denote the latter with $\vctN$, and we use an exponential
notation to indicate a recursive application of a function, i.e.,
\begin{equation}
  \vc{t+n} = \pdec(\pdec(\cdots \pdec( \project \vr{t}  )\cdots)) = \pdec^n ( \project \vr{t} ) \ .
\end{equation}

% (done, removed) As measuring differences in chaotic systems is a challenging
% topic by itself
% \cite{}, % Philipp: This motivation for using L2 makes no sense.
% check \cite{christie2005,press2007,kat2012} , \cite{wang2006} ,
% \cite{mehta2016} 88 numrecipe 700 29 , ssim? 16 nasa

% of the form $\corr (\vcN)=\vcN + \nnfunc (\vcN | \theta)$, where $\nnfunc$ is
% represented as a deep neural network with weights $\theta$. (done, kept C & NN
% for now...) Philipp: f is not the ideal symbol for a neural network, also
% write it as a distribution Net(u|theta). Why introduce the letter C if we
% always use the form u + f? Why not rename f to C and change the equation above
% to incorporate the "correction" aspect? Then we could simply call the network
% C.

\begin{wrapfigure}[14]{r}{0.6\linewidth}
  \centering
  \begin{overpic}[width=\linewidth]{figs/Manifolds}
    \put(0, 2){\footnotesize $\manifref$}  % Reference
    \put(36, 2){\footnotesize $\manifsrc$} % Source
  \end{overpic}
  \caption{Transformed solutions of the reference sequence computed on
    $\manifref$ (blue) differ from solutions computed on the source manifold
    $\manifsrc$ (orange). A correction function $\corr$ (green) updates the
    state after each iteration to more closely match the projected reference
    trajectory on $\manifsrc$.}
  \label{fig:methods}
\end{wrapfigure}

Within this setting, any type of learning method naturally needs to compare
states from the source domain with the reference domain in order to bridge the
gap between the two solution manifolds. How the evolution in the source manifold
at training time is computed, i.e., if and how the corrector interacts with the
PDE, has a profound impact on the learning process and the achievable final
accuracy. We distinguish three cases: no interaction, a pre-computed form of
interaction, and a tight coupling via a differentiable solver in the training
loop.

% $\vct{t+1} = \pdec(\vct{t})$. In the following we use an exponential notation
% to indicate a recursive application of a function, i.e.,
% \begin{equation}
%   \vct{t+n} = \pdec(\pdec(\cdots \pdec( \project \vr{t}  )\cdots)) = \pdec^n ( \project \vr{t} ),
% \end{equation}
% $\vct{t+n} = \pdec(\pdec(\cdots \pdec( \project \vr{t} )\cdots))$ , from (a)
% $G$ via $\corrPre(\vu)=\vu+G(\vu)$. $\corrPre(\vu)=\vu+F(\vu,\theta)$. from
% (c) over time is given
% by %$\vct{t+1} = \pdec(\corr(\vct{t}))$ with $\corr(\vct{})=\vu+F(\vct{}, \theta)$, %i.e.

\vspace{-0.5em}
\begin{itemize}[leftmargin=1.5em]
  \itemsep0em

\item \textbf{Non-interacting (NON)}: The learning task purely uses the
  unaltered PDE trajectories, i.e., $\vc{t+n} = \pdec^n ( \project \vr{t} )$
  with $n$ evaluations of $\pdec$. These trajectories are fully contained in the
  source manifold $\manifsrc$. Learning from these states means that a model
  will not see any states that deviate from the original solutions. As a
  consequence, models trained in this way can exhibit undesirably strong error
  accumulations over time. This corresponds to learning from the difference
  between the orange and blue trajectories in \myfigref{fig:methods}, and most
  commonly applied supervised approaches use this variant.

\item \textbf{Pre-computed interaction (PRE)}: To let an algorithm learn from
  states that are closer to those targeted by the correction, i.e., the
  reference states, a pre-computed or analytic correction is applied. Hence, the
  training process can make use of phase space states that deviate from those in
  $\manifsrc$, as shown in green in \myfigref{fig:methods}, to improve inference
  accuracy and stability. This approach can be formulated as
  $\vc{t+n} = (\pdec \corrPre)^n ( \project \vr{t} )$ with a pre-computed
  correction function $\corrPre$. In this setting, the states $\vcN$ are
  corrected without employing a neural network, but they should ideally resemble
  the states achievable via the learned correction later on. As the modified
  states $\vcN$ are not influenced by the learning process, the training data
  can be pre-computed. A correction model $\corr (\vcN | \theta)$ is trained via
  $\vctN$ that replaces $\corrPre$ at inference time.

\item \textbf{Solver-in-the-loop (SOL)}: By integrating the learned function
  into a differentiable physics pipeline, the corrections can interact with the
  physical system, alter the states, and receive gradients about the future
  performance of these modifications. The learned function $\corr$ now depends
  on states that are modified and evolved through $\pde$ for one or more
  iterations. A trajectory for $n$ evaluations of $\pdec$ is given by
  $\vct{t+n} = ( \pdec \corr )^n ( \project \vr{t} )$, with
  $\corr (\vctN | \theta)$. The key difference with this approach is that
  $\corr$ is trained via $\vctN$, i.e., states that were affected by previous
  evaluations of $\corr$, and it affects $\vctN$ in the following iterations. As
  for (PRE), this learning setup results in a trajectory like the green one
  shown in \myfigref{fig:methods}, however, in contrast to before, the learned
  correction itself influences the evolution of the trajectory, preventing a gap
  for the data distribution of the inputs.

\end{itemize}
\vspace{-0.5em}

In addition to these three types of interaction, a second central parameter is
the look-ahead trajectory per iteration and mini-batch of the optimizer during
learning. A subscript $n$ denotes the number of steps over which the future
evolution is recursively evaluated, e.g., \sol{n}. The objective function, and
hence the quality of the correction, is evaluated with the training goal to
minimize $\sum_{i=t}^{t+n} \loss (\vc{i},\vr{i})$. Below, we will analyze a
variety of learning methodologies that are categorized via learning methodology
(NON, PRE or SOL) and look-ahead horizon $n$.

% Below, we will denote these three types of interaction with the physics
% environment as NON, PRE, and SOL, for a, b and c above, respectively. For NON
% and PRE, all training data are pre-computed, while for SOL, due to the
% interdependencies of $\vctN$ and $\nnfunc$, we require a differentiable
% PDE-solver for $\pdec$, and the model is dynamically trained via the changing
% states that develop over the course of a training run.

% to categorizing methods w.r.t. to the three categories
% above, % as a first dimension,
% a function $\loss(\vcN,\vrN)$, for which we choose $|\vcN - \project \vrN|_2$
% in the following. For $n>1$ the for large $N$.

\myspacesec{}
\section{Experiments}
\myspacesec{}

We now provide a summary and discussion of our experiments with the different
types of PDE interactions for a selection of physical models. Full details of
boundary conditions, parameters, and discretizations of all five PDE scenarios
are given in App. B. % \myappref{app:exp}.

\myspacessec{}
\subsection{Model Equations and Data Generation}
\myspacessec{}

We investigate a diverse set of constrained advection-diffusion models of which
the general form is
\begin{equation}
  \label{eq:model-adv-diff}
  % \frac{\partial\vuN}{\partial{t}}
  \partial{\vu} / \partial{t}
  = - \vu \cdot\nabla\vu +
  % - \frac{1}{\rho}\nabla{p} +
  \nu \nabla\cdot \nabla \vu + \mathbf{g}
  \quad \text{subject to} \quad \mM \vu = 0,
\end{equation}
where $\vu$ is the velocity, $\nu$ denotes the diffusion coefficient (i.e.,
viscosity), and $\mathbf{g}$ denotes external forces. The constraint matrix
$\mM$ contains an additional set of equality constraints imposed on $\vu$. In
total, we target four scenarios: pure non-linear advection-diffusion (Burger's
equation), two-dimensional Navier-Stokes flow, Navier-Stokes coupled with a
second advection-diffusion equation for a buoyancy-driven flow, and a 3D
Navier-Stokes case. Also, we discuss CG solvers in the context of differentiable
operators below.

For each of the five scenarios, we implement the non-interacting evaluation
(NON) by pre-computing a large-scale data set that captures a representative and
non-trivial space of solutions in $\manifsrc$. The reference solutions from
$\manifref$ are typically computed with the same numerical method using a finer
discretization (4x in our setting, with effective resolutions of $128^2$ and
higher). The PDEs are parametrized such that the change of discretization leads
to substantial differences when integrated over time. For several of the 2D
scenarios, we additionally train models with data sets of trajectories that have
been corrected with other pre-computated correction functions. For these PRE
variants, we use a time-regularized, constrained least-squares corrector
\cite{henderson1975best} to obtain corrected phase state points. For the SOL
variants, we employ a differentiable PDE-solver that runs mini-batches of
simulations and provides gradients for all operations of the solving process
within the deep learning framework. This allows gradients to freely propagate
through the PDE-solver and coupled neural networks via automatic
differentiation. For $n>1$, i.e., PDE-based look-ahead at training time, the
gradients are back-propagated through the solver $n-1$ times, and the difference
w.r.t. a pre-computed reference solution is evaluated for all intermediate
results.

% Details of boundary conditions, parameters, and discretizations of all five
% PDE scenarios are given in \myappref{app:pdes}.

% NNs
\myspacessec{}
\subsection{Training Procedure}
\myspacessec{}

The neural network component $\nnfunc (\vc{} \, | \, \theta)$ of the correction
function % $\corr $
is realized with a fully convolutional architecture. As our focus lies on the
methodology for incorporating PDE models into the training, the architectures
are intentionally kept simple. However, they were chosen to yield high accuracy
across all variants.
% and we provide an ablation study in \myappref{app:arch}.
Our networks typically consist of 10 convolutional layers with 16 features each,
interspresed with ReLU activation functions using kernel sizes of $3^d$ and
$5^d$. The networks parameters $\theta$ are optimized with a fixed number of
steps with an ADAM optimizer \cite{kingma2014} and a learning rate of $10^{-4}$.
For validation, we use data sets generated from the same parameter distribution
as the training sets. All results presented in the following use test data sets
whose parameter distributions differ from the ones of the training data set.

We quantify the performance of the trained models by computing the mean absolute
error between a computed solution and the corresponding projected reference
% , i.e., $\sum_{i=t}^{t+n} |\vc{i} - \project \vr{i}|/n$
for $n$ consecutive steps of a simulation. We report absolute error values for
different models in comparison to an unmodified source trajectory from
$\manifsrc$. Additionally, relative improvements are given w.r.t. the difference
between unmodified source and reference solutions.
% (err[src]-err[model])/err[src]
An improvement by 100\% would mean that the projected reference is reproduced
perfectly, while negative values indicate that the modified solution deviates
more from the reference than the original source trajectory.
% nils: could be shortened?

% To realize the correction function $\corr (\vc{})=\vc{} + f(\theta,\vc{})$, we
% approximate $f$ with a neural network that receives a single state of the
% physical where $f$ is represented as a deep neural network with weights
% $\theta$ that receives $\vu$ as input.

\newcommand{\heightSumFig}{0.29\linewidth}
\newcommand{\heightSumLeg}{0.096\linewidth}
\begin{figure}[tb]
  \centering \hfill
  \begin{overpic}[height=\heightSumFig]{figs/summary-karman}
    \put(80,92){\footnotesize (a)}
    \put(-5,66){\scriptsize\rotatebox{90}{\makebox[0.097\linewidth]{Source}}}
    \put(-5,33){\scriptsize\rotatebox{90}{\makebox[0.097\linewidth]{Corrected}}}
    \put(-5,0) {\scriptsize\rotatebox{90}{\makebox[0.097\linewidth]{Reference}}}
  \end{overpic}
  \begin{overpic}[height=\heightSumLeg]{figs/summary-legend-karman}\end{overpic} % legend karman
  \hspace{1pt} % ---
  \begin{overpic}[height=\heightSumFig]{figs/summary-smoke}
    \put(92,75){\footnotesize {\color{white}(b)}}
  \end{overpic}
  \begin{overpic}[height=\heightSumLeg]{figs/summary-legend-buoy}\end{overpic} % legend buoyancy
  \hspace{1pt} % ---
  \begin{overpic}[height=\heightSumFig]{figs/summary-burgers}
    % \put(90,70){\footnotesize {\color{black}(c)}} % with 4 imgs
    \put(90,93){\footnotesize {\color{black}(c)}} % for 3 images
  \end{overpic}
  \begin{overpic}[height=\heightSumLeg]{figs/summary-legend-burgers}\end{overpic} % legend burgers
  \caption{Our PDE scenarios cover a wide range of behavior including (a) vortex
    shedding, (b) complex buoyancy effects, and (c) advection-diffusion systems.
    Shown are different time steps (l.t.r.) in terms of vorticity for (a),
    transported density for (b), and angle of velocity direction for (c).}
  \label{fig:visualResultSummary}
\end{figure}


\myspacesec{}
\section{Results}
\myspacesec{}

% Tentative naming scheme for the models: NON$_1$ - regular model 1-step model;
% PRE - pre-computed supervised model without temporal coherence; PRE-TC - with
% temporal coherence; SIL$_{32}$ - Solver-in-the-Loop model (e.g., with 32
% steps)}

% [note: the following is only a rough outline for the argumentation]} todo,
% discuss in terms of phase space trajectories}
%
Our experiments show that learned correction functions can achieve substantial
gains in accuracy over a regular simulation. When training the correction
functions with differentiable physics, this additionally yields further
improvements of more than 70\% over supervised and pre-computed approaches from
previous work. A visual overview of the different tests is given in
\myfigref{fig:visualResultSummary}, and a summary of the full evaluation from
the appendix is provided in \myfigref{fig:mainResultSummary} and
\mytabref{tab:main}. In the appendix, we also provide error measurements w.r.t.
physical quantities such as kinetic energy and frequency content.
% For the cases above, additional quantitative evaluations with respect to
% physical quantities such as kinetic energy and frequency content are given in
% the appendix.
The source code of our experiments and analysis will be published upon
acceptance. %with this paper.
% The source code for PDE-solvers, training and data generation of all
% experiments will be published.

\myspacepara{}
\paragraph{Unsteady Wake Flow}
% karman2d
%
The PDE scenario for unsteady wake flows represents a standard benchmark case
for fluids \cite{rajani2009numerical,morton2018deep} and involves a continuous
inflow with a fixed, circular obstacle, which induces downstream vortex shedding
with distinct frequencies depending on the Reynolds number. For coarse
discretizations, the approximation errors distort the flow leading to
deteriorated motions or suppressed vortex shedding altogether. An example flow
configuration is shown in \myfigref{fig:visualResultSummary}a. In this scenario,
the simplest method (NON) yields stable training and a model that already
reduces the mean absolute error (MAE) from 0.146 for a regular simulation
without correction (SRC) to an MAE of 0.049 when applying the learned
correction. The pre-computed correction (PRE) improves on this behavior via its
time regularization with an error of 0.031. A \sol{32} model trained with a
differentiable physics solver for 32 time steps in each iteration of ADAM yields
a significantly lower error of 0.013. This means, the numerical errors of the
source simulation w.r.t. the reference were reduced by more than a factor of 10.
%
% next, DP versions significantly outperform both baselines, on avg over all
% examples by X\%
Despite the same architecture and weight count for all three models, the overall
performance varies strongly, with the \sol{32} version outperforming the simpler
variants by 73\% and more.
%
An example of the further evaluations provided in the appendix is given in
\myfigref{fig:sum-karman-freq}.

% Kiwon: 73% = (err[NON] - err[SOL])/err[NON]
% clearest case unsteady wake flow, characteristic vortex shedding frequency ,
% driven by shearing at obstacle wall

\myspacepara{}
\paragraph{Buoyancy-driven Flow}
% smoke / buoyancy
%
We evaluate buoyancy-driven flows as a scenario with increased complexity. In
addition to an incompressible fluid, a second, non-uniform marker quantity is
advected with the flow that exerts a buoyancy force. This coupled system of
equations leads to interesting and complex swirling behavior over time. We
additionally use this setup to highlight that the reference solutions can be
obtained with different discretization schemes. We use a higher-order advection
scheme in addition to a 4$\times$ finer spatial discretization to compute the
reference data.
% buoyancy - additional equation for transported density, more complex physics

% even very long rollout periods at training time yield substantial improvements
% for the correction functions in this scenario.
Interestingly, the correction functions benefit from particularly long rollouts
at training time in this scenario. Models with simple pre-computed or unaltered
trajectories yield mean errors of 1.37 and 1.07 compared to an error of 1.59 for
the source simulation, respectively. Instead, a model trained with
differentiable physics with 128 steps (\sol{128}) successfully reduces the error
to 0.62, an improvement of more than 59\% compared to the unmodified simulation.

\myspacepara{}
\paragraph{Forced Advection-Diffusion}
% burgers
%
A third scenario employs Burger's equation as a physical model. We mimic the
setup from previous work \cite{barsinai2019data} to inject energy into the
system via a forcing term with a spectrum of %randomized
sine waves. This forcing prevents the system from dissipating to relatively
static and slowly moving configurations. While the PRE and NON versions yield
clear improvements, the SOL versions do not significantly outperform the simpler
baselines. This illustrates a limitation of long rollouts via differentiable
physics: Learned correction functions need to be able to anticipate future
behavior to make high-quality corrections. The randomized forcing in this
example severely limits the number of future steps that can accurately be
predicted given one state. This behavior contrasts with other physical systems
without external disturbances, where a single state uniquely determines its
evolution.
%
\new{We show in the appendix that the SOL models with an increased number of
  interaction steps pay off when the external disturbances are absent.}

\myspacepara{}
\paragraph{Conjugate Gradient Solver}
% cgsolver
%
We turn to iterative solvers for linear systems of equations to illustrate
another aspect of learning from differentiable physics: its importance for the
propagation of boundary condition effects. As our learning objective, we target
the inference of initial guesses for CG solvers \cite{hestenes1952cg}. Following
previous work \cite{tompson2017}, we target Poisson problems of the form
$\nabla \cdot \nabla p = \nabla \cdot \vu$, which arise for projections of a
velocity $\vu$ to a divergence-free state. Instead of fully relying on an ANN to
produce the pressure field $p$, we instead target the learning objective to
produce an initial guess, which is improved by a regular CG solver until a given
accuracy threshold is reached.

This goal can be reached by directly minimizing the right-hand side term
$\nabla \cdot \vu$, similar to physics-based loss terms proposed in a variety of
studies \cite{raissi2018hiddenfluid,sirignano2018dgm}. Alternatively, we can
employ a differentiable CG solver and formulate the learning goal as minimizing
the same residual after $n$ steps of the CG solver (similar to the \sol{n}
models above). While the physics-based loss version reduces the initial
divergence more successfully, it fares badly when interacting with the CG
solver: compared to the SOL version, it requires 63\%
% (iter[SOL-DIV]-iter[SOL-5])/iter[SOL-DIV] = 63%
more steps to reach a desired accuracy. Inspecting the inferred solutions
reveals that the former model leads to comparatively large errors near
boundaries, which are small for each grid cell but significantly influence the
solution on a large scale. The SOL version immediately receives feedback about
this behavior via the differentiable solver iterations. I.e., the differentiable
solver provides a look-ahead of how different parts of the solution affect
future states. In this way, it can anticipate problems such as those in the
vicinity of boundary conditions.

% learning from physical constraints: interesting variant: initial guess for CG
% solver, physical constraints by themselves no guarantee for good solution.
% learning unsupervised by minimizing constraint minimizes loss extremely well,
% but does not necessarily learn to interact with PDE environment. training with
% CG iterations yields substantial benefits, iterations reduced by X\%. while
% unsupervised version yields Y\% better error reduction initially, it does not
% yield benefits when used with solver (0\% improvement). among others, BC
% effects: error small on average, but certain regions of solution are crucial
% for subsequent steps: BC errors strongly propoagate throughout solution. DP
% training directly yields feedback regarding such critical behavior at training
% time.

\myspacepara{}
\paragraph{Three-dimensional Fluid Flow}
% karman3d
%
Lastly, we investigate a 3D case of incompressible flow. The overall setup is
similar to the unsteady wake flow in two dimensions outlined above, but the
third dimension extends the axes of rotation in the fluid from one to three,
yielding a very significant increase in complexity. As a result, the flow behind
the cylindrical obstacle quickly becomes chaotic and forms partially turbulent
eddies, as shown in \myfigref{fig:teaser}. This scenario requires significantly
larger models to learn a correction function, and the NON version does not
manage to stabilize the flow consistently. Instead, the \sol{16} version
achieves stable rollouts for several hundred time steps and successfully
corrects the numerical inaccuracies of the coarse discretization, improving the
numerical accuracy of the source (SRC) simulation by more than 22\% across a
wide range of configurations.

% k3d: base 0.055782 , sol-16 0.043373 -> (base-sol)/base = 0.222455

% increased complexity due to space of rotations; complex turbulent motions 1
% step model is unstable, larger steps (eg 32) yield stable prediction complex
% example: 3D wake flow, complex 3D vortex shedding and turbulent motions, model
% with 1m parameters, trained with 32 recurrent simulation steps.

% \subsection{PDE Scenarios}

\begin{figure}[t!]
  \centering \subcaptionbox{\scriptsize Unsteady wake\label{fig:sum-karman}}
  {\includegraphics[height=0.23\columnwidth,page=4]{figs/summary-karman-plot}}
  \hfill \subcaptionbox{\scriptsize Buoyancy-driven\label{fig:sum-buoy}}
  {\includegraphics[height=0.23\columnwidth,page=3]{figs/out-vgf-jmsmaller-steps-smoke-uns-jmsmaller-msteps}}
  \hfill \subcaptionbox{\scriptsize Advection-diffusion\label{fig:sum-burgers}}
  {\includegraphics[height=0.23\columnwidth,page=3]{figs/summary-burgers-plot}}
  \hfill \subcaptionbox{\scriptsize CG solver\label{fig:sum-cg}}
  {\includegraphics[height=0.23\columnwidth,page=1]{figs/out-cg-solver}} \hfill
  \subcaptionbox{\scriptsize 3D wake\label{fig:sum-karman3d}}
  {\includegraphics[height=0.23\columnwidth,page=1]{figs/out-vgf-karman3d-velocity}}
  \\
  \subcaptionbox{\scriptsize Unsteady wake,
    look-ahead\label{fig:sum-karman-rel}}
  {\includegraphics[height=0.20\columnwidth,page=6]{figs/summary-karman-plot}}
  \hfill \subcaptionbox{\scriptsize Buoyancy-driven,
    look-ahead\label{fig:sum-buoy-rel}}
  {\includegraphics[height=0.20\columnwidth,page=6]{figs/out-vgf-jmsmaller-steps-smoke-uns-jmsmaller-msteps}}
  \hfill \subcaptionbox{\scriptsize Unsteady wake, frequency
    error\label{fig:sum-karman-freq}}{\includegraphics[height=0.20\columnwidth,page=1]{figs/summary-karman-plot}}
  \caption{(a)-(e) Numerical approximation error w.r.t. reference solution for
    unaltered simulations (SRC) and with learned corrections. The models trained
    with differentiable physics and look-ahead achieve significant gains over
    the other models. (f,g) Relative improvement over varying look-ahead
    horizons. (h) A frequency-based evaluation for the unsteady wake flow
    scenario.}
  \label{fig:mainResultSummary}
\end{figure}

\myspacesec{}
\section{Ablations and Discussion}
\myspacesec{}

We performed an analysis of the proposed training via differentiable physics to
highlight which hyperparameters most strongly influence results. Specifically,
we evaluate varying look-ahead horizons, different model architectures, training
via perturbations, and pre-computed variants.

\myspacepara{}
\paragraph{Future Look-Ahead}
% steps already discussed above?} choices influence the results most: we vary
% ... (1 step L2) version yields an error of X, a correction function trained
% with 8 steps yields an error of Y ... buoyancy ...
For systems with deterministic behavior,
% and within the scope of our tests especially the complex dynamics of the
% buoyancy-driven flow,
long rollouts via differentiable physics at training time yield significant
improvements, as shown in \myfigref{fig:sum-karman-rel} and
\ref{fig:sum-buoy-rel}. While training with a few (1 to 4) steps yields
improvements of up to 40\% for the buoyancy-driven flow scenario, this number
can be raised significantly by increasing the look-ahead at training time. A
performance of more than 54\% can be achieved by 64 recurrent solver iterations,
while raising the look-ahead to 128 yields average improvements of 60\%.
%
% while training with 64 steps reduces this by 30\%. Extending the time
% iterations at training time to 128 further reduces this by 10\%.
Our tests consistently show that, without changing the number of weights or the
architecture of a network, the gradients provided by the longer rollout times
allow the network to anticipate the behavior of the physical system better and
react to it. Throughout our tests, similar performances could not be obtained by
other means.
% such as increased traininig iterations or regularization terms. This improved
% behavior can not be learned by longer training or other methods for
% regularization.

\myspacepara{}
\paragraph{Generalization}
The buoyancy scenario also highlights the very good generalizing capabilities of
the resulting models. All test simulations were generated with an
out-of-distribution parametrization of the initial conditions, leading to
substantially different structures, and velocity ranges over time.

\myspacepara{}
\paragraph{Training with Noise}
% noise
An interesting variant to stabilize physical predictions in the context of Graph
Network-based Simulators %(GNS)
was proposed by Sanchez et al. \cite{sanchez2020gns}. They report that
perturbations of input features with %random-walk
noise lead to more stable long-term rollouts. We mimic this setup in our
Eulerian setting by perturbing the inputs to the neural networks with
$\mathcal{N}(0,\sigma)$ for varying strengths $\sigma$. While a sweet spot with
improvements of 34.5\% seems to exist around $\sigma=10^{-4}$, the increase in
performance is small compared to a model with less perturbations (30.6\%), as
training with an increased look-ahead for the SOL models gives improvements up
to 60.0\%.
% In contrast to the GNS, we found that our Eulerian representations do not
% significantly benefit from training with perturbed inputs, as shown for a
% \sol{2} buoyancy model in \myfigref{fig:mainResultSummary}(x)}. While overly
% strong noise deteriorates the performance, smaller amounts of noise do not
% significantly surpass the baseline without noise. We believe this is caused by
% the improved regularization of the fully convolutional architectures we
% employ.

% in the form of perturbed input features. instead of random walk, we only add
% noise!} perturb input features, to stabilize inference, no substantial gains
% achievable, models typically well regularized with sufficient data and fully
% conv arch.

\myspacepara{}
\paragraph{Training Stability}
% stepping
The physical models we employ introduce a large amount of complexity into the
training loop.
% For example, the self-advection term ($\vu \cdot \nabla \vu$) in all of the
% used PDE models} is highly non-linear, and
Especially during the early stages of training, an inferred correction can
overly distort the physical state. Performing time integration via the
PDE %in the form of iterated PDE solving steps
then typically leads to exponential increases of existing oscillations and a
diverging calculation. Hence, we found it important to pre-train networks with
small look-aheads (we usually use \sol{2} models),
% to bring the initially random correction down to a reasonable level
and then continue training with longer recurrent iterations for the look-ahead.
While this scheme can be applied hierarchically, we saw no specific gains from,
e.g., starting a \sol{32} training with a \sol{2} model versus a \sol{16} model.

% \subsection{Performance}
\myspacepara{}
\paragraph{Runtime Performance}
% 
The training via differentiable physics incurs an increased computational cost
at training time, as the PDE model has to be evaluated for $n$ steps for each
learning iteration, and the calculation of the gradients is typically of similar
complexity as the evaluation of the PDE itself.
% (done) Philipp: But you get n times more gradients per iteration and have the
% model look at n times more data
However, this incurs only moderate costs in our tests. For example, for the
buoyancy-driven flow, the training time increases from 0.21 seconds per
iteration on average for \sol{2} to 0.42s for \sol{4}, and 1.25s for \sol{16}.
The look-ahead additionally provides $n$ times more gradients at training time,
and the inference time of the resulting models is not affected. Hence, the
training cost can quickly pay off in practical scenarios by yielding more
accurate results without any increase in cost at inference time.
% cost, 12*49 steps per epoch m01 model: 22m per 10 epochs , 19m -> 0.21s m02:
% 22m ! , 23m -> 0.23s m04: 43m , 42m -> 0.43s m08: 77m , 74m -> 0.77s m16:
% 132m, 111m , 124m -> 1.25s

% DP incurs computational cost at training time, but only one time, once trained
% the model can be re-used and yields high accuracy}

Computing solutions with the resulting hybrid method which alternates PDE
evaluations and ANN inference also provides benefits in terms of evaluation
performance:
%
% A key benefit of our approach is the gain in performance resulting from our
% trained models. A correction velocity field for a given input is inferred only
% on the basic simulation grid, i.e., the low-resolution grid. This ...
% inference happens for each solving step of the underlying numerical solver and
% only requires a fixed
A pre-trained, fully convolutional CNN has an $\mathcal O(n)$ cost for $n$
degrees of freedom, in contrast to many PDE-solvers with a super-linear
complexity.
%
\new{For example, a simulation as shown in \myfigref{fig:teaser} involving the
  trained model took 13.3s on average for 100 time steps, whereas a CPU-based
  reference simulation required 913.2s. A speed-up of more than 68$\times$.}
% a buoyancy-driven flow simulation involving the trained model took 0.37s on
% average for 100 time steps, whereas its corresponding reference counterpart
% required 5.79s. old numers 2.0s vs 10.5s todo, re-run with smaller model?} See
% \mytabref{tab:performance} in \myappref{app:performance} for more details.

% removed $_\mathrm{(best)}$ \vspace{1em}
% \begin{table}[t!]
%   \caption{A summary of the quantitative evaluation for the five PDE
%   scenarios} } \scriptsize
%   \label{tab:main}
%   \begin{center}
%     \begin{tabular}{ccccccccc} % 2 name+res, 4 vel err, 3 relative
%       \toprule
%       \textbf{Exp.} & \textbf{Res.} & \multicolumn{4}{c}{\textbf{Mean absolute error of velocity}} & \multicolumn{3}{c}{\textbf{Rel. improvement}} \\

%       \cmidrule(lr){3-6}\cmidrule(l){7-9}
%                  &                 & SRC              & PRE                & NON               & SOL               & PRE    & NON  & SOL   \\
%       \midrule
%       Adv.-diff. & 32$\times$32    & 0.248 $\pm$ 0.019 & 0.218 $\pm$ 0.017 & 0.159 $\pm$ 0.015 & 0.148 $\pm$ 0.016 & 12\%   & 36\% & 40\%  \\
%       Wake Flow  & 32$\times$64    & 0.146 $\pm$ 0.004 & 0.031 $\pm$ 0.010 & 0.049 $\pm$ 0.012 & 0.013 $\pm$ 0.003 & 79\%   & 67\% & 91\%  \\
%       Buoyancy   & xx$\times$yy    & 1.590 $\pm$ 1.033 & 1.373 $\pm$ 0.985 & 1.080 $\pm$ 0.658 & 0.620 $\pm$ 0.390 & 19\%   & 29\% & 60\%  \\
%       CG Solver  & xx$\times$yy    & x.xxx $\pm$ y.y   & x.xxx $\pm$ y.y   & -                 & x.xxx $\pm$ y.y   & x \%   & x \% & x.x\% \\
%       3D Wake    & 32$^2 \times$64 & 0.056 $\pm$ 0.020 & -                 & 0.048 $\pm$ 0.025 & 0.043 $\pm$ 0.019 & -      & 14\% & 22\%  \\
%       \bottomrule
%     \end{tabular}
%   \end{center}
% \end{table}

\begin{table}[t!]
  \caption{A summary of the quantitative evaluation for the five PDE scenarios.
    \sol{s} denotes a variant with shorter look-ahead compared to SOL. ($^*$ For
    the CG solver scenario, iterations to reach an accuracy of 0.001 are given.
    Here, SOL$_s$ denotes the physics-based loss version.)}
  \scriptsize
  \label{tab:main}
  \begin{center}
    \addtolength{\tabcolsep}{-1pt}
    \begin{tabular}{cccccccccc}
      \toprule
      \textbf{Exp.} & \multicolumn{5}{c}{\textbf{Mean absolute error of velocity}} & \multicolumn{4}{c}{\textbf{Rel. improvement}} \\

      \cmidrule(lr){2-6}\cmidrule(l){7-10}
                    & SRC             & PRE             & NON             & \sol{s}         & SOL             & PRE  & NON  & \sol{s} & SOL     \\
      \midrule
      Wake Flow    & 0.146$\pm$0.004 & 0.031$\pm$0.010 & 0.049$\pm$0.012 & 0.041$\pm$0.009 & 0.013$\pm$0.003 & 79\% & 67\% & 72\%    & 91\%    \\
      Buoyancy     & 1.590$\pm$1.033 & 1.373$\pm$0.985 & 1.080$\pm$0.658 & 0.944$\pm$0.614 & 0.620$\pm$0.390 & 19\% & 29\% & 41\%    & 60\%    \\
      Adv.-diff.   & 0.248$\pm$0.019 & 0.218$\pm$0.017 & 0.159$\pm$0.015 & 0.152$\pm$0.015 & 0.158$\pm$0.017 & 12\% & 36\% & 39\%    & 36\%    \\
      $^*$CG Solver& 121.6$\pm$13.44 & -               & -               & 79.03$\pm$10.02 & 29.59$\pm$14.83 & -    & -    & 35\%    & 76\%    \\
      3D Wake      & 0.167$\pm$0.061 & -               & 0.144$\pm$0.074 & -               & 0.130$\pm$0.058 & -    & 14\% & -       & 22\%    \\
      \bottomrule
    \end{tabular}
    \addtolength{\tabcolsep}{1pt}
  \end{center}
\end{table}

% ------------------------------------------------------------------------------

\myspacesec{}
\section{Conclusions}
\myspacesec{}

% errors have recognizable structures that can be learned first detailed
% analysis of learning with iterative solvers (not replacing them!)

We have demonstrated how to achieve significant reductions of numerical errors
in PDE-solvers by training ANNs with long look-ahead rollouts and differentiable
physics solvers. The resulting models yield substantially lower errors than
models trained with pre-computed data. We have additionally provided a first
thorough evaluation of different methodologies for letting PDE-solvers interact
with recurrent ANN evaluations.

Identical networks yield significantly better results purely by having the
solver in the learning loop. This indicates that the numerical errors have
regular structures that can be learned and corrected via learned
representations. The resulting networks likewise improve generalization for
out-of-distribution samples and provide stable, long-term recurrent predictions.
Our results have the potential to enhance learning physical priors for a variety of deep learning tasks. 
%and to 
%
\new{
Beyond engineering applications and medical simulations, a particularly interesting application of our approach is weather prediction \cite{rasp2020weatherbench}, where a simple differentiable solver could be augmented with a learned correction function to recover the costly predictions of operational forecasting systems.} % from rebuttal

\new{
Overall, we hope that the demonstrated gains in accuracy will help to establish trained neural networks as components in the
numerical toolbox of computational science.
}% from nils

% more outlook? ... use measured data for reference manifold ... Despite
% focusing on fluids, we envision that our approach can be applied to a variety
% of other application domains that involve numerical methods for
% spatio-temporal problems, from plasma physics \cite{lewis1984computational} to
% climate modeling \cite{randall2007climate}.


% ------------------------------------------------------------------------------

%\section{Impact Statement}
\section*{Broader Impact}

PDE-based models are very commonly used and can be applied to a wide range of
applications, including weather and climate, epidemics, civil engineering,
manufacturing processes, and medical applications. Our work has the potential to
improve how these PDEs are solved. As PDE-solvers have a long history, there is
a wide range of established tools, some of which still use COBOL and FORTRAN.
Hence, it will not be easy to integrate deep learning methods into the existing
solving pipelines, but in the long run, our method could yield solvers that
compute more accurate solutions with a given amount of computational resources.

Due to the wide range of applications of PDEs, our methods could also be used in
the development of military equipment (machines and weapons) or other harmful
systems.
% Due to the wide range of applications, potential risks include that PDE-based
% systems could be misused or abused, e.g., to manufacture illegal weapons or
% other systems that harm humans. As our work aims for general PDE-solvers, it
% could also accelerate the computations in such undesirable application
% scenarios.
However, our method shares this danger with all numerical methods. For the
discipline of computational science as a whole, we see more positive aspects
when computer simulations become more powerful. Nonetheless, we will encourage
users of our method likewise to consider ethical implications when employing
PDE-solvers with learning via differentiable physics.

% ... for humanity and our environment. Example below is from this medium
% website
% \url{https://medium.com/@operations_18894/a-guide-to-writing-the-neurips-impact-statement-4293b723f832}

% GNNs (and hence GNNExplainer) could be applied to a wide range of
% applications, including computer vision, natural language processing,
% recommender systems, traffic prediction, generative models and many more (Zhou
% et al 2019, Wu et al 2019). Our research could be used to provide explanations
% for GNNs in these applications, improving understanding of individual
% decisions, as well as the underlying models.

% While there will be important impacts resulting from the use of GNNs in
% general, here we focus on the impact of using our tool to provide explanations
% for such systems. There are many benefits to using such a tool, such as
% increasing the transparency in decision-critical applications. This can help
% mitigate fairness, privacy and safety risks — see the introduction of the
% paper for more details. The potential risks of increasing explainability have
% typically received less attention. These include: (i) the risk of automation
% bias, i.e. an undue trust in models (see e.g. PAI 2019), (ii) if the use of
% explanations means systems may now be used by those with lower levels of
% domain or ML expertise, this could increase the risk of the model or its
% outputs being used incorrectly, (iii) if explanations are only used during the
% development phase, and the model is retrained over time, changing its
% behaviour, this could again lead to a false sense of security.

% We see opportunities for research applying GNNExplainer to beneficial
% purposes, such as investigating whether GNNExplainer could improve algorithmic
% fairness. To mitigate the risks associated with using generated explanations,
% we encourage research to understand the impacts of using GNNExplainer in
% particular real-world scenarios. In these scenarios, do users understand the
% explanations given, and act accordingly, not falling prey to automation bias?
% Does use of the system improve or diminish domain expertise over time?

\begin{ack}
This work is supported by the ERC Starting Grant {\em realFlow} (StG-2015-637014).
\end{ack}

% ERC grant RealFlow and dataFlow Bjoern List - careful proofreading Aristotelis
% Economides - rendering Stephan Rasp - burgers

% Use unnumbered third level headings for the acknowledgments. All
% acknowledgments go at the end of the paper. Do not include acknowledgments in
% the anonymized submission, only in the final paper.



%%% Local Variables:
%%% TeX-master: "neurips_2020_main"
%%% End:
