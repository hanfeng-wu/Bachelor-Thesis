
\newcommand{\mytitle}{Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE-Solvers}

% tweak spacing...
\newcommand{\myspacepara}{\vspace{-0.5em}}
\newcommand{\myspacesec}{\vspace{-0.8em}}
\newcommand{\myspacessec}{\vspace{-0.5em}}

% notation
\newcommand{\pde}{\mathcal{P}}         % PDE ops
\newcommand{\pdec}{\pde_{s}}
\newcommand{\manifsrc}{\mathscr{S}}    % coarse / "source"
\newcommand{\pder}{\pde_{R}}
\newcommand{\manifref}{\mathscr{R}}

% vc - coarse solutions
\renewcommand{\vc}[1]{\vs_{#1}}            % plain coarse state at time t
\newcommand{\vcN}{\vs}                     % plain coarse state without time 
% vc - coarse solutions, modified by correction
\newcommand{\vct}[1]{\tilde{\vs}_{#1}}     % modified / over time at time t
\newcommand{\vctN}{\tilde{\vs}}            % modified / over time without time
% vr - fine/reference solutions
\renewcommand{\vr}[1]{\mathbf{r}_{#1}}            % fine / reference state at time t , never modified
\newcommand{\vrN}{\mathbf{r}}                     % plain coarse state without time 

\newcommand{\project}{\mathcal{T}}           % transfer operator fine <> coarse
\newcommand{\loss}{\mathcal{L}}              % generic loss function
\newcommand{\nn}{f_{\theta}}
\newcommand{\dt}{\Delta t}                   % timestep
\newcommand{\corrPre}{\mathcal{C}_{\text{pre}}}            % analytic correction , "pre computed"
%\newcommand{\ct}{\mathcal{C}_{\text{sol}}}           % time aware correction , avoid low caps t , old
\newcommand{\corr}{\mathcal{C}}                         % just C for now...
\newcommand{\nnfunc}{F} % {\text{NN}}

\newcommand{\sol}[1]{SOL$_{#1}$}
\newcommand{\non}[1]{NON$_{#1}$}
\newcommand{\pre}[1]{PRE$_{\text{#1}}$}

\title{\mytitle}

% The \author macro works with any number of authors. There are two commands
% used to separate the names and addresses of multiple authors: \And and \AND.
%
% Using \And between authors leaves it to LaTeX to determine where to break the
% lines. Using \AND forces a line break at that point. So, if LaTeX puts 3 of 4
% authors names on the first line, and the last on the second line, try using
% \AND instead of \And before the third author name.

% org:

% \author{%
%   Kiwon Um \\ %\thanks{\url{https://perso.telecom-paristech.fr/kum/}} \\
%   LTCI, Telecom Paris, IP Paris \\
%   %19 Place Marguerite Perey,
%   Palaiseau 91120, France \\
%   \texttt{kiwon.um@telecom-paris.fr} \\
%   % examples of more authors
%   \And
%   Yun (Raymond) Fei \\
%   Columbia University \\
%   New York 10027, USA \\
%   \texttt{yf2320@columbia.edu} \\
%   \AND
%   Philipp Holl \\
%   Technical University of Munich \\
%   %Boltzmannstr. 3,
%   Garching 85748, Germany \\
%   \texttt{philipp.holl@tum.de} \\
%   \And
%   Robert Brand \\
%   Technical University of Munich \\
%   %Boltzmannstr. 3,
%   Garching 85748, Germany \\
%   \texttt{robert.brand@tum.de} \\
%   \And
%   Nils Thuerey \\
%   Technical University of Munich \\
%   %Boltzmannstr. 3,
%   Garching 85748, Germany \\
%   \texttt{nils.thuerey@tum.de} \\
% }

% shortened header
\author{%
	Kiwon Um$^{1,2}$
	%\texttt{kiwon.um@telecom-paris.fr}
	\And
	Robert Brand$^{1}$
	%\texttt{robert.brand@tum.de}
	\And
	Yun (Raymond) Fei$^{3}$
	%\texttt{yf2320@columbia.edu}
	\And
	Philipp Holl$^{1}$
	%\texttt{philipp.holl@tum.de}
	\And
	Nils Thuerey$^{1}$
	%\texttt{nils.thuerey@tum.de}\\
	\AND
	\textnormal{$^{1}$Technical University of Munich, $^{2}$LTCI, Telecom Paris, IP Paris, $^{3}$Columbia University}\\\\
	\texttt{kiwon.um@telecom-paris.fr, robert.brand@tum.de}\\\texttt{yf2320@columbia.edu}, \texttt{\{philipp.holl, nils.thuerey\}@tum.de}
}

\hypersetup{pdfinfo={
Title=\mytitle,
Author={Kiwon Um, Robert Brand, Yun (Raymond) Fei, Philipp Holl, Nils Thuerey}
}}

% \affil[1]{Technical University of Munich}
% \affil[2]{LTCI, Telecom Paris, IP Paris}
% \affil[3]{Columbia University}