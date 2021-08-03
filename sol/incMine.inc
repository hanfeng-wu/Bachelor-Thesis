\usepackage{acro}

\DeclareAcronym{nn}{
short = NN,
long = neural network,
}

\DeclareAcronym{cnn}{
short = CNN,
long = convolutional neural network,
}

\DeclareAcronym{cnns}{
short = CNNs,
long = convolutional neural networks,
}

\DeclareAcronym{pde}{
short = PDE,
long = partial differential equation,
}

\DeclareAcronym{2afc}{
short = 2AFC,
long = two-alternative forced choice,
}

\DeclareAcronym{hvs}{
short = HVS,
long = human visual system,
}

\DeclareAcronym{sph}{
short = SPH,
short-indefinite = an,
long = smoothed particle hydrodynamics,
}

\DeclareAcronym{flip}{
short = FLIP,
long = fluid-implicit-particle,
}

\DeclareAcronym{simple}{
short = SIMPLE,
long = semi-implicit method for pressure linked equations,
}

\DeclareAcronym{eno}{
short = ENO,
long = essentially non-oscillatory,
}

\DeclareAcronym{rmse}{
short = RMSE,
long = root-mean-square error,
}

\DeclareAcronym{psnr}{
short = PSNR,
long = peak signal-to-noise ratio,
}

\DeclareAcronym{ssim}{
short = SSIM,
long = structure similarity metric,
}

\DeclareAcronym{tgv}{
short = TGV,
long = Taylor-Green vortex,
}

\DeclareAcronym{ncm}{
short = NCM,
long = near-convergence consistency metric,
}

% note - alternatives in math_commands.tex
\newcommand{\maxhi}[1]{\underline{\textbf{#1}}}
\newcommand{\myeqref}[1]{Eq.~\ref{#1}}
\newcommand{\myeqsref}[2]{Eq.~\ref{#1} and \ref{#2}}
\newcommand{\myfigref}[1]{Fig.~\ref{#1}}
\newcommand{\mytabref}[1]{Table~\ref{#1}}
\newcommand{\myfigsref}[2]{Figs.~\ref{#1}--\ref{#2}}
\newcommand{\mysecref}[1]{Sec.~\ref{#1}}
\newcommand{\myappref}[1]{App.~\ref{#1}}

\usepackage[normalem]{ulem}     % for marking strike through
\usepackage{color}

\definecolor{R}{rgb}{0.9, 0, 0}
\definecolor{G}{rgb}{0, 0.6, 0}
\definecolor{B}{rgb}{0, 0, 1.0}

\definecolor{kCol}{rgb}{0.1, 0, 0.9}
\definecolor{nCol}{rgb}{0.8, 0.3, 0}
\definecolor{yCol}{rgb}{0.9, 0, 0.9}
\definecolor{todoCol}{rgb}{0.9, 0.0, 0}

% \newcommand{\myfix}[2]{\sout{#1}{\color{R}{#2}}}
% \newcommand{\mynote}[2]{{\color{G}[\uline{\textbf{#1:}}~#2]}}

% \newcommand{\new}[1]{{\color{B}#1}}
% \newcommand{\ku}[1]{{\color{kCol}#1}}
% \newcommand{\nt}[1]{{\color{nCol}#1}}
% \newcommand{\yf}[1]{{\color{yCol}#1}}
\newcommand{\new}[1]{#1}
% \renewcommand{\ku}[1]{#1}
% \renewcommand{\nt}[1]{#1}
% \renewcommand{\yf}[1]{#1}

% \newcommand{\todo}[1]{{\color{todoCol}[TODO: #1]}}

% \newenvironment{obsolete}{\footnotesize\color{gray}}{\par}

% \usepackage{comment}
% \excludecomment{obsolete}
