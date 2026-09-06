% chulet.cls -- A Cambridge University Letterhead class for LaTeX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simon's Lab Letter Class for 2001
% Modified by Robert Whittaker for use in DAMTP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simon Moore, University of Cambridge, Computer Laboratory
% Please use freely (it's just a hack!)
% Source from: http://www.cl.cam.ac.uk/~swm11/letters/index.html
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Class Options:
%  usebwarms    - uses a black and white version of the Univ. arms
%  usecmfont    - uses computer modern font rather than Sabon (works
%                 better with xdvi
%  usetimesfont - uses times font rather than Sabon
%
%  e.g.:
%  \documentclass[usecmfont,usebwarms,usetimesfont]{/homes/swm11/texmacros/letters/lablet2001}
%
% Compulsory commands:
%  \fromwhome{title}{your name}{qualifications}{job-title}{email}{phone}{fax}
%  \towhom{to name}{to address}
%  \begin{letter}[salutation name]
%  \end{letter}
%
%  The 'to name' is used in address, the optional 'salutation name' is
%  used in 'Dear ... ,'. In its absence, 'to name' is used there instead.
%
% Extra commands:
%  \lettertitle{title of letter}
%  \setdate{date string}
%  \setsigned{greeting}{signature}
%  \setsalutation{salutation}
%  \cclist{list of people\\to cc the\\message to}
%  \enclosures{list of enclosures} 
%  \reference{reference string}

\ProvidesClass{uhhlet}[2013/07/14]

\typeout{Univeristy of Cambridge Dept of Chemistry letter class by Alex Thom}
\typeout{Adapted from Churchill College Letter class by Alex Thom}
\typeout{Adapted from DAMTP Letter class by Robert Whittaker}
\typeout{Adapted from Simon Moore's Computer Lab Letter Class}
%\usepackage{showframe}
%\usepackage{layout}
%\usepackage{sabon}                % Univ. font is Sabon!
\renewcommand{\ttdefault}{cmtt}   % but don't use courier for typewriter
%I'm not sure this has any effect with the sabon package we have in DAMTP

\usepackage{color}
\usepackage{graphicx}
\usepackage{fancyhdr}
\usepackage{marvosym}
\usepackage{scrextend}
%\usepackage{tweaklist}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A few new options (rest are passed through to article class).



\DeclareOption{usebwarms}{\AtBeginDocument{\bwarms}}
\DeclareOption{usetimesfont}{\typeout{Replacing Sabon with Times}
  \AtEndOfClass{\usepackage{times}}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\ProcessOptions\relax

\LoadClass[10.75pt,a4paper]{article}

% page layout
%\pagestyle{plain}
\setlength{\oddsidemargin}{0mm}
\setlength{\evensidemargin}{0mm}
\setlength{\topmargin}{-16mm}
\setlength{\textwidth}{16cm}
\setlength{\textheight}{220mm}
\setlength{\headheight}{25mm}
\setlength{\footskip}{20mm}

\parskip 10pt
\parindent 0.0pt

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main content commands
\newcommand{\fromtitle}{your title?}
\newcommand{\fromnamedetails}{Use $\backslash$from to speficfy your name,
position, email, phone, fax}
\newcommand{\fromquoldetails}{your qualifications?}
\newcommand{\fromjobdetails}{your position?}
\newcommand{\fromemaildetails}{email?}
\newcommand{\fromphonedetails}{phone?}
\newcommand{\fromfaxdetails}{fax?}
\newcommand{\fromwhom}[7]{
        \renewcommand{\fromtitle}{#1}
        \renewcommand{\fromnamedetails}{#2}
        \renewcommand{\fromquoldetails}{#3}
        \renewcommand{\fromjobdetails}{#4}
        \renewcommand{\fromemaildetails}{#5}
        \renewcommand{\fromphonedetails}{#6}
        \renewcommand{\fromfaxdetails}{#7}
        }

\newcommand{\towhomaddress}{Use $\backslash$towhom to set the name
and address}
\newcommand{\towhomname}{Name?}
\newcommand{\towhom}[2]{
        \renewcommand{\towhomname}{#1}
        \renewcommand{\towhomaddress}{#2}
}

\newcommand{\salutationname}{Name of recipient}

\newcommand{\referencedetails}{\null}
\newcommand{\reference}[1]{\renewcommand{\referencedetails}{#1}}

\newcommand{\datedetails}{\today}
\newcommand{\setdate}[1]{\renewcommand{\datedetails}{#1}}

\newcommand{\salutation}{Dear}
\newcommand{\setsalutation}[1]{\renewcommand{\salutation}{#1}}

\newcommand{\yoursdetails}{Yours sincerely,}
\newcommand{\signeddetails}{\fromnamedetails}

\newcommand{\setsigned}[2]{\renewcommand{\yoursdetails}{#1}\renewcommand{\signeddetails}{#2}}

\newcommand{\encldetails}{}
\newcommand{\enclosures}[1]{\renewcommand{\encldetails}{\vspace*{10em}\textbf{Encl}:~~\begin{minipage}[t]{100mm}#1\end{minipage}}}

\newcommand{\cclistdetails}{}
\newcommand{\cclist}[1]{\renewcommand{\cclistdetails}{\vspace*{1em}cc:~~\begin{minipage}[t]{100mm}#1\end{minipage}}}

\newcommand{\cuarms}{\includegraphics[width=40mm]{uhh-logo.png}} %65mm from UC style guidelines v5
%\newcommand{\bwarms}{\typeout{Using B&W version of the arms}\renewcommand{\cuarms}{\includegraphics{BWUni3}}}
%\newcommand{\cuarms}{\includegraphics{CUni3}}
%\newcommand{\bwarms}{\typeout{Using B&W version of the arms}\renewcommand{\cuarms}{\includegraphics{BWUni3}}}

\newcommand{\dostamp}{}
\newcommand{\stamp}[1]{\renewcommand{\dostamp}{#1}}

\newcommand{\DeptHead}{\fontsize{14pt}{14pt}}

\renewcommand{\headrulewidth}{0pt}
\newcommand{\lettertitle}[1]{\begin{center}\bf#1\end{center}}
\fancypagestyle{firststyle}
{
\fancyhf[HL]{%
\setlength{\unitlength}{1mm}%
\begin{picture}(70,0)(0,0)
\put(-13,-1){\makebox(66,20)[tl]{\cuarms}}%14mm from top and 12mm from left
\put(1,-1){\makebox(52,5)[tl]{\DeptHead}}
\end{picture}}
\fancyhf[HR]{%
\begin{flushright}{\bf\fromtitle~\fromnamedetails~\fromquoldetails}\\{\it\fromjobdetails}\end{flushright}}
%\fancyhf[FR]{\vspace{-35mm}\swmfoot}
}
\pagestyle{fancyplain}



\newcommand{\dolettertop}{%
\thispagestyle{firststyle}%
\begin{flushright}
\swmfoot
\end{flushright}
\makeatletter
\ifx\towhomname\@empty
\setlength{\unitlength}{1mm}
%\begin{picture}(160,80)(0,0)
\begin{picture}(190,35)(0,0)
%\put(95,60){\makebox(75,25)[tl]{\parbox{75mm}{\begin{flushright}{\bf\fromtitle~\fromnamedetails~\fromquoldetails}\\{\it\fromjobdetails}\\\fromphonedetails\\\fromemaildetails\end{flushright}}
%}}
\put(0,10){\makebox(100,25)[tl]{\begin{tabular}{@{}p{90mm}@{}}\towhomname\\\towhomaddress\end{tabular}}}
\put(0,40){\makebox(160,00)[tr]{\dostamp}}
\end{picture}
\else
\fi
\hfill\referencedetails \\
\datedetails

\salutation\ \salutationname,\\
%\enlargethispage{-25mm}%
%\begin{figure}[b!]\swmfoot\end{figure}
\makeatother
}


\renewcommand{\footnoterule}{}
\renewcommand{\thefootnote}{}

\newcommand{\swmfoot}{\fontsize{9pt}{10pt}\rm\hfill\begin{tabular}{@{}r@{~~}r@{ }r@{}}
 & & \\
\multicolumn{2}{@{}r@{}}{University of Hamburg}\\
\multicolumn{2}{@{}r@{}}{Martin-Luther-King-Platz 6}\\
\multicolumn{2}{@{}r@{}}{22761 Hamburg}\\
\multicolumn{2}{@{}r@{}}{Germany}\\
 & \\
 & \fromphonedetails\\
 & \fromemaildetails\\
\end{tabular}
}


\newcommand{\doletterbottom}{
{
%\centering
\begin{addmargin}[3em]{0em}
\yoursdetails

%\hspace{-2.5em}\includegraphics[scale=0.3, trim={0cm 1cm 0cm 0cm}, clip]{HGAB}


\signeddetails

\encldetails

\cclistdetails
\end{addmargin}

}
\cleardoublepage}

\newenvironment{letter}[1]
  {\renewcommand{\salutationname}{#1}%
   \dolettertop}
  {\doletterbottom}

\def\opening#1{\ifx\@empty\fromaddress
 \thispagestyle{firstpage}
 \hspace*{\longindendation}\today\par
 \else \thispagestyle{empty}
 {\centering\fromaddress \vspace{5\parskip} \\
\today\hspace*{\fill}\par}
 \fi
 \vspace{3\parskip}
 {\raggedright \toname \\ \toaddress \par}\vspace{3\parskip}
 \noindent #1\par\raggedright\parindent 5ex\par
 }
