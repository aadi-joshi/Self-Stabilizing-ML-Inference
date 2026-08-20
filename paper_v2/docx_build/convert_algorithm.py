"""pandoc's LaTeX reader does not understand the algorithm/algorithmic
environments (no line numbers, no indentation, no caption -- everything
comes out as one unstructured paragraph). This replaces the single
Algorithm 1 float in the docx-tex copy with a hand-written, plain-LaTeX
rendering (bold keywords, manual line numbers, \hspace indentation, a
top/bottom rule) that pandoc converts into a readable Word paragraph block
equivalent in content to the real algorithmic-package output in the PDF.
"""

path = '../ftr_phase_transition_docx.tex'
text = open(path, encoding='utf-8').read()

start_marker = '\\begin{algorithm}[t]'
end_marker = '\\end{algorithm}'

start = text.find(start_marker)
if start == -1:
    print('WARNING: no \\begin{algorithm}[t] found, skipping')
else:
    end = text.find(end_marker, start) + len(end_marker)

    replacement = r'''\noindent\rule{\linewidth}{0.6pt}

\noindent\textbf{Algorithm 1.} \textbf{FTR training with dual-ascent constraint enforcement}
\label{alg:ftr}

\noindent\textbf{Require:} task sequence $T_1, T_2, \ldots$; initial parameters $\theta$; stability budget $\eps$; hyperparameters $\lambda_{\mathrm{init}}, \lambda_{\max}, \eta_\theta, \eta_\lambda, \rho, T$; constrained steps per task $N$

\noindent\textbf{Ensure:} trained parameters $\theta$

\noindent\hspace{0em}1:\ \textbf{for} each task $T_t$ in sequence \textbf{do}

\noindent\hspace{1.8em}2:\ $\lambda \leftarrow \lambda_{\mathrm{init}}$;\ \ $m \leftarrow 0$ \quad\textit{(dual variable and momentum reset at every task boundary)}

\noindent\hspace{1.8em}3:\ $f_{\mathrm{old}} \leftarrow f_\theta$ \quad\textit{(freeze reference model before task $t$)}

\noindent\hspace{1.8em}4:\ run 1 warmup epoch of gradient steps on $\mathcal{L}_{\mathrm{task}}$ alone \quad\textit{($\lambda$ held at $\lambda_{\mathrm{init}}$, constraint not enforced)}

\noindent\hspace{1.8em}5:\ \textbf{for} $n = 1$ to $N$ \textbf{do}

\noindent\hspace{3.6em}6:\ sample batch $x \sim \mathcal{D}_t$

\noindent\hspace{3.6em}7:\ $D \leftarrow \DKL\big(f_\theta(x) \,\|\, f_{\mathrm{old}}(x)\big)$ at temperature $T$

\noindent\hspace{3.6em}8:\ $\theta \leftarrow \theta - \eta_\theta \nabla_\theta\big[\mathcal{L}_{\mathrm{task}}(\theta) + \lambda(D - \eps)\big]$ \quad\textit{(primal step, Eq.~2)}

\noindent\hspace{3.6em}9:\ $m \leftarrow \rho\, m + \eta_\lambda (D - \eps)$ \quad\textit{(dual ascent, Eq.~4)}

\noindent\hspace{3.6em}10:\ $\lambda \leftarrow \max\big(0,\, \min(\lambda_{\max},\, \lambda + m)\big)$

\noindent\hspace{1.8em}11:\ \textbf{end for}

\noindent\hspace{0em}12:\ \textbf{end for}

\noindent\hspace{0em}13:\ \textbf{return} $\theta$

\noindent\rule{\linewidth}{0.6pt}
'''

    text = text[:start] + replacement + text[end:]
    open(path, 'w', encoding='utf-8').write(text)
    print('algorithm block converted to plain-text rendering')
