import re


def search_and_replace_with_gradient(text: str) -> str:
    text = re.sub(r'\d.\d\d', '\\\\gradient{\g<0>}', text)
    return text


if __name__ == '__main__':
    input_txt = r"""
% Please add the following required packages to your document preamble:
% \usepackage{multirow}
% \usepackage{graphicx}
\begin{table}[]
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccccccc}
\multicolumn{2}{l}{\textbf{OOD Metric $\rightarrow$}} & \multicolumn{2}{c}{\textbf{$\ell_{1}$}} & \multicolumn{2}{c}{\textbf{$D_{KL}$}} & \multicolumn{2}{c}{\textbf{$\mathcal{L}$}} & \multicolumn{2}{c}{\textbf{$WAIC$}} & \multicolumn{2}{c}{\textbf{$DoSE_{(l_{1}, D_{KL}, \mathcal{L})}$}} \\ \hline
\textbf{OOD Dataset $\downarrow$} & \multicolumn{1}{c}{} & AU_{ROC} & AU_{PRC} & AU_{ROC} & AU_{PRC} & AU_{ROC} & AU_{PRC} & AU_{ROC} & AU_{PRC} & AU_{ROC} & AU_{PRC} \\
\multirow{3}{*}{\textbf{\begin{tabular}[c]{@{}l@{}}BraTS\\ T2\\ HM\end{tabular}}} & all & 0.89 & 0.86 & 0.68 & 0.61 & 0.85 & 0.81 & 0.93 & 0.95 & 0.78 & 0.70 \\
 & healthy & 0.85 & 0.84 & 0.73 & 0.71 & 0.79 & 0.79 & 0.87 & 0.91 & 0.78 & 0.76 \\
 & lesion & 0.92 & 0.89 & 0.63 & 0.55 & 0.90 & 0.87 & 0.99 & 0.99 & 0.78 & 0.70 \\ \hline
\multirow{3}{*}{\textbf{\begin{tabular}[c]{@{}l@{}}BraTS \\ T2\end{tabular}}} & all & 0.87 & 0.85 & 0.69 & 0.62 & 0.83 & 0.81 & 0.91 & 0.92 & 0.79 & 0.72 \\
 & healthy & 0.84 & 0.83 & 0.74 & 0.72 & 0.77 & 0.78 & 0.84 & 0.85 & 0.80 & 0.77 \\
 & lesional & 0.92 & 0.90 & 0.64 & 0.56 & 0.89 & 0.86 & 0.98 & 0.98 & 0.78 & 0.70 \\ \hline
\multirow{3}{*}{\textbf{\begin{tabular}[c]{@{}l@{}}BraTS \\ T1 \\ HM\end{tabular}}} & all & 0.92 & 0.91 & 0.73 & 0.64 & 0.88 & 0.88 & 0.92 & 0.95 & 0.82 & 0.76 \\
 & healthy & 0.88 & 0.88 & 0.76 & 0.71 & 0.83 & 0.85 & 0.86 & 0.91 & 0.82 & 0.79 \\
 & lesional & 0.96 & 0.95 & 0.70 & 0.60 & 0.95 & 0.93 & 0.99 & 0.99 & 0.83 & 0.76 \\ \hline
\multirow{3}{*}{\textbf{\begin{tabular}[c]{@{}l@{}}BraTS \\ T1\end{tabular}}} & all & 0.95 & 0.94 & 0.77 & 0.74 & 0.91 & 0.90 & 0.97 & 0.98 & 0.86 & 0.82 \\
 & healthy & 0.94 & 0.94 & 0.82 & 0.82 & 0.88 & 0.88 & 0.94 & 0.96 & 0.88 & 0.86 \\
 & lesional & 0.97 & 0.96 & 0.73 & 0.66 & 0.96 & 0.96 & 1.00 & 1.00 & 0.84 & 0.78 \\ \hline
\multirow{3}{*}{\textbf{\begin{tabular}[c]{@{}l@{}}BraTS \\ T2 \\ HM H-flip\end{tabular}}} & all & 0.86 & 0.83 & 0.66 & 0.60 & 0.82 & 0.79 & 0.89 & 0.92 & 0.77 & 0.69 \\
 & healthy & 0.81 & 0.80 & 0.69 & 0.68 & 0.76 & 0.76 & 0.83 & 0.88 & 0.76 & 0.73 \\
 & lesional & 0.91 & 0.87 & 0.63 & 0.50 & 0.89 & 0.85 & 0.98 & 0.98 & 0.78 & 0.69 \\ \hline
\multirow{3}{*}{\textbf{\begin{tabular}[c]{@{}l@{}}BraTS \\ T2 \\ HM V-flip\end{tabular}}} & all & 0.94 & 0.92 & 0.63 & 0.59 & 0.92 & 0.90 & 0.99 & 0.99 & 0.79 & 0.73 \\
 & healthy & 0.94 & 0.93 & 0.75 & 0.72 & 0.89 & 0.89 & 0.99 & 0.99 & 0.85 & 0.82 \\
 & lesional & 0.94 & 0.93 & 0.49 & 0.46 & 0.94 & 0.93 & 0.99 & 0.99 & 0.73 & 0.64 \\ \hline
\multirow{3}{*}{\textbf{\begin{tabular}[c]{@{}l@{}}CamCAN \\ T2 \\ artificial\end{tabular}}} & all & 0.69 & 0.69 & 0.55 & 0.58 & 0.66 & 0.68 & 0.74 & 0.79 & 0.63 & 0.61 \\
 & healthy & 0.51 & 0.51 & 0.51 & 0.50 & 0.52 & 0.52 & 0.68 & 0.73 & 0.52 & 0.50 \\
 & lesional & 0.87 & 0.85 & 0.60 & 0.64 & 0.84 & 0.82 & 0.81 & 0.84 & 0.74 & 0.70 \\ \hline
\textbf{MNIST} & \multicolumn{1}{c}{\textbf{}} & 1.00 & 1.00 & 0.00 & 0.31 & 1.00 & 1.00 & 1.00 & 1.00 & 1.00 & 1.00 \\ \hline
\textbf{Gaussian Noise} & \multicolumn{1}{c}{\textbf{}} & 1.00 & 1.00 & 0.00 & 0.30 & 1.00 & 1.00 & 1.00 & 1.00 & 1.00 & 1.00 \\ \hline
\end{tabular}%
}
\caption{OOD detection performance. $\ell_{1}$ is the mean reconstruction error per pixel for a slice, $D_{KL}$ the KL divergence between the prior and approximate posterior and $\mathcal{L}$ the ELBO.}
\label{tab:ood_results}
\end{table}
"""
    output_txt = search_and_replace_with_gradient(input_txt)
    print(output_txt)
