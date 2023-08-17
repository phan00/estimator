import numpy as np
from IPython.display import display, Math, Latex


def Display_ltx(*args):
    ltx = ''
    for arg in args:
        if isinstance(arg, str):
            ltx = ltx + arg
        if isinstance(arg, np.ndarray):
            ltx = ltx + to_latex(arg)
    display(Math(ltx))


def convert2latex(*args):
    ltx = ''
    for arg in args:
        if isinstance(arg, str):
            ltx = ltx + arg
        if isinstance(arg, np.ndarray):
            ltx = ltx + to_latex(arg)
    return Math(ltx)



def to_latex(M):
    latex = '\\begin{bmatrix}\n'
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            v = M[r, c]
            if isinstance(v, str):
                latex = latex + v
            else:
                latex = latex + "{:.2f}".format(v)
            
            if c!=(M.shape[1]-1):
                latex = latex + ' & '
            elif r!=(M.shape[0]-1):
                latex = latex + '\\\\\n'
    latex = latex + '\n\\end{bmatrix}'

    print(latex)
    
    return latex
