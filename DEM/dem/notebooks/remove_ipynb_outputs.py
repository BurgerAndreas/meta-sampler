"""
usage: 
python remove_ipynb_outputs.py plot_gmm.ipynb
"""
import sys
import os
import nbformat


def remove_outputs(nb):
    """remove the outputs from a notebook"""
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None

if __name__ == '__main__':
    fname = sys.argv[1]
    with open(fname, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    remove_outputs(nb)
    # print(current.writes(nb, 'json'))
    
    # save to new file
    output_fname = fname.replace('.ipynb', '_nooutput.ipynb')
    # remove file if it exists
    if os.path.exists(output_fname):
        os.remove(output_fname)
    with open(output_fname, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
