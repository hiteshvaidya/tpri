# Target Propagation through Layer Inverses
The present code implements an ideal formulation of Target Propagation using regularized inverses
computed analytically rather than using some reverse layer optimized to approximate the inverse.

The code focuses on Recurrent Neural Networks for which vanishing/exploding gradients phenomena are known to impede the performance of a classical gradient back-propagation formula. The experiments demonstrate that TP may be beneficial for optimizing long sequences with RNNs.

The main part of the code consisted in modifying the structure of classical RNNs by adding a target_prop function
in the definition of the RNN, see ``src/model/rnn.py``.
The optimization is done in ``src/optim/run_optimizer.py``.
The following code is provided to reproduce the plots in the paper.


## Setup
Create a conda environment and activate it  
``conda create -n target_prop python=3.8``  
``conda activate target_prop``

Install dependencies  
``conda install seaborn matplotlib pandas``

For PyTorch, the installation depends on your OS. For Mac for example, use  
``conda install pytorch torchvision -c pytorch``

## Setup PythonPath
Add the parent directory to the PYTHONPATH environment variable: </br>
For Unix-like systems (Linux, macOS): </br>
`export PYTHONPATH=$PYTHONPATH:/path/to/parent_directory`

For Windows: </br>
`set PYTHONPATH=%PYTHONPATH%;C:\path\to\parent_directory`

Replace `/path/to/parent_directory` or `C:\path\to\parent_directory` with the actual path to your parent directory.

**Other way is using `sys` module** </br>
```
import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
```

In Windows PowerShell, you need to use a different syntax to set environment variables. Here's the correct way to set the PYTHONPATH in PowerShell:

```powershell
$env:PYTHONPATH += ";D:\tpri"
```

This command appends the path "D:\tpri" to the existing PYTHONPATH environment variable. If you want to completely replace the existing PYTHONPATH, you can use:

```powershell
$env:PYTHONPATH = "D:\tpri"
```

After setting the PYTHONPATH, you can verify it by running:

```powershell
echo $env:PYTHONPATH
```

If you prefer to use the Command Prompt (cmd.exe) instead of PowerShell, you can use the original syntax:

```cmd
set PYTHONPATH=%PYTHONPATH%;D:\tpri
```

Remember that these changes to the environment variables are only temporary and will be lost when you close the terminal. If you want to make permanent changes to your PYTHONPATH, you should modify the system environment variables through the Windows Control Panel or use a script that sets these variables each time you open a new terminal session.

## Experiments
To reproduce the plots presented in the paper run from the folder exp   
``python paper_conv_plots.py``
``python heatmap_reg_stepsize.py``
``python heatmap_perf.py``
``python sensitivity_analysis.py``
``python grad_behavior.py``

The file ``exp_rnn.py`` illustrates a simple pipeline for an experiment on RNNs. The code is composed of data
 generation in ``src/data/get_data.py``, model definition in ``src/model/make_model.py`` and optimization in ``src/optim
 /run_optimizer.py``. They are wrapped in the function ``exp/exp_neck.py`` that is further used with pipeline tools
  presented in the folder ``utils_pipeline``



## Contact
You can report issues and ask questions in the repository's issues page. If you choose to send an email instead, please direct it to Vincent Roulet at vroulet@uw.edu and include [tpri] in the subject line.

## Paper
**Target Propagation through Layer Inverses**  
Vincent Roulet, Zaid Harchaoui.   
*arXiv preprint*  

Reference
```
@article{roulet2023target,
  title={Target Propagation via Regularized Inversion},
  author={Roulet, Vincent and Harchaoui, Zaid},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```

## License
This code has a GPLv3 license.
