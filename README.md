# Target Propagation through layer inverses

## Setup
Create a conda environment and activate it  
``conda create -n target_prop python=3.8``  
``conda activate target_prop``

Install dependencies  
``conda install seaborn matplotlib pandas``

For PyTorch, the installation depends on your OS. For Mac for example, use  
``conda install pytorch torchvision -c pytorch``


## Experiments
The main part of the code consisted in modidfying the structure of classical RNNs by adding a target_prop function
in the definition of the RNN, see src/model/rnn.py.
The optimization is done in src/optim/run_optimizer.py.
The following code is provided to reproduce the plots in the paper.
We are currently working on an user-friendly implementation of the target propagation algorithm
to make the approach more broadly available.

To reproduce the plots presented in the paper run from the folder exp   
``python paper_conv_plots.py``  
``python paper_regimes.py``
``python paper_reg_plots.py``
