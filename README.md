# Target Propagation via Regularized Inversion
The present code implements an ideal formulation of target propagation using regularized inverses
computed analytically rather than using some reverse layer optimized to approximate the inverse.

The code focuses on Recurrent Neural Networks for which vanishing/exploding gradients phenomena are known to impede the performance of a classical gradient back-propagation formula. The experiments demonstrate that TP may be beneficial for optimizing long sequences with RNNs.

The main part of the code consisted in modifying the structure of classical RNNs by adding a target_prop function
in the definition of the RNN, see src/model/rnn.py.
The optimization is done in src/optim/run_optimizer.py.
The following code is provided to reproduce the plots in the paper.


## Setup
Create a conda environment and activate it  
``conda create -n target_prop python=3.8``  
``conda activate target_prop``

Install dependencies  
``conda install seaborn matplotlib pandas``

For PyTorch, the installation depends on your OS. For Mac for example, use  
``conda install pytorch torchvision -c pytorch``


## Experiments
To reproduce the plots presented in the paper run from the folder exp   
``python paper_conv_plots.py``  
``python paper_regimes.py``
``python paper_reg_plots.py``


## Contact
You can report issues and ask questions in the repository's issues page. If you choose to send an email instead, please direct it to Vincent Roulet at vroulet@uw.edu and include [tpri] in the subject line.

## Paper
**Target Propagation via Regularized Inversion**  
Vincent Roulet, Zaid Harchaoui.   
*arXiv preprint*  

Reference
```
@article{roulet2021target,
  title={Target Propagation via Regularized Inversion},
  author={Roulet, Vincent and Harchaoui, Zaid},
  journal={arXiv preprint}
}
```

## License
This code has a GPLv3 license.
