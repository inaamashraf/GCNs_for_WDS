![](m_GCN.png)

# Spatial Graph Convolution Neural Networks for Water Distribution Systems

Official Code for the paper "Spatial Graph Convolution Neural Networks for Water Distribution Systems" (under submission, preprint available at arXiv: https://arxiv.org/abs/2211.09587).

## Simulating scenarios to generate data

Pressure data for all nodes in the WDS can be generated for longer periods of time (Vrachimis et al. https://github.com/KIOS-Research/BattLeDIM). Two types of demands can be used for this purpose:
```
- Toy:    smooth changes in demand patterns.
- Real:   noisy demand patterns that are an approximation of actual demand patterns.
```

Start and end dates can be specified in 'dataset_configuration.yaml' in the respective directory of both demand patterns (networks/L-Town/Toy or networks/L-Town/Real).

Simulations can be run as:
- `python dataset_generator.py 'Toy'` or 
- `python dataset_generator.py 'Real'`

The simulation will produce a csv file named 'Measurements_All_Pressures.csv' in the respective directories. 

## Training and Evaluation

Models can be trained using the `python run.py`. A number of arguments can be passed to specify model types and hyperparameters:
```
- --mode:             train_test i.e. train and test a new model, or evaluate i.e. evaluate on an already trained model; default is train_test.
- --warm_start:       specify True if you want to further train a partially trained model. model_path must also be specified; default is False. 
- --model_path:       specify model path in case of re-training or evaluation; default is None.
- --model:            m_GCN (our model) or ChebNet (model from Hajgato et al.); default is m_GCN.
- --n_days:           number of days of data to be used for training; default is 30 days.
- --batch_size:       mini-batch size used for training; default is 48.
- --n_epochs:         number of epochs of training; default is 5000.
- --lr:               learning rate; default is 1e-4.
- --decay:            weight decay for Adam Optimizer; defaults is 0.
- --n_aggr:           number of GCN layers; default is 45.
- --n_hops:           number of hops within each GCN layer; default is 1.
- --n_mlp:            number of layers in the MLP; default is 2.
- --latent_dim:       latent dimension; default is 96.
```

Both training and evaluation require the model '.inp' file and the dataset '.csv' file with pressure values. The model '.inp' file for Toy demand patterns is provided in the respective directory, while the '.inp' file for demand patterns can be downloaded from https://github.com/KIOS-Research/BattLeDIM. Both files need to be placed in respective directories i.e. (networks/L-Town/Toy or networks/L-Town/Real).

Trained models can be used for evaluation using run.py by specifying the 'evaluate' mode and 'model_path'.

## Citation
### Preprint:
```
@misc{ashraf2022gcn_wds,
  author        = {Ashraf, Inaam and Hermes, Luca and Artelt, Andr{\'{e}} and Hammer, Barbara},
  title         = {Spatial Graph Convolution Neural Networks for Water Distribution Systems},
  year          = {2022},
  month         = nov,
  archiveprefix = {arXiv},
  eprint        = {2211.09587},
  copyright     = {Creative Commons Attribution Share Alike 4.0 International}
}
```
### Repository:
```
@misc{GCNs_for_WDS,
  author        = {Ashraf, Inaam and Hermes, Luca and Artelt, Andr{\'{e}} and Hammer, Barbara},
  title         = {{GCNs_for_WDS}},
  year          = {2022},
  publisher     = {GitHub}
  journal       = {GitHub repository},
  organization  = {CITEC, Bielefeld University, Germany},
  howpublished  = {\url{https://github.com/inaamashraf/GCNs_for_WDS}},
}
```


## Acknowledgments
We gratefully acknowledge funding from the European
Research Council (ERC) under the ERC Synergy Grant Water-Futures (Grant
agreement No. 951424). This research was also supported by the research training
group “Dataninja” (Trustworthy AI for Seamless Problem Solving: Next Generation Intelligence Joins Robust Data Analysis) funded by the German federal
state of North Rhine-Westphalia, and by funding from the VW-Foundation for
the project IMPACT funded in the frame of the funding line AI and its Implications for Future Society.
 