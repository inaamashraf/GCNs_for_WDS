import os, datetime
import numpy as np
from utils.utils import *
from models.models import *
from train_test import train, test
import argparse


def create_cli_parser():
    # ----- ----- ----- ----- ----- -----
    # Command line arguments
    # ----- ----- ----- ----- ----- -----
    parser  = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        default = 'train_test',
                        type    = str,
                        choices = ['train_test', 'evaluate'],
                        help    = "train_test i.e. train and test a new model, or evaluate i.e. evaluate on an already trained model; default is train_test. ")
    parser.add_argument('--warm_start',
                        default = False,
                        type    = bool,
                        help    = "specify True if you want to further train a partially trained model. model_path must also be specified; default is False.")
    parser.add_argument('--model_path',
                        default = None,
                        type    = str,
                        help    = "specify model path in case of re-training or evaluation; default is None.")
    parser.add_argument('--model',
                        default = 'm_GCN',
                        type    = str,
                        choices = ['m_GCN', 'ChebNet'],
                        help    = "m_GCN or ChebNet; default is m_GCN.")   
    parser.add_argument('--n_days',
                        default = '30',
                        type    = int,
                        help    = "number of days of data to be used for training; default is 30 days.")
    parser.add_argument('--batch_size',
                        default = '48',
                        type    = int,
                        help    = "mini-batch size used for training; default is 48.")
    parser.add_argument('--n_epochs',
                        default = '250',
                        type    = int,
                        help    = "number of epochs of training; default is 5000.")    
    parser.add_argument('--lr',
                        default = '1e-4',
                        type    = float,
                        help    = "learning rate; default is 1e-4.")
    parser.add_argument('--decay',
                        default = '0',
                        type    = float,
                        help    = "weight decay for Adam Optimizer; defaults is 0.")
    parser.add_argument('--n_aggr',
                        default = '45',
                        type    = int,
                        help    = "number of GCN layers; default is 45.")
    parser.add_argument('--n_hops',
                        default = '1',
                        type    = int,
                        help    = "number of hops within each GCN layer; default is 1.")
    parser.add_argument('--n_mlp',
                        default = '2',
                        type    = int,
                        help    = "number of layers in the MLP; default is 2.")
    parser.add_argument('--latent_dim',
                        default = '96',
                        type    = int,
                        help    = "latent dimension; default is 96.")
    return parser


def run(args):

    """ Creating directories. """
    file_dir = os.path.dirname(os.path.realpath(__file__)) 
    if not os.path.isdir(os.path.join(file_dir, "tmp")):
        os.system('mkdir ' + os.path.join(file_dir, "tmp"))
    save_dir = os.path.join(file_dir, "tmp", str(datetime.date.today()))
    if not os.path.isdir(save_dir):
        os.system('mkdir ' + save_dir)

    """ 
        List of installed sensors as specified by Vrachimis et al. 
        https://github.com/KIOS-Research/BattLeDIM
    """     
    installed_sensors = np.array([0, 3, 30, 53, 104, 113, 162, 187, 214, 228, \
                                287, 295, 331, 341, 409, 414, 428, 457, 468, 494, \
                                505, 515, 518, 548, 612, 635, 643, 678, 721, 725, \
                                739, 751, 768])
    
    """ Computing the number of samples based on the specified number of days. """
    n_samples = 4 * 24 * args.n_days
   
    """ Specifying the model and printing the number of parameters. """
    if args.model == 'ChebNet': 
        model = ChebNet(in_dim = 1, 
                        out_dim = 1, 
                        latent_dim = args.latent_dim, 
                        K = args.n_hops
                        ).to(device)
    elif args.model == 'm_GCN':
        model = m_GCN(in_dim = 1, 
                      out_dim = 1, 
                      edge_dim = 3, 
                      latent_dim = args.latent_dim, 
                      batch_size = args.batch_size, 
                      n_aggr = args.n_aggr, 
                      n_hops = args.n_hops, 
                      num_layers = args.n_mlp
                      ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print('Total parameters: ', total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable parameters: ', trainable_params)

    """ Creating an output file to log progress. """
    out_f = open(save_dir+"/output_"+args.model+"_"+str(n_samples)+"_"+str(datetime.date.today())+".txt", "a")

    if args.mode == 'train_test':
        """ Creating Graph for L-Town WDN from the data files. """
        inp_file = "networks/L-Town/Toy/L-TOWN.inp"
        path_to_data = "networks/L-Town/Toy/Measurements_All_Pressures.csv"
        wdn_graph = create_graph(inp_file, path_to_data)   

        """ Normalizing pressure values using the limits used for generating the data. """
        X_min, X_max = 0, 80
        wdn_graph.X, wdn_graph.edge_attr = normalize(wdn_graph.X, _min=X_min, _max=X_max), normalize(wdn_graph.edge_attr, dim=1)

        """ Creating train-val-test data based on the specified number of samples. """
        X_tvt = wdn_graph.X[:n_samples]

        """ Creating train-val-test splits. """
        tv_N = int(0.8 * n_samples)
        X_tv, X_test = X_tvt[:tv_N, :].clone(), X_tvt[tv_N:n_samples, :].clone()    
        
        """ Training """
        state, model_path = train(X_tv, wdn_graph.edge_indices, wdn_graph.edge_attr, model, installed_sensors, args, save_dir, out_f)

        """ Testing """
        Y, Y_hat, test_losses  = test(X_test, wdn_graph.edge_indices, wdn_graph.edge_attr, model, installed_sensors, args, save_dir, out_f)

        """ Analysis """
        mean_abs_errors, abs_errors, p_coefs = plot_errors(Y[:,:,0], Y_hat[:,:,0], args, save_dir)
        print("Mean Absolute Error and PCC: ", np.round(abs_errors.mean().item(), 6), np.round(np.mean(p_coefs), 6))
        print("Mean Absolute Error and PCC: ", np.round(abs_errors.mean().item(), 6), np.round(np.mean(p_coefs), 6), file=out_f)
        plot_graph(inp_file, wdn_graph.edge_indices, args, save_dir, node_errors=mean_abs_errors, plot=True, labels=True, cmap='Reds', flag='errors')


    elif args.mode == 'evaluate':
        """ Creating Graph for L-Town WDN from the data files """
        inp_file = "networks/L-Town/Real/L-TOWN_Real.inp"
        path_to_data = "networks/L-Town/Real/Measurements_All_Pressures.csv"    
        wdn_graph = create_graph(inp_file, path_to_data)   

        """ Normalizing pressure values using the limits used for generating the data """
        X_min, X_max = 0, 80
        wdn_graph.X, wdn_graph.edge_attr = normalize(wdn_graph.X, _min=X_min, _max=X_max), normalize(wdn_graph.edge_attr, dim=1)

        """ Creating train-val-test data based on the specified number of samples. """
        X_test = wdn_graph.X[n_samples:]

        """ Evaluating """
        Y, Y_hat, test_losses  = test(X_test, wdn_graph.edge_indices, wdn_graph.edge_attr, model, installed_sensors, args, save_dir, out_f)

        """ Analysis """
        mean_abs_errors, abs_errors, p_coefs = plot_errors(Y[:,:,0], Y_hat[:,:,0], args, save_dir)
        print("Mean Absolute Error and PCC: ", np.round(abs_errors.mean().item(), 6), np.round(np.mean(p_coefs), 6))
        print("Mean Absolute Error and PCC: ", np.round(abs_errors.mean().item(), 6), np.round(np.mean(p_coefs), 6), file=out_f)
        plot_graph(inp_file, wdn_graph.edge_indices, args, save_dir, node_errors=mean_abs_errors, plot=True, labels=True, cmap='Reds', flag='errors')
        


if __name__ == '__main__':
    parser = create_cli_parser()

    args = parser.parse_args()    

    print(args)
    run(args)

    