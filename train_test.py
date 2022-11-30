import datetime
import numpy as np
import torch
from utils.utils import load_dataset
from torch_geometric.loader import DataLoader
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

""" Training """
def train(X_tv, edge_indices, edge_attr, model, installed_sensors, args, save_dir, out_f):
    """ Initializing hyperparameters. """
    n_epochs, learn_r, alpha = args.n_epochs, args.lr, args.decay

    """ Initiating the Optimizer and Learning rate scheduler. """
    optimizer = Adam(model.parameters(), lr=learn_r, weight_decay=alpha)    
    lr_decay_step, lr_decay_rate = 300, .75
    opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000, lr_decay_step), gamma=lr_decay_rate)
    
    if args.model_path == None:
        model_path = save_dir+"/model_"+args.model+"_"+str(args.n_aggr)+"_"+str(args.n_hops)+"_"+str(datetime.date.today())+".pt"

    """ Checking if training using a partially trained model. """
    if args.warm_start and args.model_path != None:                
        model_state = torch.load(model_path)
        model.load_state_dict(model_state["model"])
        optimizer.load_state_dict(model_state["optimizer"])

    print("Number of sensors installed: ", len(installed_sensors))
    
    """ Loading dataset and creating test and validation splits and batches. """
    N, n_nodes, _ = X_tv.shape
    idx_train = int(0.75*N) 
    tv_dataset, _ = load_dataset(X_tv, n_nodes, installed_sensors, edge_indices, edge_attr)         
    train_loader = DataLoader(tv_dataset[:idx_train], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(tv_dataset[idx_train:], batch_size=args.batch_size, shuffle=True)
    
    """ Train-validation loop """
    for epoch in tqdm(range(n_epochs)): 

        train_losses = []
        for batch in train_loader:             
            
            model.train()
            optimizer.zero_grad()
            y, y_hat = model(batch)
            train_loss = model.loss(y, y_hat)

            train_loss.backward()
            train_losses.append(train_loss.detach().cpu().item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e10)
            optimizer.step()                   

            del y, y_hat, batch
        
        opt_scheduler.step()        
        del train_loss

        if epoch % 10 == 0:
            model.eval()
            val_losses = []            
            for batch_val in val_loader:                
                with torch.no_grad():
                    y, y_hat = model(batch_val)
                val_loss = model.loss(y, y_hat)
                val_losses.append(val_loss)                
                
            mean_val_losses = torch.mean(torch.stack(val_losses)).detach().cpu().item()           
            del y, y_hat, batch_val, val_loss

            print("Epoch ", epoch, ": Train loss: ", np.round(np.mean(train_losses), 8), \
                " Val loss: ", np.round(mean_val_losses, 8))
            print("Epoch ", epoch, ": Train loss: ", np.round(np.mean(train_losses), 8), \
                " Val loss: ", np.round(mean_val_losses, 8), file=out_f)
    
        if epoch % 200 == 0:
            """ Saving the model. """
            state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }
            print('model path:', model_path)
            print('model path:', model_path, file=out_f)
            torch.save(state, model_path)
        
    return state, model_path

""" Testing """
@torch.no_grad()
def test(X_test, edge_indices, edge_attr, model, installed_sensors, args, save_dir, out_f):

    """ Loading the trained model. """
    if args.model_path is None:
        args.model_path = save_dir+"/model_"+args.model+"_"+str(args.n_aggr)+"_"+str(args.n_hops)+"_"+str(datetime.date.today())+".pt"
    model_state = torch.load(args.model_path)
    model.load_state_dict(model_state["model"])

    model.eval()
    
    """ Initializing parameters and loading dataset and batches. """
    N, n_nodes, V = X_test.shape
    print("Number of sensors installed: ", len(installed_sensors))
    test_dataset, Y_test = load_dataset(X_test.clone(), n_nodes, installed_sensors, edge_indices, edge_attr)    
    del X_test    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    """" Testing and saving the results. """
    test_losses = []
    Y_hat = []
    for batch in test_loader:
        y, y_hat = model(batch)
        test_loss = model.loss(y, y_hat)
        test_losses.append(test_loss.detach().cpu().item())
        del y        
        y_hat = torch.hstack(y_hat.split(n_nodes)).view(n_nodes, -1, y_hat.shape[1])

        Y_hat.append(y_hat.detach().cpu())
        
        del y_hat, test_loss, batch
    Y_hat = torch.cat(Y_hat, dim=1).transpose(1,0)
    
    print("Test loss: ", np.round(np.mean(test_losses), 8))
    print("Test loss: ", np.round(np.mean(test_losses), 8), file=out_f)
    
    return Y_test, Y_hat, test_losses 

