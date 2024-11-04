import torch.optim as optim

def get_all_optimizers(model, lr=0.01):
    # Define each optimizer and scheduler separately
    sgd_optimizer = optim.SGD(model.parameters(), lr=lr)
    momentum_optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    adam_optimizer = optim.Adam(model.parameters(), lr=lr)
    rmsprop_optimizer = optim.RMSprop(model.parameters(), lr=lr)
    
    # Create a dictionary with each optimizer and its scheduler
    optimizers = {
        'SGD': (sgd_optimizer, optim.lr_scheduler.StepLR(sgd_optimizer, step_size=10, gamma=0.1)),
        'Momentum': (momentum_optimizer, optim.lr_scheduler.StepLR(momentum_optimizer, step_size=10, gamma=0.1)),
        'Adam': (adam_optimizer, optim.lr_scheduler.ExponentialLR(adam_optimizer, gamma=0.9)),
        'RMSprop': (rmsprop_optimizer, optim.lr_scheduler.ExponentialLR(rmsprop_optimizer, gamma=0.9))
    }
    
    return optimizers


def get_optimizer(model, optimizer_type, lr=0.01):
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    elif optimizer_type == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    return optimizer, scheduler
