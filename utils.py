import torch
import random
from trainer import train_and_evaluate
from model import SimpleNet
from optimizers import get_optimizer



def add_noise_to_data(data_loader, noise_factor=0.3, noise_type='gaussian', random_noise=False):
    noisy_loader = []
    for images, labels in data_loader:
        current_noise_factor = noise_factor * torch.rand(1).item() if random_noise else noise_factor
        
        if noise_type == 'gaussian':
            noisy_images = images + current_noise_factor * torch.randn(*images.shape).to(images.device)
        elif noise_type == 'salt_pepper':
            noisy_images = images.clone()
            probs = torch.rand(images.shape).to(images.device)
            noisy_images[probs < current_noise_factor / 2] = 0 
            noisy_images[probs > 1 - current_noise_factor / 2] = 1
        else:
            raise ValueError("Unsupported noise_type. Choose 'gaussian' or 'salt_pepper'.")
        
        noisy_images = torch.clamp(noisy_images, 0., 1.)
        noisy_loader.append((noisy_images, labels))

    return noisy_loader


def learning_rate_search(train_loader, test_loader, device, epochs=3, optimizers=['SGD', 'Momentum', 'Adam', 'RMSprop'], lr_list=[0.1, 0.01, 0.001, 0.0001]):
    best_params = {}
    for name in optimizers:
        print(f'Learning Rate Search for {name}...')

        best_accuracy = 0
        best_lr = None
        for lr in lr_list:
            model = SimpleNet().to(device=device)
            optimizer, scheduler = get_optimizer(model=model, optimizer_type=name, lr=lr)
            
            _, accuracy, _, _ = train_and_evaluate(model=model, optimizer=optimizer, scheduler=scheduler, train_loader=train_loader, test_loader=test_loader, device=device, epochs=epochs, show_plot=False)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_lr = lr
        
        best_params[name] = {'optimizer': optimizer, 'scheduler': scheduler, 'best_lr': best_lr}
        print(f'The Best Learning Rate for {name} is {best_lr}\n')
    return best_params
