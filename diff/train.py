import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm, trange

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--data", type=str, help="Path to the data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timesteps", type=int, default=50, help="Timesteps")
    return parser.parse_args()

def main():
    # load arguments
    args = parse_args()

    # load data
    # TODO: implement data loading
    data = None

    # load model
    # TODO: implement model loading
    model = None

    # set random seed
    torch.manual_seed(args.seed)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # schedule learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    
    # training
    pbar = tqdm(range(args.epochs), desc="Epochs")

    for epoch in pbar:
        mean_loss = 0.0
        
        for num_iter, batch in enumerate(tqdm(data, desc="Batches", leave=False)):
            x_0 = ...              # batch ... .to(device)
            bs = x_0.size(0)       # batch_size of the batch 
            t = ...   # t ~ Uniform({1, 2, ..., timesteps})
            
            random_noise = ... # € ~ N(0, 1)
            alpha = ... # TODO: implement alpha
            x_t = ... # sqrt(alpha) * x_0 + sqrt(1 - alpha) * random_noise

            pred = ... # model(x_0, x_t)
            loss = F.l1_loss(pred, x_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss += loss.item()
            pbar.set_postfix_str('Loss: {:.4f}'.format(mean_loss/len(data)))

        scheduler.step()

    # save model
    torch.save(model.state_dict(), "model.pth")

if __name__ == '__main__':
    main()