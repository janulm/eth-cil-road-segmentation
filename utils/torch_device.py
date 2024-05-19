import torch


# get the correct device for torch: either cpu, cuda gpu or for m-macs mps gpu 
def get_torch_device(allow_mps=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() and allow_mps else "cpu")
    return device
