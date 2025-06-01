from burgers_load import get_burgers_dataloader
from darcy_load import get_darcy_dataloader
loader, _, _, _= get_darcy_dataloader(43, 16)
sample = next(iter(loader))
print(sample['a'].shape)
print(sample['u'].shape)

