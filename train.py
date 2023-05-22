import pytorch_lightning as pl
import torch


from torch_geometric.data import LightningDataset
from torch_geometric import seed_everything
import argparse

import os
from torch_geometric.data import Dataset
from model import Model

class PPMDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):

        self.data_dir = root
        self.total_files =  os.listdir(root)
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return self.total_files

    @property
    def num_classes(self) -> int:
        return 1

    def process(self):
        pass

    def len(self):
        return len(self.processed_file_names) - 1

    def get(self, idx):
        data = torch.load(os.path.join(self.data_dir, f'data_{idx}.pt'))
        return data





def main(model_type, epoch):
    seed_everything(42)
    dataset = PPMDataset(root="data/PT/")
  
    dataset = dataset.shuffle()

    train_dataset = dataset[:7355]
    val_dataset = dataset[7355:8930]
    test_dataset = dataset[8930:]

    datamodule = LightningDataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size= 1,
    )

    model = Model(        
        num_node_features=32,
        gnn_hidden=512,
        fc_hidden=512,
        reduce_func="sum",
        norm_sf=False,
        model_type=model_type,
        )

    devices = torch.cuda.device_count()
    strategy = pl.strategies.DDPSpawnStrategy(find_unused_parameters=False) # type: ignore
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min') # type: ignore
    trainer = pl.Trainer(strategy=strategy, accelerator='gpu', devices=devices,
                         max_epochs=epoch, callbacks=[checkpoint])

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

    # trainer.test(model, ckpt_path='/root/work/dippm/others/epoch=487-step=3589240.ckpt', datamodule=datamodule)
    

if __name__ == '__main__':
    # get arguments epoch and model type using argparse
    args = argparse.ArgumentParser()
    args.add_argument("--epoch", type=int, default=10)
    args.add_argument("--model_type", type=str, default="GraphSAGE")
    args = args.parse_args()
    main(model_type=args.model_type, epoch=args.epoch)