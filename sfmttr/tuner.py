import torch
import pytorch_lightning as pl

class SfMTuner(pl.LightningModule):
    def __init__(self, model):
        super(SfMTuner, self).__init__()
        self.model = model
        
    def loss(self, y_pred, target, errors, coords, s):

        # Sample pred on points where we have COLMAP pseudo gt
        y_pred = torch.nn.functional.grid_sample(
            y_pred, coords, mode='nearest', align_corners=True
        )[:, 0, 0, :]

        # Loss weighted with the COLMAP error
        return torch.mean(torch.abs(y_pred - s * target) * torch.exp(-errors * errors))

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, target, errors, coords, s = batch

        if target.size(1) == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        # Run the model
        y_pred = self(x)

        # Colmap pseudo-gt loss
        loss = self.loss(y_pred, target, errors, coords, s)
        
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.get_encoder_params(), 1e-4)
