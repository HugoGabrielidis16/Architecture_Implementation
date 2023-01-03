import torch


class Trainer:
    def __init__(
        self,
        model,
        trainloader,
        testloader,
        reconstruction_loss,
        kl_loss,
        criterion,
        device,
        total_epochs,
    ):

        self.model = model
        self.train_loader = trainloader
        self.test_loader = testloader
        self.reconstruction_loss = reconstruction_loss
        self.kl_loss = kl_loss
        self.criterion = criterion
        self.device = device
        self.total_epochs = total_epochs

        self.train_loss_dict = {
            "reconstruction_loss": [],
            "combined_loss": [],
            "kl_loss": [],
        }
        self.test_loss_dict = {
            "reconstruction_loss": [],
            "combined_loss": [],
            "kl_loss": [],
        }

    def train_one_step(self, epoch):
        self.model.to(self.device)
        self.model.train()
        for batch_id, data in enumerate(self.train_loader):
            x = data[0].to(self.device)
            pred_img, pred_mean, pred_var = self.model(x)
            kl_loss = self.kl_loss(pred_mean, pred_var)
            reconstruction_loss = self.reconstruction_loss(pred_img, x)

            loss = kl_loss + reconstruction_loss
            self.optimizer.zero_grad()
            loss.backward()

            self.train_loss_dict["reconstruction_loss"].append(
                reconstruction_loss.item()
            )
            self.train_loss_dict["kl_loss"].append(kl_loss.item())
            self.test_loss_dict["combined_loss"].append(loss.item())

            if batch_id % 30 == 0:
                print(
                    "Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f"
                    % (
                        epoch + 1,
                        self.total_epochs,
                        batch_id,
                        len(self.train_loader),
                        loss,
                    )
                )

    def test_one_step(self, epoch):
        self.model.to(self.device)
        self.model.train()
        for batch_id, data in enumerate(self.test_loader):
            x = data.to(self.device)
            pred_img, pred_mean, pred_var = self.model(x)
            kl_loss = self.kl_loss(pred_mean, pred_var)
            reconstruction_loss = self.reconstruction_loss(pred_img, x)

            loss = kl_loss + reconstruction_loss
            self.test_loss_dict["reconstruction_loss"].append(
                reconstruction_loss.item()
            )
            self.test_loss_dict["kl_loss"].append(kl_loss.item())
            self.test_loss_dict["combined_loss"].append(loss.item())

            if batch_id % 30 == 0:
                print(
                    "Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f"
                    % (
                        epoch + 1,
                        self.total_epochs,
                        batch_id,
                        len(self.test_loader),
                        loss,
                    )
                )

    def fit(self):
        for epoch in range(self.total_epochs):
            print(f"Epoch : {epoch}/{self.total_epochs}")
            self.test_one_step(epoch=epoch)
            self.test_one_step(epoch=epoch)
