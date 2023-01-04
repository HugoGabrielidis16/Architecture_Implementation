import torch
import matplotlib.pyplot as plt


class Trainer:
    def __init__(
        self,
        model,
        trainloader,
        testloader,
        reconstruction_loss,
        kl_loss,
        criterion,
        total_epochs,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
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
        # self.model.to(self.device)
        self.model.train()
        for batch_id, data in enumerate(self.train_loader):
            x = data.to(self.device)

            pred_img, pred_mean, pred_var = self.model(x)
            print(pred_img[0])
            kl_loss = self.kl_loss(pred_mean, pred_var)
            reconstruction_loss = self.reconstruction_loss(pred_img, x)

            loss = kl_loss + reconstruction_loss
            self.criterion.zero_grad()
            loss.backward()

            self.train_loss_dict["reconstruction_loss"].append(
                reconstruction_loss.item()
            )
            self.train_loss_dict["kl_loss"].append(kl_loss.item())
            self.test_loss_dict["combined_loss"].append(loss.item())

            if batch_id % 30 == 0:

                print(
                    "Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f | Reconstruction Loss: %.4f | KL Loss: %.4f"
                    % (
                        epoch + 1,
                        self.total_epochs,
                        batch_id,
                        len(self.test_loader),
                        loss,
                        reconstruction_loss,
                        kl_loss,
                    )
                )
            if batch_id % 50 == 0:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(x[0].permute((1, 2, 0)).cpu().detach().numpy(), cmap="turbo")
                ax2.imshow(
                    pred_img[0].permute((1, 2, 0)).cpu().detach().numpy(), cmap="turbo"
                )
                plt.show()

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
                    "Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f | Reconstruction Loss: %.4f | KL Loss: %.4f"
                    % (
                        epoch + 1,
                        self.total_epochs,
                        batch_id,
                        len(self.test_loader),
                        loss,
                        reconstruction_loss,
                        kl_loss,
                    )
                )

    def fit(self):
        for epoch in range(self.total_epochs):
            print(f"Epoch : {epoch}/{self.total_epochs}")
            self.train_one_step(epoch=epoch)
            self.test_one_step(epoch=epoch)
