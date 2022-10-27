# Autoencoder
Simple denoising autoencoder (DAE) written in python, using the pytorch framework. Utilizing a convolutional neural network to minimize loss. 

Deep learning just got... derp

# Requirements
```
pip install -r requirements.txt
```
# What's going on
After importing the pytorch modules we downlad the dataset supplied by torchvision, transform the pictures into tensors and iterate through the data.

```python
transform = transforms.ToTensor()

mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)

dataiter = iter(data_loader)
images, labels = dataiter.next()
print(torch.min(images), torch.max(images))
```
We then create the autoencoder, making sure the kernel size, stride and padding are in line, so the number of channels in the input image are equal to the number of channels produced by the convolution. Read the documentation at https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16 , 3, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
           
        )
```
We create the model, set the criteria, and use the torch.optim package by constructing the object 'optimizer'. This object holds the current state, and will update the parameters based on the computed gradients. We utilize the Adam optimization algorithm, since it's gradient based. Documentation for torch.optim.Adam: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

```python
model = Autoencoder()
criteria = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
```
The rest of the code is self-explanatory. We set a for loop to complete 8 passes of the entire dataset, and plot every outher result.
# Output
This test was run in VSCode. The model trains against data provided by torchvision, completing 8 passes of the entire dataset before plotting the results after every other pass of the dataset. 

<img width="635" alt="Screenshot 2022-10-27 at 21 16 47" src="https://user-images.githubusercontent.com/42718681/198379039-fd8f57ea-bd27-48d6-83c5-71126b85482a.png">


