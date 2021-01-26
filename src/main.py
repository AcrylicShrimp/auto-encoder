
import model

from matplotlib import pyplot as plt
from numpy.lib.function_base import average
import torch
from torch import nn
from torch import optim
from torch.utils.data import dataloader
from torchvision import datasets
from torchvision import transforms

torch.set_num_threads(16)
print(torch.__config__.show())

mnist_train = datasets.MNIST(
    './data', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(
    './data', train=False, transform=transforms.ToTensor(), download=True)

mnist_train_loader = dataloader.DataLoader(
    mnist_train, batch_size=32, shuffle=True, drop_last=True)
mnist_test_loader = dataloader.DataLoader(
    mnist_test, batch_size=1, shuffle=True, drop_last=True)

encoder = model.Encoder()
decoder = model.Decoder()

# encoder.load_state_dict(torch.load('./save/encoder-0.8674.pt'))
# decoder.load_state_dict(torch.load('./save/decoder-0.8674.pt'))

parameters = list(encoder.parameters()) + list(decoder.parameters())

loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(parameters, lr=0.001)

plt.ion()

while True:
    encoder.train()
    decoder.train()
    running_loss = []

    for index, [image, _] in enumerate(mnist_train_loader):
        optimizer.zero_grad()

        noise = torch.randn_like(image) * 0.1 ** 0.5
        output = encoder(image + noise)

        output = decoder(output)

        loss = loss_func(output, image)
        loss.backward()

        optimizer.step()

        running_loss.append(loss.item() * 32)

    loss = average(running_loss)
    print('train loss: {}'.format(loss))

    _, [image, _] = next(enumerate(mnist_test_loader))
    encoder.eval()
    decoder.eval()
    output = encoder(image)
    output = decoder(output)
    plt.imshow(output.detach()[0].permute(1, 2, 0))
    plt.show()
    plt.pause(.001)

    torch.save(encoder.state_dict(), './save/encoder-{:.4f}.pt'.format(loss))
    torch.save(decoder.state_dict(), './save/decoder-{:.4f}.pt'.format(loss))
