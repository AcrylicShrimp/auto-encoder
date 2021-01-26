

import model

from matplotlib import pyplot as plt
import torch

decoder = model.Decoder()
decoder.load_state_dict(torch.load('./save/decoder-0.8674.pt'))
decoder.eval()

while True:
    output = torch.rand(1, 3 ** 2)
    print(output)
    output = decoder(output)
    plt.imshow(output.detach()[0].permute(1, 2, 0))
    plt.show()
