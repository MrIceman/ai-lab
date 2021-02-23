from datetime import datetime

import torch
from numpy import random

from models.discriminator import Discriminator
from models.generator import Generator


def generate_real_data() -> torch.float:
    real_data = torch.FloatTensor(
        [
            random.uniform(0.8, 1.0),
            random.uniform(0.0, 0.2),
            random.uniform(0.8, 1.0),
            random.uniform(0.0, 0.2),
        ])

    return real_data


def generate_random(size):
    random_data = torch.rand(size)
    return random_data


D = Discriminator()
G = Generator()

start = datetime.now()
for i in range(10000):
    D.train(generate_real_data(), torch.FloatTensor([1.0]))
    D.train(G.forward(torch.FloatTensor([0.5])).detach(), torch.FloatTensor([0.0]))
    G.train(D, torch.FloatTensor([0.5]), torch.FloatTensor([1.0]))
print(f'took {datetime.now().second - start.second} seconds')
