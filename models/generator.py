import pandas
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(nn.Linear(1, 3),
                                   nn.Sigmoid(),
                                   nn.Linear(3, 4),
                                   nn.Sigmoid()
                                   )

        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, D, inputs, targets):
        g_output = self.forward(inputs)
        d_output = D.forward(g_output)
        loss = D.loss_function(d_output, targets)
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        # Gradienten nullen, einen Rückwärts-Pass ausführen, Gewichte aktualisieren self.optimiser.zero_grad() loss.backward() self.optimiser.step()
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.',
                grid=True, yticks=(0, 0.25, 0.5))
        plt.show()
