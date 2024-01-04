import torch
import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        combined = torch.cat((input,hidden), 1)
        hidden = torch.tanh(self.i2h(combined))
        output = self.h2o(hidden)
        return output, hidden
    
    def get_hidden(self):
        return torch.zeros(1, self.hidden_size)
    


rnn_model = MyRNN(input_size=4, hidden_size=1024, output_size=2)
hidden = rnn_model.get_hidden()
