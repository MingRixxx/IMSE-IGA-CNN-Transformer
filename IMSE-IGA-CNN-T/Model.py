import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer

torch.manual_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EMDCNNTransformer(nn.Module):
    def __init__(self, batch_size, input_channels, conv_archs, output_dim, hidden_dim, num_layers, num_heads,dropout_rate=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.batch_size = batch_size
        self.conv_arch = conv_archs
        self.input_channels = input_channels
        self.cnn_features = self.make_layers()
        self.dropout = dropout_rate
        self.hidden_dim = hidden_dim
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(conv_archs[-1][-1], num_heads, hidden_dim, dropout=0.5, batch_first=True),
            num_layers
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(conv_archs[-1][-1], output_dim)

    def make_layers(self):
        layers = []
        for (num_convs, out_channels) in self.conv_arch:
            for _ in range(num_convs):
                layers.append(nn.Conv1d(self.input_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                self.input_channels = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, input_seq):
        input_seq = input_seq.view(self.batch_size, -1, 128)
        cnn_features = self.cnn_features(input_seq)
        cnn_features = cnn_features.permute(0, 2, 1)
        transformer_output = self.transformer(cnn_features)
        output_avgpool = self.avgpool(transformer_output.transpose(1, 2))
        output_avgpool = output_avgpool.reshape(self.batch_size, -1)
        output = self.classifier(output_avgpool)
        return output

def model_train(batch_size, epochs, model, optimizer, loss_function, train_loader, val_loader):
    model = model.to(device)
    train_size = len(train_loader) * batch_size
    val_size = len(val_loader) * batch_size
    best_accuracy = 0.0
    train_loss = []
    train_acc = []
    validate_acc = []
    validate_loss = []
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        loss_epoch = 0.0
        correct_epoch = 0
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(seq)
            probabilities = F.softmax(y_pred, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            correct_epoch += (predicted_labels == labels).sum().item()
            loss = loss_function(y_pred, labels)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

        train_Accuracy = correct_epoch / train_size
        train_loss.append(loss_epoch / train_size)
        train_acc.append(train_Accuracy)
        print(f'Epoch: {epoch + 1:2} train_Loss: {loss_epoch / train_size:10.8f} train_Accuracy:{train_Accuracy:4.4f}')

        with torch.no_grad():
            loss_validate = 0.0
            correct_validate = 0
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                pre = model(data)
                probabilities = F.softmax(pre, dim=1)
                predicted_labels = torch.argmax(probabilities, dim=1)
                correct_validate += (predicted_labels == label).sum().item()
                loss = loss_function(pre, label)
                loss_validate += loss.item()

            val_accuracy = correct_validate / val_size
            print(f'Epoch: {epoch + 1:2} val_Loss:{loss_validate / val_size:10.8f},  validate_Acc:{val_accuracy:4.4f}')
            validate_loss.append(loss_validate / val_size)
            validate_acc.append(val_accuracy)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    print("best_accuracy :", best_accuracy)
    return validate_acc, validate_loss