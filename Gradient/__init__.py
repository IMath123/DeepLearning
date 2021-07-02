import torch

def GetGradientofInput(model, input, label, loss_function):
    input.requires_grad = True
    predict = model(input)
    loss = loss_function(predict, label)

    loss.backward()
    return input.grad