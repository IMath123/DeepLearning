from imath.Controller import *

def StepLrController(optimizer, lr, steps, gamma):
    ret = []
    for step in [0] + steps:
        event = EventStepGreaterThan(step, True)
        action = ActionChangeLrByFixedValue(optimizer, lr)
        ret.append(Controller(event, action))
        lr = lr * gamma

    return ControllerSet(*ret)

def EpochLrController(optimizer, lr, epochs, gamma):
    ret = []
    for epoch in [0] + epochs:
        event = EventEpochGreaterThan(epoch, True)
        action = ActionChangeLrByFixedValue(optimizer, lr)
        ret.append(Controller(event, action))
        lr = lr * gamma

    return ControllerSet(*ret)
