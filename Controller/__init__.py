import imath as pt


class _EventBase():

    def __init__(self):
        pass

    def __call__(self, data_packages):
        assert False

    def state_dict(self):
        assert False

    def load_state_dict(self, state_dict):
        assert False

    def __and__(self, other):
        return EventAnd(self, other)

    def __or__(self, other):
        return EventOr(self, other)

class EventEpochEqualTo(_EventBase):

    def __init__(self, epoch):
        super(EventEpochEqualTo, self).__init__()

        self.epoch = epoch

    def __call__(self, data_package):
        value = data_package['epoch']
        if value == self.epoch:
            return True

        return False

    def state_dict(self):
        return {'epoch': self.epoch}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']


class EventEpochGreaterThan(_EventBase):

    def __init__(self, threshold, include_equal=False):
        super(EventEpochGreaterThan, self).__init__()

        self.threshold     = threshold
        self.include_equal = include_equal

    def __call__(self, data_package):
        value = data_package['epoch']
        if (self.include_equal and value >= self.threshold) or (not self.include_equal and value > self.threshold):
            return True

        return False

    def state_dict(self):
        return {'threshold':     self.threshold,
                'include_equal': self.include_equal}

    def load_state_dict(self, state_dict):
        self.threshold = state_dict['threshold']
        self.include_equal = state_dict['include_equal']

class EventEpochLessThan(_EventBase):

    def __init__(self, threshold, include_equal=False):
        super(EventEpochLessThan, self).__init__()

        self.threshold     = threshold
        self.include_equal = include_equal

    def __call__(self, data_package):
        value = data_package['epoch']
        if (self.include_equal and value <= self.threshold) or (not self.include_equal and value < self.threshold):
            return True

        return False

    def state_dict(self):
        return {'threshold':     self.threshold,
                'include_equal': self.include_equal}

    def load_state_dict(self, state_dict):
        self.threshold = state_dict['threshold']
        self.include_equal = state_dict['include_equal']

class EventEvalScoreGreaterThan(_EventBase):

    def __init__(self, threshold, include_equal=False):
        super(EventEvalScoreGreaterThan, self).__init__()

        self.threshold     = threshold
        self.include_equal = include_equal

    def __call__(self, data_package):
        value = data_package.get('eval_score')
        #  print('check eval value', value)
        if value is None:
            return False

        if (self.include_equal and value >= self.threshold) or (not self.include_equal and value > self.threshold):
            return True

        return False

    def state_dict(self):
        return {'threshold':     self.threshold,
                'include_equal': self.include_equal}

    def load_state_dict(self, state_dict):
        self.threshold = state_dict['threshold']
        self.include_equal = state_dict['include_equal']

class EventEvalScoreLessThan(_EventBase):

    def __init__(self, threshold, include_equal=False):
        super(EventEvalScoreLessThan, self).__init__()

        self.threshold     = threshold
        self.include_equal = include_equal

    def __call__(self, data_package):
        value = data_package.get('eval_score')
        if value is None:
            return False

        if (self.include_equal and value <= self.threshold) or (not self.include_equal and value < self.threshold):
            return True

        return False

    def state_dict(self):
        return {'threshold':     self.threshold,
                'include_equal': self.include_equal}

    def load_state_dict(self, state_dict):
        self.threshold = state_dict['threshold']
        self.include_equal = state_dict['include_equal']

class EventStepEqualTo(_EventBase):

    def __init__(self, step):
        super(EventStepEqualTo, self).__init__()

        self.step = step

    def __call__(self, data_package):
        value = data_package['step']
        if value == self.epoch:
            return True

        return False

    def state_dict(self):
        return {'step': self.step}

    def load_state_dict(self, state_dict):
        self.step = state_dict['step']

class EventStepGreaterThan(_EventBase):

    def __init__(self, threshold, include_equal=False):
        super(EventStepGreaterThan, self).__init__()

        self.threshold     = threshold
        self.include_equal = include_equal

    def __call__(self, data_package):
        value = data_package['step']
        #  print(f"check step {value}")
        if (self.include_equal and value >= self.threshold) or (not self.include_equal and value > self.threshold):
            return True

        return False

    def state_dict(self):
        return {'threshold':     self.threshold,
                'include_equal': self.include_equal}

    def load_state_dict(self, state_dict):
        self.threshold = state_dict['threshold']
        self.include_equal = state_dict['include_equal']

class EventStepLessThan(_EventBase):

    def __init__(self, threshold, include_equal=False):
        super(EventStepLessThan, self).__init__()

        self.threshold     = threshold
        self.include_equal = include_equal

    def __call__(self, data_package):
        value = data_package['step']
        if (self.include_equal and value <= self.threshold) or (not self.include_equal and value < self.threshold):
            return True

        return False

    def state_dict(self):
        return {'threshold':     self.threshold,
                'include_equal': self.include_equal}

    def load_state_dict(self, state_dict):
        self.threshold = state_dict['threshold']
        self.include_equal = state_dict['include_equal']

class EventValueGreaterThan(_EventBase):

    def __init__(self, value_name, threshold, include_equal=False):
        super(EventValueGreaterThan, self).__init__()

        self.value_name    = value_name
        self.threshold     = threshold
        self.include_equal = include_equal

    def __call__(self, data_package):
        value = data_package['ret_train'][self.value_name]
        if (self.include_equal and value >= self.threshold) or (not self.include_equal and value > self.threshold):
            return True

        return False

    def state_dict(self):
        return {'value_name':    self.value_name,
                'threshold':     self.threshold,
                'include_equal': self.include_equal}

    def load_state_dict(self, state_dict):
        self.value_name = state_dict['value_name']
        self.threshold = state_dict['threshold']
        self.include_equal = state_dict['include_equal']


class EventValueLessThan(_EventBase):

    def __init__(self, value_name, threshold, include_equal=False):
        super(EventValueLessThan, self).__init__()

        self.value_name    = value_name
        self.threshold     = threshold
        self.include_equal = include_equal

    def __call__(self, data_package):
        value = data_package['ret_train'][self.value_name]
        if (self.include_equal and value <= self.threshold) or (not self.include_equal and value < self.threshold):
            return True

        return False

    def state_dict(self):
        return {'value_name':    self.value_name,
                'threshold':     self.threshold,
                'include_equal': self.include_equal}

    def load_state_dict(self, state_dict):
        self.value_name = state_dict['value_name']
        self.threshold = state_dict['threshold']
        self.include_equal = state_dict['include_equal']

class EventValueChangeInRange(_EventBase):

    def __init__(self, value_name, step_num, std):
        super(EventValueChangeInRange, self).__init__()

        self.step_num = step_num
        self.std = std
        self.values = []

    def __call__(self, data_package):
        value = data_package['ret_train'][self.value_name]

        self.values.append(value)
        if len(self.values) > self.step_num:
            self.values.pop(0)
            array = np.array(self.values)

            std = array.std()
            if std < self.std:
                return True
        else:
            return False

class EventAnd(_EventBase):

    def __init__(self, *events):
        super(EventAnd, self).__init__()

        for e in events:
            assert issubclass(type(e), _EventBase), "所有元素都必须是事件类的子类"

        self.events = events

    def __call__(self, data_package):
        for e in self.events:
            if not e(data_package):
                return False

        return True

    def state_dict(self):
        state_dict = {}
        for i, e in self.events:
            state_dict[i] = e.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        for i, e in enumerate(self.events):
            e.load_state_dict(state_dict[i])


class EventOr(_EventBase):

    def __init__(self, *events):
        super(EventOr, self).__init__()

        for e in events:
            assert issubclass(type(e), _EventBase), "所有元素都必须是事件类的子类"

        self.events = events

    def __call__(self, data_package):
        for e in self.events:
            if e(data_package):
                return True

        return False

    def state_dict(self):
        state_dict = {}
        for i, e in enumerate(self.events):
            state_dict[i] = e.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        for i, e in enumerate(self.events):
            e.load_state_dict(state_dict[i])


class _ActionBase():

    def __init__(self):
        pass

    def __call__(self):
        assert False

    def state_dict(self):
        assert False

    def load_state_dict(self, state_dict):
        assert False


class ActionChangeLrByFixedValue(_ActionBase):

    def __init__(self, optimizer, lr):
        super(ActionChangeLrByFixedValue, self).__init__()

        self.optimizer = optimizer
        self.lr        = lr

    def __call__(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

        #  print(f"改变学习率: {self.lr}")

    def state_dict(self):
        return {'lr': self.lr}

    def load_state_dict(self, state_dict):
        self.lr = state_dict['lr']


class ActionSequence(_ActionBase):

    def __init__(self, actions):
        super(ActionSequence, self).__init__()

        self.actions_sequence = actions

    def __call__(self):
        for a in self.actions_sequence:
            a()

    def state_dict(self):
        state_dict = {}
        for i, a in enumerate(self.actions_sequence):
            state_dict[i] = a.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        for i, a in enumerate(self.actions_sequence):
            a.load_state_dict(state_dict[i])


class Controller():

    def __init__(self, event, action, infinite_trigger=False):
        self.event            = event
        self.action           = action

        self.enable           = True
        self.infinite_trigger = infinite_trigger

    def __call__(self):
        '''
        运行成功则返回True
        '''
        #  print(self.enable)
        if self.enable:
            self.action()

            if not self.infinite_trigger:
                self.enable = False

            return True
        else:
            return False

    def state_dict(self):
        state_dict = {
                'event': self.event.state_dict(),
                'action': self.action.state_dict(),
                'enable': self.enable,
                'infinite_trigger': self.infinite_trigger
                }

        return state_dict

    def load_state_dict(self, state_dict):
        self.event.load_state_dict(state_dict['event'])
        self.action.load_state_dict(state_dict['action'])
        self.enable = state_dict['enable']
        self.infinite_trigger = state_dict['infinite_trigger']

class ControllerSet(Controller):

    def __init__(self, *controllers):
        super(ControllerSet, self).__init__(None, None)

        self.controllers = list(controllers)

    def __call__(self):
        assert False

    def state_dict(self):
        ret = {}
        for i, c in enumerate(self.controllers):
            ret[i] = c.state_dict()

        return ret

    def load_state_dict(self, state_dict):
        for i, c in enumerate(self.controllers):
            c.load_state_dict(state_dict[i])

from imath.Controller.ExtendedController import *
