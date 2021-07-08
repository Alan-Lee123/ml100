import torch
from torch.optim import Optimizer

class CustomAdam(Optimizer):
    def __init__(self, parameters, lr, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(CustomAdam, self).__init__(parameters, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        params = []
        grads = []
        states = []

        if 'step' not in self.state:
            self.state['step'] = 1
        else:
            self.state['step'] += 1
        step = self.state['step']

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if 'exp_avg_momentum' not in state:
                        state['exp_avg_momentum'] = torch.zeros_like(p)
                    if 'exp_avg_momentum_square' not in state:
                        state['exp_avg_momentum_square'] = torch.zeros_like(p)
                    states.append(state)
        
            for k in range(len(params)):
                param = params[k]
                grad = grads[k]
                state = states[k]
                m = state['exp_avg_momentum']
                v = state['exp_avg_momentum_square']

                m_hat = m * beta1 + (1 - beta1) * grad
                v_hat = v * beta2 + (1 - beta2) * (grad ** 2)
                m_hat_adjusted = m_hat / (1 - beta1 ** step)
                v_hat_adjusted = v_hat / (1 - beta2 ** step)
                param_update = -lr * m_hat_adjusted / (torch.sqrt(v_hat_adjusted) + eps)
                param.data.add_(param_update)
                state['exp_avg_momentum'] = m_hat
                state['exp_avg_momentum_square'] = v_hat
        
        return loss
