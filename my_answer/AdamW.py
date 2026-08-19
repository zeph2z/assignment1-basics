import torch, math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float, betas: tuple[float, float], eps: float):

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }

        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            b1, b2 = group["betas"]
            eps = group["eps"]

            for param in group["params"]:
                if "t" not in self.state[param]:
                    self.state[param]["m"] = torch.zeros_like(param)
                    self.state[param]["v"] = torch.zeros_like(param)
                    self.state[param]["t"] = 1

                m = self.state[param]["m"]
                t = self.state[param]["t"]
                v = self.state[param]["v"]

                lr_t = lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
                param.data *= 1 - lr * wd
                m = b1 * m + (1 - b1) * param.grad
                v = b2 * v + (1 - b2) * param.grad ** 2
                param.data -= lr_t * m / (torch.sqrt(v) + eps)

                self.state[param]["m"] = m
                self.state[param]["v"] = v
                self.state[param]["t"] += 1