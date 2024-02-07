import torch
import torch.nn.functional as F
from torch.func import hessian, jvp, grad
from torch.utils.benchmark import Timer
from functools import partial

_ = torch.manual_seed(0)

# ===== NOTE
#       we assume Hessian matrix is a nxn matrix which means the loss function must return a SCALAR
#       so that we can use `torch.func.grad` function
def predict(weight, bias, x):
    # F.linear performs y=xA^T+b, so x should be (bsize, feat_dim) or (feat_dim)
    rtn = F.linear(x, weight, bias).tanh()
    assert rtn.numel() == 1 and rtn.ndim == 1
    return rtn[0]

m = 1  # output dimension
n = 1024  # input dimension
weight = torch.randn(m, n) / n**0.5
bias = torch.randn(m) / n**0.5
x = torch.randn(n) / n**0.5  # feature vector
y = predict(weight, bias, x)  # (m)
print(f"y.shape={y.shape}")

v = torch.randn(n)

hess_api = hessian(predict, argnums=2)(weight, bias, x)
print(f"hess_api.shape={hess_api.shape}")  # shape of hessian, (n, n)

def explicit_hessian_hvp(f, primals, tangents):
    # primals: x
    # tangents: v
    hess_api = hessian(f)(primals)  # (m, n, n) or (n, n)
    return hess_api @ tangents  # (m, n) or (n, )

def hvp(f, primals, tangents):
    # primals: x
    # tangents: v
    return jvp(grad(f), primals, tangents)[1]  # (n, )

hvp_out1 = explicit_hessian_hvp(partial(predict, weight, bias), x, v)  # (n, )
hvp_out2 = hvp(partial(predict, weight, bias), (x,), (v,))  # (n, )
assert torch.allclose(hvp_out1, hvp_out2)

# ===== profile times
with_hessian_hvp = Timer(stmt="explicit_hessian_hvp(partial(predict, weight, bias), x, v)", globals=globals())
efficient_hvp = Timer(stmt="hvp(partial(predict, weight, bias), (x,), (v,))", globals=globals())

profile_num = 3000
explicit_hessian_hvp_timer = with_hessian_hvp.timeit(profile_num)
efficient_hvp_timer = efficient_hvp.timeit(profile_num)

def get_perf(first, first_descriptor, second, second_descriptor):
    """takes torch.benchmark objects and compares delta of second vs first."""
    faster = second.times[0]
    slower = first.times[0]
    gain = (slower - faster) / slower
    if gain < 0:
        gain *= -1
    final_gain = gain * 100
    print(f" Performance delta: {final_gain:.4f} percent improvement with {second_descriptor} ")

print(explicit_hessian_hvp_timer)
print(efficient_hvp_timer)
get_perf(explicit_hessian_hvp_timer, "explicit_hessian_hvp_timer", efficient_hvp_timer, "efficient_hvp_timer")