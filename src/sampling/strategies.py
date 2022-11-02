from typing import Tuple, List

import PIL.Image
import torch
from tqdm import trange

from src.data_utils.utils import tensor_to_image


@torch.no_grad()
def ddpm(
    encoder,
    im_size: int = 64,
    n_samples: int = 1,
    n_diffusion_steps: int = 1000,
    noise_range: Tuple[float] = (0.0001, 0.04),
    use_gpu: bool = False,
    seed: int = 57,
) -> List[PIL.Image.Image]:
    def p_xt(xt, noise, t):
        alpha_t = alpha[t].reshape(-1, 1, 1, 1)
        alpha_bar_t = alpha_bar[t].reshape(-1, 1, 1, 1)
        eps_coef = (1 - alpha_t) / (1 - alpha_bar_t) ** 0.5
        mean = 1 / (alpha_t ** 0.5) * (xt - eps_coef * noise)  # Note minus sign
        var = beta[t].reshape(-1, 1, 1, 1)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** 0.5) * eps

    device = torch.device("cuda") if use_gpu else torch.device("cpu")
    torch.manual_seed(seed)

    x = torch.randn(n_samples, 3, im_size, im_size).to(device)
    beta = torch.linspace(*noise_range, n_diffusion_steps).to(device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    for i in trange(n_diffusion_steps, desc="Sampling..."):
        t = torch.ones(n_samples, dtype=torch.long, device=device) * (n_diffusion_steps - i - 1)
        pred_noise = encoder(x, t)
        x = p_xt(x, pred_noise, t.unsqueeze(0))

    ims = [tensor_to_image(t.cpu()) for t in x]

    return ims
