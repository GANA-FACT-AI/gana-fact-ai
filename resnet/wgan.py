from resnet.generator import Generator
from resnet.critic import Critic
import model.wgan


class WGAN(model.wgan.WGAN):
    def __init__(self, k, log_fn, blocks, random_swap):
        super().__init__(k, log_fn, random_swap)
        self.generator = Generator(blocks, random_swap)
        self.critic = Critic(k)
