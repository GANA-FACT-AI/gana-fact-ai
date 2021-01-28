from resnet_alpha.generator import Generator
from resnet_alpha.critic import Critic
import model.wgan


class WGAN(model.wgan.WGAN):
    def __init__(self, k, log_fn, blocks):
        super().__init__(k, log_fn)
        self.generator = Generator(blocks)
        self.critic = Critic(k)
