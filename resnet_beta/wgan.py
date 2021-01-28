from resnet_beta.critic import Critic
from resnet_beta.generator import Generator
import model.wgan


class WGAN(model.wgan.WGAN):
    def __init__(self, k, log_fn, blocks):
        super().__init__(k, log_fn)
        self.generator = Generator(blocks)
        self.critic = Critic(k)
