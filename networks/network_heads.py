from .network_utils import *
from .network_bodies import *


class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 device='cpu'):
        super(DeterministicActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params)
        self.device = device
        self.to(self.device)

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.phi_body(obs)

    def actor(self, phi):
        return torch.tanh(self.fc_action(self.actor_body(phi)))

    def critic(self, phi, a):
        return self.fc_critic(self.critic_body(torch.cat([phi, a], dim=1)))
