'''
Examples of rule based agents
to do
abstract class and inherit
for now just some dumb classes
'''

class SampleAgentCP():

  def __int__(self, cp) -> None:
    self.cp = cp
    self.obs_cp_index = 2

  def get_action_vf(self, obs):
    if obs[self.obs_cp_index] >= self.cp:
      action = 0.
    else:
      action = 1.

    vf = None

    return action, vf


class SampleAgentPB():

  def __int__(self, c) -> None:
    self.c = c
    self.obs_c_index = 3

  def get_action_vf(self, obs):
    if obs[self.obs_c_index] >= self.cp:
      action = 0.
    else:
      action = 1.

    vf = None

    return action, vf