import numpy as np
from gym.envs.classic_control import pendulum

class PendulumEnv(pendulum.PendulumEnv):

    def __init__(self, task={}):
        self._task = task
        self._theta = task.get('theta', 0.0)
        super(PendulumEnv, self).__init__()

    def sample_tasks(self, num_tasks):
        theta = np.array([np.pi])
        thetas = self.np_random.uniform(low=-theta, high=theta, size=(num_tasks,))
        tasks = [{'theta': theta} for theta in thetas]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._theta = task['theta']

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = [self._theta,0.5]  #assume a fixed joint velocity
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta),np.sin(theta),thetadot], dtype=np.float32)

    def step(self,u):
        th, thdot = self.state # th := theta

        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = pendulum.angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])

        infos = dict(newth=newth,
            newthdot=newthdot, task=self._task)

        return (self._get_obs(), -costs, False, infos)
