from gymnasium import Wrapper
import numpy as np
import quaternion


class AntFlippedWrapper(Wrapper):
    """Wrapper around Ant Maze environment that terminates when the Ant has flipped over."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        ant_obs = observation["observation"]
        torso_orientation = quaternion.as_quat_array(ant_obs[1:5])
        z_dir = np.quaternion(0, 0, 0, 1)
        rotated_z_dir = torso_orientation * z_dir * torso_orientation.conjugate()
        terminated = quaternion.as_float_array(rotated_z_dir)[-1] < 0 or terminated

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Pass through reset unchanged
        return self.env.reset(**kwargs)


class AntNoWallFlippedWrapper(Wrapper):
    """Wrapper around Ant environment that terminates when the Ant has flipped over."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        ant_obs = observation
        torso_orientation = quaternion.as_quat_array(ant_obs[3:7])
        z_dir = np.quaternion(0, 0, 0, 1)
        rotated_z_dir = torso_orientation * z_dir * torso_orientation.conjugate()
        terminated = quaternion.as_float_array(rotated_z_dir)[-1] < 0 or terminated

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Pass through reset unchanged
        return self.env.reset(**kwargs)
