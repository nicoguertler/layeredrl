from typing import Optional
from pathlib import Path

from gymnasium import Env
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


class VideoLogger:
    """Class for logging video clips of rollouts."""

    def __init__(
        self,
        log_dir: str,
        fps: int = 10,
        max_episodes: int = 10,
        camera_name: Optional[str] = None,
    ):
        """Initialize the logger.

        Args:
            log_dir: The directory to save the videos.
            fps: The frames per second of the video.
        """
        self.log_dir = log_dir
        self.fps = fps
        self.frames = []
        self.max_episodes = max_episodes
        self.camera_name = camera_name
        self._episode_counter = 0

    def log_frame(self, env: Env, terminated: bool, truncated: bool):
        """Log a frame from the environment.

        Requires the environments to be in render mode 'rbg_array'.

        Args:
            env: The environment.
            terminated: Whether the episode is terminated.
            truncated: Whether the episode is truncated.
        """
        if self._episode_counter < self.max_episodes:
            unwrapped = env.envs[0].unwrapped
            if self.camera_name is not None:
                env = unwrapped
                camera_name = self.camera_name
            # requires environment to have a bird view camera
            elif hasattr(unwrapped, "ant_env"):
                env = unwrapped.ant_env
                camera_name = "bird"
            else:
                env = unwrapped
                camera_name = None
            # render to make sure all markers are updated etc.
            render = env.render()
            # render again with the right camera
            render = env.mujoco_renderer.render(
                render_mode="rgb_array", camera_name=camera_name
            )
            assert render is not None, "Environment must be in render mode 'rgb_array'."
            self.frames.append(render)
            if terminated[0] or truncated[0]:
                self._episode_counter += 1
                print("Recorded an episode on video.")
        if self._episode_counter == self.max_episodes:
            self.save_video()
            self.frames = []
            self._episode_counter += 1

    def save_video(self):
        """Save video to disk."""
        path = self.log_dir / Path("rollouts.mp4")
        print(f"Saving video to {path}")
        clip = ImageSequenceClip(self.frames, fps=self.fps)
        clip.write_videofile(str(path))
