import os
from typing import Optional

import dm_env
import imageio
from acme.utils.paths import process_path
from acme.wrappers import EnvironmentWrapper
from dm_control.rl.control import Environment


class VideoWrapper(EnvironmentWrapper):
    def __init__(
        self,
        env: Environment,
        path: str = "~/chain_control",
        filename: str = "",
        file_extension: str = ".mp4",
        fps: int = 25,
        record_every: int = 1,
        camera_id: Optional[str] = None,
        width: int = 320,
        height: int = 240,
        add_uid_to_path: bool = True,
    ):
        """Wraps an Environment and records a video.

        Args:
            environment (dm_env.Environment): Environment to be wrapped.
            fps (int, optional): Frame rate of video. Defaults to 25.
            record_every (int, optional): Record every `record_every` episodes.
                Defaults to 1.
            camera_id (str, optional): The Camera-ID used when recoding.
                Defaults to the first camera.
            width (int, optional): Pixel-count in width. Defaults to 320.
            height (int, optional): Pixel-count in heigth. Defaults to 240.
        """
        super().__init__(env)
        control_rate = int(1 / env.control_timestep())
        assert control_rate > fps
        assert control_rate % fps == 0

        self._frames = []
        self._record_every = record_every
        self._camera_id = camera_id
        self._number_of_episodes = 0
        self._record_frame_every = int(control_rate / fps)
        self._env_step = 0
        self._fps = fps
        self._path = process_path(path, "videos", add_uid=add_uid_to_path)
        self._filename = filename
        self._file_extension = (
            file_extension if file_extension[0] == "." else "." + file_extension
        )
        self._width = width
        self._height = height

    def step(self, action) -> dm_env.TimeStep:
        record_episode = self._number_of_episodes % self._record_every == 0
        record_step = self._env_step % self._record_frame_every == 0
        self._env_step += 1

        if record_episode and record_step:
            frame = self._render_frame()
            self._frames.append(frame)

        ts = super().step(action)

        if self._reset_next_step:
            if len(self._frames) > 1:
                self._write_frames()

        return ts

    def _render_frame(self):
        if self._camera_id:
            frame = self._environment.physics.render(
                camera_id=self._camera_id, width=self._width, height=self._height
            )
        else:
            frame = self._environment.physics.render(
                width=self._width, height=self._height
            )
        return frame

    def _write_frames(self):
        path_with_extension = os.path.join(
            self._path,
            f"{self._filename}_{self._number_of_episodes:04d}" + self._file_extension,
        )

        imageio.mimwrite(path_with_extension, self._frames, fps=self._fps)
        self._frames = []

    def reset(self):
        self._number_of_episodes += 1
        return self.environment.reset()

    def close(self):
        self._frames = []
        self.environment.close()
