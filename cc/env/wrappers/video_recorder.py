import os
import pathlib
from subprocess import DEVNULL, STDOUT, call
from typing import Optional

import dm_env
import numpy as np
from acme.wrappers import EnvironmentWrapper
from PIL import Image


class RecordVideoWrapper(EnvironmentWrapper):
    def __init__(
        self,
        environment: dm_env.Environment,
        fps: int = 25,
        record_every: int = 1,
        control_rate: int = 100,
        camera_id: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        video_name: Optional[str] = None,
        path_to_folder: str = "./videos",
        cleanup_imgs: bool = False,
    ):
        """Wraps a `dm_env.Environment` and records a video.

        Args:
            environment (dm_env.Environment): Environment to be wrapped.
            fps (int, optional): _description_. Defaults to 25.
            record_every (int, optional): Record every `record_every` episodes.
                Defaults to 1.
            control_rate (int, optional): The control rate / action rate of
                the Environment to be wrapped. Defaults to 100.
            camera_id (str, optional): The Camera-ID used when recoding.
                Defaults to None.
            width (int, optional): Pixel-count in width. Defaults to 640.
            height (int, optional): Pixel-count in heigth. Defaults to 480.
            video_name (str, optional): Name of the video-file save.
                By default into `./videos/episode_X.mp4`. Defaults to None.
            cleanup_imgs (bool, optional): If enabled will automatically delete
                the generated images for recording the video. Defaults to False.
        """
        super().__init__(environment)
        assert control_rate == 100
        assert control_rate % fps == 0
        assert (
            not cleanup_imgs
        ), """Please do this manually for now.
        The current implementation seems too dangerous."""
        self._cleanup_imgs = cleanup_imgs
        self._frames = []
        self._record_every = record_every
        self._camera_id = camera_id
        self._number_of_episodes = 0
        self._record_frame_every = control_rate // fps
        self._env_step = 0
        self._fps = fps
        self._path_to_folder = pathlib.Path(path_to_folder)
        self._path_to_folder.mkdir(exist_ok=True, parents=True)
        self._video_name = video_name
        self._width = width
        self._height = height

    def step(self, action) -> dm_env.TimeStep:
        record_episode = self._number_of_episodes % self._record_every == 0
        record_step = self._env_step % self._record_frame_every == 0
        self._env_step += 1

        if record_episode and record_step:
            if self._camera_id:
                frame = self._environment.physics.render(
                    camera_id=self._camera_id, width=self._width, height=self._height
                )
            else:
                frame = self._environment.physics.render(
                    width=self._width, height=self._height
                )
            self._frames.append(frame)

        ts = super().step(action)

        if ts.last():
            self._number_of_episodes += 1
            if record_episode:
                self._make_video()

        return ts

    def get_frames(self):
        if self._frames == []:
            return None

        self._frames, frames = [], np.array(self._frames)
        return frames.transpose((0, 3, 1, 2))

    def _make_video(self):
        if self._video_name:
            video_name_const = self._video_name
        else:
            video_name_const = "episode"

        video_name = f"{video_name_const}_{self._number_of_episodes-1}.mp4"
        prefix = str(self._path_to_folder)

        for i, frame in enumerate(self._frames):
            i = str(i).zfill(3)
            Image.fromarray(frame).save(f"{prefix}/frame{i}.png")

        call(
            f"ffmpeg -r {self._fps} -i {prefix}/frame%03d.png -y {prefix}/{video_name}",
            stdout=DEVNULL,
            stderr=STDOUT,
            shell=True,
        )

        if self._cleanup_imgs:
            # clean up
            os.system(f"rm {prefix}/*.png")

        self._frames = []
