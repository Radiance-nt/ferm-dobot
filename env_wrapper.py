import time

try:
    from local_debug_logger import local_trace
except ImportError:
    local_trace = lambda: None
import gym
import mujoco_py

from abc import ABC
import numpy as np

import Dobot.DobotSDK as dobotSDK
from gripper import RobotiqGripper
import cv2


# from gym.envs.registration import register


def change_fetch_model(change_model):
    import os
    import shutil
    gym_folder = os.path.dirname(gym.__file__)
    xml_folder = 'envs/robotics/assets/fetch'
    full_folder_path = os.path.join(gym_folder, xml_folder)
    xml_file_path = os.path.join(full_folder_path, 'shared.xml')
    backup_file_path = os.path.join(full_folder_path, 'shared_backup.xml')
    if change_model:
        if not os.path.exists(backup_file_path):
            shutil.copy2(xml_file_path, backup_file_path)
        shutil.copy2('fetch_yellow_obj.xml', xml_file_path)
    else:
        if os.path.exists(backup_file_path):
            shutil.copy2(backup_file_path, xml_file_path)


def make(domain_name, task_name, seed, from_pixels, height, width, cameras=range(1),
         visualize_reward=False, frame_skip=None, reward_type='dense', change_model=False):
    if 'RealArm' not in domain_name:
        change_fetch_model(change_model)
        env = gym.make(domain_name, reward_type=reward_type)
        env = GymEnvWrapper(env, from_pixels=from_pixels, cameras=cameras, height=height, width=width)
    else:
        # import gym_xarm
        # env = gym.make(domain_name)
        # env.env.set_reward_mode(reward_type)
        env = DobotEnv()
        env = RealEnvWrapper(env, from_pixels=from_pixels, cameras=cameras, height=height, width=width)

    env.seed(seed)
    return env


class EnvWrapper(gym.Env, ABC):
    def __init__(self, env, cameras, from_pixels=True, height=100, width=100, channels_first=True):
        camera_0 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 90}
        camera_1 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 135}
        camera_2 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 180}
        camera_3 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 225}
        camera_4 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 270}
        camera_5 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 315}
        camera_6 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 0}
        camera_7 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 45}
        self.all_cameras = [camera_0, camera_1, camera_2, camera_3, camera_4, camera_5, camera_6, camera_7]

        self._env = env
        self.cameras = cameras
        self.from_pixels = from_pixels
        self.height = height
        self.width = width
        self.channels_first = channels_first

        self.special_reset = None
        self.special_reset_save = None
        self.hybrid_obs = False
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        shape = [3 * len(cameras), height, width] if channels_first else [height, width, 3 * len(cameras)]
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

        self._state_obs = None
        self.change_camera()
        self.reset()

    def change_camera(self):
        return

    @property
    def observation_space(self):
        if self.from_pixels:
            return self._observation_space
        else:
            return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def seed(self, seed=None):
        return self._env.seed(seed)

    def reset_model(self):
        self._env.reset()

    def viewer_setup(self, camera_id=0):
        for key, value in self.all_cameras[camera_id].items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def set_hybrid_obs(self, mode):
        self.hybrid_obs = mode

    def _get_obs(self):
        if self.from_pixels:
            imgs = []
            for c in self.cameras:
                imgs.append(self.render(mode='rgb_array', camera_id=c))
            if self.channels_first:
                pixel_obs = np.concatenate(imgs, axis=0)
            else:
                pixel_obs = np.concatenate(imgs, axis=2)
            if self.hybrid_obs:
                return [pixel_obs, self._get_hybrid_state()]
            else:
                return pixel_obs
        else:
            return self._get_state_obs()

    def _get_state_obs(self):
        return self._state_obs

    def _get_hybrid_state(self):
        return self._state_obs

    @property
    def hybrid_state_shape(self):
        if self.hybrid_obs:
            return self._get_hybrid_state().shape
        else:
            return None

    def step(self, action):
        self._state_obs, reward, done, info = self._env.step(action)
        return self._get_obs(), reward, done, info

    def reset(self, save_special_steps=False):
        self._state_obs = self._env.reset()
        return self._get_obs()

    def set_state(self, qpos, qvel):
        self._env.set_state(qpos, qvel)

    @property
    def dt(self):
        if hasattr(self._env, 'dt'):
            return self._env.dt
        else:
            return 1

    @property
    def _max_episode_steps(self):
        return self._env.max_path_length

    def do_simulation(self, ctrl, n_frames):
        self._env.do_simulatiaon(ctrl, n_frames)

    def render(self, mode='human', camera_id=0, height=None, width=None):
        if mode == 'human':
            self._env.render()

        if height is None:
            height = self.height
        if width is None:
            width = self.width

        if mode == 'rgb_array':
            if isinstance(self, GymEnvWrapper):
                self._env.unwrapped._render_callback()
            viewer = self._get_viewer(camera_id)
            # Calling render twice to fix Mujoco change of resolution bug.
            viewer.render(width, height, camera_id=-1)
            viewer.render(width, height, camera_id=-1)
            # window size used for old mujoco-py:
            data = viewer.read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            data = data[::-1, :, :]
            if self.channels_first:
                data = data.transpose((2, 0, 1))
            return data

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._env.close()

    def _get_viewer(self, camera_id):
        if self.viewer is None:
            from mujoco_py import GlfwContext
            GlfwContext(offscreen=True)
            self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
        self.viewer_setup(camera_id)
        return self.viewer

    def get_body_com(self, body_name):
        return self._env.get_body_com(body_name)

    def state_vector(self):
        return self._env.state_vector


class GymEnvWrapper(EnvWrapper):
    def change_camera(self):
        for c in self.all_cameras:
            c['lookat'] = np.array((1.3, 0.75, 0.4))
            c['distance'] = 1.2
        # Zoomed out cameras
        camera_8 = {'trackbodyid': -1, 'distance': 1.8, 'lookat': np.array((1.3, 0.75, 0.4)),
                    'elevation': -45.0, 'azimuth': 135}
        camera_9 = {'trackbodyid': -1, 'distance': 1.8, 'lookat': np.array((1.3, 0.75, 0.4)),
                    'elevation': -45.0, 'azimuth': 225}
        # Gripper head camera
        camera_10 = {'trackbodyid': -1, 'distance': 0.2, 'lookat': np.array((1.3, 0.75, 0.4)),
                     'elevation': -90, 'azimuth': 0}
        self.all_cameras.append(camera_8)
        self.all_cameras.append(camera_9)
        self.all_cameras.append(camera_10)

    def update_tracking_cameras(self):
        gripper_pos = self._state_obs['observation'][:3].copy()
        self.all_cameras[10]['lookat'] = gripper_pos

    def _get_obs(self):
        self.update_tracking_cameras()
        return super()._get_obs()

    @property
    def _max_episode_steps(self):
        return self._env._max_episode_steps

    def set_special_reset(self, mode):
        self.special_reset = mode

    def register_special_reset_move(self, action, reward):
        if self.special_reset_save is not None:
            self.special_reset_save['obs'].append(self._get_obs())
            self.special_reset_save['act'].append(action)
            self.special_reset_save['reward'].append(reward)

    def go_to_pos(self, pos):
        grip_pos = self._state_obs['observation'][:3]
        action = np.zeros(4)
        for i in range(10):
            if np.linalg.norm(grip_pos - pos) < 0.02:
                break
            action[:3] = (pos - grip_pos) * 10
            self._state_obs, r, d, i = self._env.step(action)
            self.register_special_reset_move(action, r)
            grip_pos = self._state_obs['observation'][:3]

    def raise_gripper(self):
        grip_pos = self._state_obs['observation'][:3]
        raised_pos = grip_pos.copy()
        raised_pos[2] += 0.1
        self.go_to_pos(raised_pos)

    def open_gripper(self):
        action = np.array([0, 0, 0, 1])
        for i in range(2):
            self._state_obs, r, d, i = self._env.step(action)
            self.register_special_reset_move(action, r)

    def close_gripper(self):
        action = np.array([0, 0, 0, -1])
        for i in range(2):
            self._state_obs, r, d, i = self._env.step(action)
            self.register_special_reset_move(action, r)

    def reset(self, save_special_steps=False):
        self._state_obs = self._env.reset()
        if save_special_steps:
            self.special_reset_save = {'obs': [], 'act': [], 'reward': []}
            self.special_reset_save['obs'].append(self._get_obs())
        if self.special_reset == 'close' and self._env.has_object:
            obs = self._state_obs['observation']
            goal = self._state_obs['desired_goal']
            obj_pos = obs[3:6]
            goal_distance = np.linalg.norm(obj_pos - goal)
            desired_reset_pos = obj_pos + (obj_pos - goal) / goal_distance * 0.06
            desired_reset_pos_raised = desired_reset_pos.copy()
            desired_reset_pos_raised[2] += 0.1
            self.raise_gripper()
            self.go_to_pos(desired_reset_pos_raised)
            self.go_to_pos(desired_reset_pos)
        elif self.special_reset == 'grip' and self._env.has_object and not self._env.block_gripper:
            obs = self._state_obs['observation']
            obj_pos = obs[3:6]
            above_obj = obj_pos.copy()
            above_obj[2] += 0.1
            self.open_gripper()
            self.raise_gripper()
            self.go_to_pos(above_obj)
            self.go_to_pos(obj_pos)
            self.close_gripper()
        return self._get_obs()

    def _get_state_obs(self):
        obs = np.concatenate([self._state_obs['observation'],
                              self._state_obs['achieved_goal'],
                              self._state_obs['desired_goal']])
        return obs

    def _get_hybrid_state(self):
        grip_pos = self._env.sim.data.get_site_xpos('robot0:grip')
        dt = self._env.sim.nsubsteps * self._env.sim.model.opt.timestep
        grip_velp = self._env.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = gym.envs.robotics.utils.robot_get_obs(self._env.sim)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        robot_info = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel])
        hybrid_obs_list = []
        if 'robot' in self.hybrid_obs:
            hybrid_obs_list.append(robot_info)
        if 'goal' in self.hybrid_obs:
            hybrid_obs_list.append(self._state_obs['desired_goal'])
        return np.concatenate(hybrid_obs_list)

    @property
    def observation_space(self):
        shape = self._get_state_obs().shape
        return gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype='float32')


class RealEnvWrapper(GymEnvWrapper):
    def render(self, mode='human', camera_id=0, height=None, width=None):
        if mode == 'human':
            self._env.render()

        if height is None:
            height = self.height
        if width is None:
            width = self.width

        if mode == 'rgb_array':
            data = self._env.render(mode='rgb_array', height=height, width=width)
            if self.channels_first:
                data = data.transpose((2, 0, 1))
            if camera_id == 8:
                data = data[3:]
            return data

    def _get_obs(self):
        return self.render(mode='rgb_array', height=self.height, width=self.width)

    def _get_state_obs(self):
        return self._get_obs()

    def reset(self, save_special_steps=False):
        self._state_obs = self._env.reset(rand_pos=True)
        return self._get_obs()


class DobotEnv(gym.Env):
    def __init__(self, ip="192.168.5.1", port="/dev/ttyUSB0", camera=4):
        self.grip = RobotiqGripper(port)
        self.grip.activate()
        self.api = dobotSDK.load()

        # sim "127.0.0.1"
        resultConnect = dobotSDK.ConnectDobot(self.api, ip)  # 真实机械臂
        print("resultConnect", resultConnect)
        if resultConnect == 0:
            dobotSDK.ClearAlarms(self.api)
            dobotSDK.SetControlMode(self.api, 1)
            dobotSDK.dSleep(1000)
            self._set_speed(60)
        else:
            raise ConnectionError
        self.init_point = [0, -600, 300]
        self.cap = cv2.VideoCapture(camera)
        self._state_obs = None
        self.start_time = 0
        self._max_episode_steps = 10
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype='float32')
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=self._get_obs().shape, dtype='float32')

    def _get_obs(self):
        return self.render(mode='rgb_array')

    def step(self, action):
        # norm action
        coor = action[:3]
        coor[0] *= 400
        coor[1] *= 400
        coor[2] *= 200
        g_ac = action[-1] * 300
        g_ac = int(g_ac)
        # coor.append(g_ac)
        print('action', action, end='   ')
        print('coor', coor, end='   ')
        print('grip', g_ac, end='   ')
        # self._move_action([0, -600, 140], isBlock=True)
        done = False
        if self._safety(coor):
            self._move_actionRel(coor, isBlock=True)
            self.grip.goToRel(g_ac)
            # time.sleep(0.3)
        else:
            print('safety problem')
            done = True
        ob = self._get_obs()
        coor_now = self._get_coordinate()[:3]
        reward = self._cal_reward(ob, coor_now)
        __pose_limit__ = 350  # 200mm
        __time_limit___ = 15  # 10s
        is_success = False
        if np.linalg.norm(np.array(coor_now) - np.array(self.init_point)) > __pose_limit__:
            print('out done')
            done = True
        if time.time() - self.start_time > __time_limit___:
            print('time done')
            done = True
        if reward > -1:
            print('win done')
            done = True
            is_success = True
        if done:
            self.reset()
        return ob, reward, done, dict(reward_dist='reward_dist', reward_ctrl='reward_ctrl', is_success=is_success)

    def _cal_reward(self, ob, coor_now):
        __reward_range__ = 40
        reward_point = [600, 300, 40]
        if np.linalg.norm(np.array(coor_now) - np.array(reward_point)) < __reward_range__:
            return 0
        return -1

    def render(self, mode='human', height=None, width=None):
        if mode == 'human':
            raise TypeError
        if mode == 'rgb_array':
            data = self._get_data_from_camera()
            cv2.imshow("capture", data)
            return data
        return 0

    def reset(self, rand_pos=True):
        if self._get_alarm():
            self._rescue()
        init_point = self.init_point
        self._move_action(init_point)
        self.grip.goTo(150)
        self._state_obs = self._get_data_from_camera()
        self.start_time = time.time()
        return self._get_obs()

    def _get_data_from_camera(self):
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (100, 100))
        frame = cv2.flip(frame, 0, dst=None)
        return frame

    def _rescue(self):
        current_coor = self._get_coordinate()
        dobotSDK.ClearAlarms(self.api)
        dobotSDK.SetControlMode(self.api, 1)
        if current_coor[2] < 40:
            self._move_actionRel((0, 0, 50))
        self.reset()

    def _move_action(self, pointList, isBlock=True):
        pointList = pointList + [-180, 0, 180]
        dobotSDK.MovL(self.api, pointList, rdnList=[-1, -1, -1, 1], isBlock=isBlock)

    def _move_actionRel(self, Rel_List, isBlock=False):
        dobotSDK.RelMovL(self.api, Rel_List[0], Rel_List[1], Rel_List[2], isBlock=isBlock)

    def _get_coordinate(self):
        return dobotSDK.GetExchange(self.api)[9][:3]

    def _get_alarm(self):
        return dobotSDK.GetExchange(self.api)[10]

    def _safety(self, coor):
        x_limit = (-550, 550)
        y_limit = (-800, -340)
        z_limit = (255, 999)  # 220 for short table
        if not x_limit[0] < self._get_coordinate()[0] + coor[0] < x_limit[1]:
            return False
        if not y_limit[0] < self._get_coordinate()[1] + coor[1] < y_limit[1]:
            return False
        if not z_limit[0] < self._get_coordinate()[2] + coor[2] < z_limit[1]:
            return False
        if self._get_alarm():
            return False
        return True

    def _set_speed(self, speed):
        dobotSDK.SetRapidRate(self.api, speed)

    def close(self):
        self.cap.release()
        dobotSDK.DisconnectDobot(self.api)
