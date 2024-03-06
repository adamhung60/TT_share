import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import pkg_resources
import cv2
import time
import random
import math

class TT2Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, num_stages = -1, current_stage = 0):
        
        # paths to urdfs for pybullet rendering
        self.robot_path = pkg_resources.resource_filename(__name__, 'robot/robot.urdf')
        self.table_path = pkg_resources.resource_filename(__name__, 'table/robot.urdf')
        self.ball_path = pkg_resources.resource_filename(__name__, 'ball/robot.urdf')

        self.render_mode = render_mode
        self.max_steps = 750 # max steps before truncation (defined here instead of during Gym registration)
        self.steps_taken = 0

        self.joints = [0, 1, 2, 3, 4, 5]
        self.numJoints = len(self.joints)
        self.frame_skips = 10 # frames rendered by pybullet before agent is queried for a new action
        if num_stages > 0:
            self.num_stages = num_stages # curriculum learning var, corresponds to ball placement variance
            self.current_stage = current_stage
            start = 1
            end = np.exp(1)
            values = np.linspace(start, end,self.num_stages)
            log_values = np.log(values)
            self.var_coeff = log_values[self.current_stage]
        else:
            self.var_coeff = 1
        print("new stage with variance coefficient: ", self.var_coeff)

        # vars for monitoring during training
        self.episode_count = 0
        self.ball_in_count = 0
        self.ball_touch_count = 0
        self.d2t_sum = 0
        self.ball_in_vec = []

        # pybullet setup
        if self.render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        p.setRealTimeSimulation(0)
        p.resetSimulation()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.numJoints,), dtype=float)

    def get_obs(self):

        joint_positions = [p.getJointState(self.robot, joint)[0] for joint in self.joints]
        joint_velocities = [p.getJointState(self.robot, joint)[1] for joint in self.joints]
        ball_position = list(p.getBasePositionAndOrientation(self.ball)[0])
        ball_velocity = list(p.getBaseVelocity(self.ball)[0])
        current_obs = np.array(joint_positions+joint_velocities+ball_position+ball_velocity)
        #self.obs = self.obs[1:]
        #self.obs = np.vstack((self.obs, current_obs))
        #return self.obs.flatten()
        return current_obs
    
    def reset(self,seed=None, options = None): 
        super().reset(seed=seed) # setting random seed for environment
        #reset simulation vars and entities
        self.episode_count += 1
        self.terminated = False
        self.truncated = False
        self.steps_taken = 0
        self.state = 0
        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        self.plane = p.loadURDF("plane.urdf") 
        self.robot = p.loadURDF(self.robot_path, [-1.8,0,0], useFixedBase = 1)
        self.table = p.loadURDF(self.table_path, [0,0,0],useFixedBase = 1)
        
        #load in the ball, and assign initial velocity
        start, v = self.get_trajectory()
        self.ball = p.loadURDF(self.ball_path, start)
        p.resetBaseVelocity(self.ball, linearVelocity = v)
        #self.ball = p.loadURDF(self.ball_path, [0.5,0,0.3],useFixedBase = 0)
        #p.resetBaseVelocity(self.ball, linearVelocity = [-1.6,0,3])

        #make the paddle and table bouncy
        p.changeDynamics(self.ball, -1, restitution = 0.9)
        p.changeDynamics(self.table, -1, restitution = 0.9)
        p.changeDynamics(self.robot, 2, restitution = 0.9)

        #obs = [p.getJointState(self.robot, joint)[0] for joint in self.joints] + list(p.getBasePositionAndOrientation(self.ball)[0])
        #self.obs = np.vstack([obs] * 8)
        #self.prevPos = list(p.getLinkState(self.robot, 3)[0])

        observation = self.get_obs()
        info = {"info":"info"} # dict info return required by Gym

        return observation, info
    
    # assign start position, goal position, and initial velocity vector for the ball to get from start to goal

    def get_bounds(self, floor, ciel):

        lower = (ciel+floor)/2 - (ciel-floor)*self.var_coeff/2
        upper = (ciel+floor)/2 + (ciel-floor)*self.var_coeff/2
        return [lower,upper]

    def get_trajectory(self):
        #"""
        goal_x = self.get_bounds(-1.3, -0.9)
        goal_y = self.get_bounds(-0.7, 0.7)
        start_x = self.get_bounds(0.5, 1.5)
        start_y = self.get_bounds(-0.7, 0.7)
        start_z = self.get_bounds(0.2, 0.3)
        #"""
        goal = [random.uniform(goal_x[0], goal_x[1]), random.uniform(goal_y[0], goal_y[1]), 0.1]
        start = [random.uniform(start_x[0], start_x[1]), random.uniform(start_y[0], start_y[1]), random.uniform(start_z[0], start_z[1])]
       
        goal = [-1.1, random.uniform(-0.7, 0.7), 0.1]
        start = [1.0, 0.0, 0.25]
        
        vz_bounds = self.get_bounds(3,4)
        vz = random.uniform(vz_bounds[0], vz_bounds[1])
        t = (vz + math.sqrt(vz**2 + 19.6*(start[2]+goal[2])))/9.8
        vx = (goal[0] - start[0])/t
        vy = (goal[1] - start[1])/t
        v = [vx,vy,vz]
        return start, v

    # helper function for euclidean distance
    def d_euc(self, point1, point2):
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))
    
    # distance between robot end-effector and ball
    def r2b(self):
        rob = list(p.getLinkState(self.robot, self.numJoints)[0])
        ball = list(p.getBasePositionAndOrientation(self.ball)[0])
        return self.d_euc(rob, ball)
    #r2b = self.r2b()
    #return min(1/r2b, 5)/500

    def get_reward(self):

        if self.state == 0: # up until the ball bounces on table
            if p.getContactPoints(self.table, self.ball):
                self.state = 1
        elif self.state == 1: # up until robot hits ball
            if p.getContactPoints(self.robot, self.ball, 5): # paddle makes contact with ball
                self.state = 2
                self.ball_touch_count += 1
                return 1
            elif p.getContactPoints(self.ball):
                self.terminated = True
        elif self.state == 2: # up until ball lands
            point = p.getContactPoints(self.ball, self.plane) or p.getContactPoints(self.ball, self.table)
            if point:
                point = p.getContactPoints(self.ball)
                d2t = self.d2t(point[0][5][0],point[0][self.numJoints][1])
                self.d2t_sum += d2t
                self.terminated = True
                if d2t == 0:
                    self.ball_in_count += 1
                    return 10
                else:
                    return max(-d2t/3, -1)
        return 0
    
    def d2t(self,x,y): # distance between point (x,y) and opponent's side of table
        if x < 0.1:
            dx = abs(.1 - x) # 0.1 here prevent agent from learning to hit the net
        elif x > 1.37:
            dx = x - 1.37
        else:
            dx = 0

        if y > 0.7625:
            dy = y - 0.7625
        elif y < -0.7625:
            dy = 0.7625 - y
        else:
            dy = 0
            
        return math.sqrt(dx**2 + dy**2)

    def step(self, action):
        # actions bound to (-1,1), so scale back up to more realistic values
        actions = action*5
        # set target velocities with infinite force
        p.setJointMotorControlArray(
            bodyUniqueId = self.robot, 
            jointIndices = self.joints, 
            controlMode = p.VELOCITY_CONTROL, 
            targetVelocities = actions,
            forces = [500]*self.numJoints)
        
        # step simulation by number of frame skips
        reward = 0
        for _ in range(self.frame_skips):
            p.stepSimulation()
            self.steps_taken += 1
            if self.steps_taken >= self.max_steps:
                self.truncated = True

            # for monitoring training
            if self.episode_count >= 1000:
                print("balls in / 1000 = ", self.ball_in_count)
                self.ball_in_vec += [self.ball_in_count]
                print("balls touched / 1000 = ", self.ball_touch_count)
                print("average d2t for touched balls = ", 
                      self.d2t_sum/(self.ball_touch_count+ 0.0001)) # inhibit division by 0
                self.episode_count = 0
                self.ball_in_count = 0
                self.ball_touch_count = 0
                self.d2t_sum = 0
            
            #control_cost = - self.d_euc(targetPos, self.prevPos)/20
            reward += self.get_reward() #- control_cost

            if self.render_mode == "human":
                self.render_frame()
            
            # break out of loop if episode ends
            if self.truncated or self.terminated:
                break

        observation = self.get_obs()
        info = {"info":"hi"}

        return observation, reward, self.terminated, self.truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_frame()
    
    # self.step() function handles rendering for "human" mode 
    # For "rgb_array" mode, return static image of scene
    def render_frame(self):
        if self.render_mode == "human":
            time.sleep(0.0001) # small delay for better visualization

        if self.render_mode == "rgb_array":
            focus_position,_ = p.getBasePositionAndOrientation(self.robot)
            focus_position = tuple([focus_position[0] + 0.2,focus_position[1] + 0.2,focus_position[2]])
            p.resetDebugVisualizerCamera(
                cameraDistance=0.9, 
                cameraYaw = 40, 
                cameraPitch = -12, 
                cameraTargetPosition = focus_position
            )

            h,w = 4000, 4000
            image = np.array(p.getCameraImage(h, w)[2]).reshape(h,w,4)
            image = image[:, :, :3]
            image = cv2.convertScaleAbs(image)
            return image
    
    # parent class requires override
    def close(self):
        print(self.ball_in_vec)
        return

