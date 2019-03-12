import numpy as np
from physics_sim import PhysicsSim

def angle(v1, v2, acute=True):
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if (acute == True):
        return angle
    else:
        return 2 * np.pi - angle

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self, done, rotor_speeds):
        """Uses current pose of sim to return reward."""
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        # reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        
        distance_thresh = 1/self.sim.upper_bounds[2]/10.
        to_point = self.sim.pose[:3] - self.target_pos
        distance = (abs(to_point)).sum()/self.sim.upper_bounds[2]/10
        
        # give a reward if very close
        reward = .8 - distance
        
        # gives a great reward if he is in the spot
        reward += 2 if distance < distance_thresh else 0
        
        # gives reward if the drone is moving to the direction
#          reward += 0.02*distance*(angle(to_point, self.sim.v)-np.pi/2)
        
        # remove reward if it has too much velocity
        reward -= 0.03 * abs(self.sim.angular_accels).sum()
        
        # penalize for big rotation differences
        rot_diff = abs(max(rotor_speeds) - min(rotor_speeds))/self.action_high
        reward -= 0.03 if rot_diff >.1 else 0
        
#         reward -= 0.008 * abs(self.sim.angular_v).sum()

        if done and self.sim.time < self.sim.runtime:
            reward -= 2
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(done, rotor_speeds) 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state