
""" 
File containing a Deep Deterministic Policy Gradient RL agent. Adapted from 
http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html 
"""

import numpy as np
import tensorflow as tf
import gym.spaces
import rospy
import os

from moveit_msgs.srv import GetPositionFK
import moveit_msgs.msg

from basic_agents import RLAgent
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from actor_model import ActorNetwork
from critic_model import CriticNetwork

import exploration_policies

""" 
File containing a Deep Deterministic Policy Gradient RL agent. Adapted from 
http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html 
"""


MP_TRANS_FILE = '/home/cannon/reinforcement_learning/mp_trans.txt'
EXP_TRANS_FILE = '/home/cannon/reinforcement_learning/exp_trans.txt'

class OUNoise(object):
    """ Ornstein-Uhlenbeck process simulator for random exploration, as in the paper here: 
    https://arxiv.org/pdf/1509.02971v2.pdf """

    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.3):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        """ Reset the process. """
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        """ Generate noise by simulating an O-U process. """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


def update_stats(count, mean, std, x):
    """ Update and return the new mean and standard deviation, taking a new datapoint into account. """
    new_mean = np.divide(np.add(np.multiply(count, mean), x), (count + 1))
    s_n = np.multiply(np.power(std, 2), count)
    next_s_n = np.add(s_n, np.multiply(np.subtract(x, mean), np.subtract(x, new_mean)))
    new_std = np.sqrt(np.divide(next_s_n, (count + 1)))

    return new_mean, new_std


class JacksDDPGAgent(RLAgent):
    """ Class representing a reinforcement learning agent using Deep Deterministic Policy Gradients. """

    def __init__(self, env, discount_factor, summary_dir='./results', actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3, tau=1e-3, buffer_size=10000, batch_size=64, num_motion_planned=32,
                 num_demonstrations=100, exploration_rate=0.01, use_random_goal=False,
                 actor_hidden_layers=[], critic_hidden_layers=[], restore=False, save_loc="./models/model.ckpt",
                 prioritized_replay=False, priority_epsilon=0.01, episode_length=100,
                 exp_file='/home/cannon/reinforcement_learning/explored_points.txt',
                 planning_group="manipulator", full_joint_names=["shoulder_pan_joint",
                                                                  "shoulder_lift_joint",
                                                                  "elbow_joint",
                                                                  "wrist_1_joint",
                                                                  "wrist_2_joint",
                                                                  "wrist_3_joint"
                                                                ]):
        self.env_ = env
        self.current_state_ = None
        self.discount_factor_ = discount_factor
        self.batch_size_ = batch_size
        self.num_motion_planned_ = num_motion_planned
        self.num_demonstrations_ = num_demonstrations
        self.sess_ = tf.Session()
        self.save_loc_ = save_loc
        self.exp_file_ = exp_file

        self.ep_num_ = 0
        self.episode_reward_ = 0.0
        self.ep_ave_max_q_ = 0.0
        self.episode_steps_ = 0
        self.ep_success_ = False
        self.episode_length_ = episode_length

        self.exploration_rate_ = exploration_rate
        self.use_random_goal_ = use_random_goal

        self.total_step = 0

        if isinstance(self.env_.observation_space, gym.spaces.Tuple):
            state_dim = self.env_.observation_space.spaces[0].shape[0]
        else:
            state_dim = self.env_.observation_space.shape[0]

        if isinstance(self.env_.action_space, gym.spaces.Tuple):
            self.action_dim = self.env_.action_space.spaces[0].shape[0]
            self.action_bound = self.env_.action_space.spaces[0].high
            action_dim = self.action_dim
            action_bound = self.action_bound
        else:
            self.action_dim = self.env_.action_space.shape[0]
            self.action_bound = self.env_.action_space.high
            action_dim = self.action_dim
            action_bound = self.action_bound

        self.actor_ = ActorNetwork(self.sess_, state_dim, action_dim, action_bound, actor_learning_rate, tau, hidden_layers=actor_hidden_layers)
        self.critic_ = CriticNetwork(self.sess_, state_dim, action_dim, critic_learning_rate, tau,
                                               self.actor_.get_num_trainable_vars(), hidden_layers=critic_hidden_layers)

        self.exploration_noise_ = OUNoise(self.actor_.a_dim)

        self.summary_ops_, self.summary_vars_ = self._build_summaries()

        self.full_joint_names_ = full_joint_names

        self.planning_group_ = planning_group

        self.prioritized_replay_ = prioritized_replay
        self.priority_epsilon_ = priority_epsilon

        # Previous episode number
        self.prev_ep_ = 0
        # Array of experienced transitions for HER
        self.ep_exp_ = []

        self.compute_fk_ = rospy.ServiceProxy('/compute_fk', GetPositionFK)

        if os.path.isfile(self.exp_file_):
            os.remove(self.exp_file_)

        if os.path.isfile(EXP_TRANS_FILE):
            os.remove(EXP_TRANS_FILE)

        if os.path.isfile(MP_TRANS_FILE):
            os.remove(MP_TRANS_FILE)

        # TD Error prioritized replay buffer
        # self.replay_buffer_ = PrioritizedReplayBuffer(buffer_size)
        if self.prioritized_replay_:
            self.replay_buffer_ = PrioritizedReplayBuffer(buffer_size)
            self.motion_plan_buffer_ = PrioritizedReplayBuffer(buffer_size)
        else:
            self.replay_buffer_ = ReplayBuffer(buffer_size)
            self.motion_plan_buffer_ = ReplayBuffer(buffer_size)


        def exploration_reward_func(obs):
            desired_pos = np.array(obs[-3:])

            header = rospy.Header(0, rospy.Time.now(), "/base_link")
            robot_state = moveit_msgs.msg.RobotState()
            robot_state.joint_state.name = self.full_joint_names_
            fkln = ['wrist_3_link']

            next_full_joint_vals = list(obs[:self.actor_.a_dim]) + [0.0 for _ in xrange(6 - self.actor_.a_dim)]
            robot_state.joint_state.position = next_full_joint_vals
            resp = self.compute_fk_(header, fkln, robot_state)

            position = []
            position.append(resp.pose_stamped[0].pose.position.x)
            position.append(resp.pose_stamped[0].pose.position.y)
            position.append(resp.pose_stamped[0].pose.position.z)
            position = np.array(position)

            goal_reward = 0
            done = False
            if np.linalg.norm(position - desired_pos) <= 0.1:
                goal_reward = 100
                done = True

            return goal_reward - np.linalg.norm(position - desired_pos) - np.square(obs[:self.actor_.a_dim]).sum(), done

        self.exploration_policy_ = exploration_policies.FromEnvMotionPlanningExploration(exploration_reward_func,
                                                                                    self.env_,
                                                                                    group=self.planning_group_,
                                                                                    topic="/follow_joint_trajectory",
                                                                                    joints=self.env_.get_joint_names(),
                                                                                    all_joints=self.env_.get_all_joint_names(),
                                                                                    planning_time=5.0,
                                                                                    execute_timeout=2.0)

        self.summary_writer = tf.summary.FileWriter("./logs/episodes", self.sess_.graph)
        self.exploration_policy_.connect()

        # Saver has to be last to make sure that it gets all saveable variables
        self.saver_ = tf.train.Saver()
        self.save_num = 0

        if not restore:
            self.sess_.run(tf.global_variables_initializer())

            self.actor_.update_target_network()
            self.critic_.update_target_network()
        else:
            self.saver_.restore(self.sess_, self.save_loc_)

    def __del__(self):
        self.sess_.close()

    def populate_traj_buffer(self, buf, num_trajectories):
        """ Populate the input trajectory buffer with motion planned trajectories. """
        # Testing exploration policy
        i = 0
        while i < num_trajectories:
            self.env_.reset()
            traj_seq = self.exploration_policy_.explore(None)

            if traj_seq:
                i += 1
                for obs in traj_seq:

                    tmp_prev_state, tmp_action, tmp_reward, tmp_done, tmp_state = obs
                    if isinstance(self.env_.observation_space, gym.spaces.Tuple):
                        tmp_prev_state = tmp_prev_state[0]
                        tmp_state = tmp_state[0]

                    # with open(MP_TRANS_FILE, 'a') as trans_file:
                    #     trans_file.write(str((np.reshape(tmp_prev_state, (self.actor_.s_dim,)),
                    #                           np.reshape(tmp_action, (self.actor_.a_dim,)),
                    #                           tmp_reward, tmp_done, np.reshape(tmp_state, (self.actor_.s_dim,)))) + '\n')

                    if self.prioritized_replay_:
                        # Calculate initial priority
                        priority = np.square(tmp_reward) + self.priority_epsilon_
                        buf.add(np.reshape(tmp_prev_state, (self.actor_.s_dim,)),
                                np.reshape(tmp_action, (self.actor_.a_dim,)),
                                tmp_reward, tmp_done, np.reshape(tmp_state, (self.actor_.s_dim,)),
                                priority)
                    else:
                        buf.add(np.reshape(tmp_prev_state, (self.actor_.s_dim,)),
                                np.reshape(tmp_action, (self.actor_.a_dim,)),
                                tmp_reward, tmp_done, np.reshape(tmp_state, (self.actor_.s_dim,)))

    def _build_summaries(self):
        episode_reward = tf.Variable(0.0)
        ep_merged = tf.summary.scalar("Reward", episode_reward)
        episode_ave_max_q = tf.Variable(0.0)
        ep_merged = tf.summary.merge([ep_merged, tf.summary.scalar("Qmax_Value", episode_ave_max_q)])

        summary_vars = [episode_reward, episode_ave_max_q]

        return ep_merged, summary_vars

    def _accumulate_her_exp(self):
        """
        Parse accumulated experience from an episode into HER experience.
        """
        if len(self.ep_exp_) == 0:
            return

        # # Sparse cartesian reward.
        # def reward_func(obs, position):
        #     desired_pos = np.array(obs[-3:])
        #     position = np.array(position)
        #
        #     goal_reward = -1
        #     if np.linalg.norm(position - desired_pos) <= 0.1:
        #         goal_reward = 1
        #
        #     return goal_reward

        # Smoothed cartesian reward.
        def reward_func(obs, position):
            desired_pos = np.array(obs[-3:])
            position = np.array(position)

            goal_reward = 0
            done = False
            if np.linalg.norm(position - desired_pos) <= 0.1:
                goal_reward = 100
                done = True

            return goal_reward - np.linalg.norm(position - desired_pos), done

        header = rospy.Header(0, rospy.Time.now(), "/base_link")
        robot_state = moveit_msgs.msg.RobotState()
        robot_state.joint_state.name = self.full_joint_names_
        fkln = ['wrist_3_link']     # Might want to not hardcode this

        num_diff_joints = 6 - self.actor_.a_dim  # Also might not want to hardcode a max of 6

        # Find HER goal
        final_state = np.array(self.ep_exp_[-1][-1])
        final_joints = final_state[:self.actor_.a_dim]
        final_full_joints = final_joints.tolist() + [0.0 for _ in xrange(num_diff_joints)]
        robot_state.joint_state.position = final_full_joints
        resp = self.compute_fk_(header, fkln, robot_state)
        her_goal = []
        her_goal.append(resp.pose_stamped[0].pose.position.x)
        her_goal.append(resp.pose_stamped[0].pose.position.y)
        her_goal.append(resp.pose_stamped[0].pose.position.z)
        her_goal = np.array(her_goal)

        i = 0
        for trans in self.ep_exp_:
            i += 1
            prev_state, action, reward, done, next_state = trans

            next_full_joint_vals = list(next_state[:self.actor_.a_dim]) + [0.0 for _ in xrange(num_diff_joints)]
            robot_state.joint_state.position = next_full_joint_vals
            resp = self.compute_fk_(header, fkln, robot_state)
            position = []
            position.append(resp.pose_stamped[0].pose.position.x)
            position.append(resp.pose_stamped[0].pose.position.y)
            position.append(resp.pose_stamped[0].pose.position.z)

            with open(self.exp_file_, 'a') as exp_file:
                exp_file.write("{},{},{},{},{}\n".format(position[0], position[1], position[2],
                                                         prev_state[-3:],  # Record goal
                                                         self.ep_success_))  # Record success for visualization
                exp_file.flush()

            prev_full_joint_vals = list(prev_state[:self.actor_.a_dim]) + [0.0 for _ in xrange(num_diff_joints)]
            robot_state.joint_state.position = prev_full_joint_vals
            resp = self.compute_fk_(header, fkln, robot_state)
            prev_position = []
            prev_position.append(resp.pose_stamped[0].pose.position.x)
            prev_position.append(resp.pose_stamped[0].pose.position.y)
            prev_position.append(resp.pose_stamped[0].pose.position.z)

            next_goal_diff = (her_goal - np.array(position)).tolist()
            prev_goal_diff = (her_goal - np.array(prev_position)).tolist()
            goal = her_goal.tolist()

            her_prev_state = prev_state[:-6].tolist() + prev_goal_diff + goal
            her_next_state = next_state[:-6].tolist() + next_goal_diff + goal

            her_reward, new_done = reward_func(her_next_state, position)
            her_reward -= np.square(action).sum()

            if self.prioritized_replay_:
                # Calculate initial priority
                priority = np.square(her_reward) + self.priority_epsilon_
                self.replay_buffer_.add(np.reshape(her_prev_state, (self.actor_.s_dim,)),
                                        np.reshape(action, (self.actor_.a_dim,)),
                                        her_reward, new_done, np.reshape(her_next_state, (self.actor_.s_dim,)),
                                        priority)
            else:
                self.replay_buffer_.add(np.reshape(her_prev_state, (self.actor_.s_dim,)),
                                        np.reshape(action, (self.actor_.a_dim,)),
                                        her_reward, new_done, np.reshape(her_next_state, (self.actor_.s_dim,)))

            if new_done:
                break

        self.ep_exp_ = []

    def reset(self):
        """
        Reset the agent by resetting the contained environment.

        :return: The observed reset state.
        """
        self.current_state_ = np.array(self.env_.reset())
        if isinstance(self.env_.observation_space, gym.spaces.Tuple):
            self.current_state_ = self.current_state_[0]

        self.exploration_noise_.reset()

        if self.episode_steps_ > 0:
            summary_str = self.sess_.run(self.summary_ops_, feed_dict={
                self.summary_vars_[0]: self.episode_reward_,
                self.summary_vars_[1]: self.ep_ave_max_q_ / self.episode_steps_
            })

            self.summary_writer.add_summary(summary_str, self.ep_num_)
            self.summary_writer.flush()

        self.ep_ave_max_q_ = 0.0
        self.episode_reward_ = 0.0
        self.episode_steps_ = 0
        self._accumulate_her_exp()

        self.ep_success_ = False

        return self.current_state_

    def step(self, episode_num, step_num, num_explore=1, testing=False):
        """
        Step by taking an action according to the actor network, updating networks.
        :param num_explore: Number of motion planning trajectories to explore.
        :return: (state, action, reward, next_state, done) 
        """

        self.ep_num_ = episode_num

        if self.total_step == 0 and not testing:
            self.populate_traj_buffer(self.motion_plan_buffer_, self.num_demonstrations_)

        if not testing:
            self.total_step += 1

        if episode_num % 1000 == 0 and step_num == 0 and not testing:
            self.save_num += 1000

        prev_state = self.current_state_


        #action, _ = self.actor_.predict(prev_state.reshape(1, self.actor_.s_dim))


        # take a random action.
        
        action = np.random.uniform(-self.action_bound,self.action_bound, size=(1,self.action_dim))
        print "action:",action

        if not testing:
            # Exploration term
            epsilon = np.exp(-episode_num / 25)

            if isinstance(self.env_.action_space, gym.spaces.Tuple):
                action += epsilon * self.exploration_noise_.noise() / self.env_.action_space.spaces[0].high
            else:
                action += epsilon * self.exploration_noise_.noise() / self.env_.action_space.high

        # Actually step the environment.
        self.current_state_, reward, done, _ = self.env_.step(action)
        if isinstance(self.env_.observation_space, gym.spaces.Tuple):
            self.current_state_ = self.current_state_[0]

        # To fit with motion planned trajectories
        if step_num >= self.episode_length_ - 1:
            done = True

        # We assume done means success here, as it does in the robot env
        if done:
            self.ep_success_ = True

        self.current_state_ = np.array(self.current_state_)

        self.episode_steps_ += 1

        self.ep_exp_.append((prev_state, action, reward, done, self.current_state_))

        # Add transition to replay buffer.
        if not testing:
            # with open(EXP_TRANS_FILE, 'a') as exp_trans_file:
            #     exp_trans_file.write(str((np.reshape(prev_state, (self.actor_.s_dim,)), np.reshape(action, (self.actor_.a_dim,)),
            #                     reward, done, np.reshape(self.current_state_, (self.actor_.s_dim,)))) + '\n')
            if self.prioritized_replay_:
                # Calculate initial priority
                priority = np.square(reward) + self.priority_epsilon_
                self.replay_buffer_.add(np.reshape(prev_state, (self.actor_.s_dim,)),
                                        np.reshape(action, (self.actor_.a_dim,)),
                                        reward, done, np.reshape(self.current_state_, (self.actor_.s_dim,)),
                                        priority)
            else:
                self.replay_buffer_.add(np.reshape(prev_state, (self.actor_.s_dim,)), np.reshape(action, (self.actor_.a_dim,)),
                                    reward, done, np.reshape(self.current_state_, (self.actor_.s_dim,)))

        if self.replay_buffer_.size() > self.batch_size_ and not testing:
            # Train multiple times as in HER paper
            for _ in xrange(2):

                if self.motion_plan_buffer_.size() > self.batch_size_:
                    if self.prioritized_replay_:
                        data, indices = self.replay_buffer_.sample_batch(self.batch_size_ - self.num_motion_planned_)
                        plan_data, plan_indices = self.motion_plan_buffer_.sample_batch(self.num_motion_planned_)
                        s_batch, a_batch, r_batch, t_batch, s2_batch = data
                        s_plan_batch, a_plan_batch, r_plan_batch, t_plan_batch, s2_plan_batch = plan_data
                    else:
                        s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer_.sample_batch(self.batch_size_ - self.num_motion_planned_)
                        s_plan_batch, a_plan_batch, r_plan_batch, t_plan_batch, s2_plan_batch = self.motion_plan_buffer_.sample_batch(self.num_motion_planned_)

                    s_batch = np.append(s_batch, s_plan_batch, axis=0)
                    a_batch = np.append(a_batch, a_plan_batch, axis=0)
                    r_batch = np.append(r_batch, r_plan_batch, axis=0)
                    t_batch = np.append(t_batch, t_plan_batch, axis=0)
                    s2_batch = np.append(s2_batch, s2_plan_batch, axis=0)
                else:
                    if self.prioritized_replay_:
                        data, indices = self.replay_buffer_.sample_batch(self.batch_size_)
                        s_batch, a_batch, r_batch, t_batch, s2_batch = data
                    else:
                        s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer_.sample_batch(self.batch_size_)

                predicted_action, tanh_preactivations = self.actor_.predict_target(s2_batch)

                # Calculate target values
                target_q = self.critic_.predict_target(s2_batch, predicted_action)

                y = []
                for i in xrange(self.batch_size_):
                    if t_batch[i]:
                        y.append(r_batch[i])
                    else:
                        y.append(r_batch[i] + self.discount_factor_ * target_q[i])

                # Update the critic network
                predicted_q_value, _ = self.critic_.train(s_batch, a_batch, np.reshape(y, (self.batch_size_, )),
                                                          tanh_preactivations, self.total_step)

                # Perform priority update based on squared TD error
                if self.prioritized_replay_:
                    current_target_q = self.critic_.predict(s_batch, a_batch)
                    priorities = []

                    for i in xrange(self.batch_size_):
                        priorities.append(np.square((y[i] - current_target_q[i])) + self.priority_epsilon_)

                    if self.motion_plan_buffer_.size() > self.batch_size_:
                        self.replay_buffer_.priority_update(indices, priorities[:self.num_motion_planned_])
                        self.motion_plan_buffer_.priority_update(plan_indices, priorities[self.num_motion_planned_:])
                    else:
                        self.replay_buffer_.priority_update(indices, priorities)

                self.ep_ave_max_q_ += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs, tanh_preactivations = self.actor_.predict(s_batch)
                grads = self.critic_.action_gradients(s_batch, a_outs)
                self.actor_.train(s_batch, grads[0], self.total_step)

                # Update target networks
                self.actor_.update_target_network()
                self.critic_.update_target_network()

        self.episode_reward_ += reward

        return prev_state, action, reward, self.current_state_, done

    def save(self):
        """ Save the tensorflow graph defining the critic and actor networks. """
        save_path = self.saver_.save(self.sess_, self.save_loc_, global_step=self.save_num)
        print "Model saved to %s" % save_path
