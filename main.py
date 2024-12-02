import matplotlib.pyplot as plt
from absl import app
from absl import flags
from environment.env import SumoEnv
from agents.dqn import DqnAgent
from replay import ReplayBuffer
import torch
from datetime import datetime
import math
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_integer('skip_range', 10, 'time(seconds) range for skip randomly at the beginning')
flags.DEFINE_float('simulation_time', 5000, 'time for simulation')
flags.DEFINE_integer('yellow_time', 2, 'time for yellow phase')
flags.DEFINE_integer('delta_rs_update_time', 10, 'time for calculate reward')
flags.DEFINE_string('reward_fn', 'choose-min-waiting-time', 'Reward function for simulation')
flags.DEFINE_string('net_file', 'nets/2way-single-intersection/single-intersection.net.xml', 'Network file for SUMO')
flags.DEFINE_string('route_file', 'nets/2way-single-intersection/single-intersection-vhvh.rou.xml', 'Route file for SUMO')
flags.DEFINE_bool('use_gui', True, 'Use SUMO GUI')
flags.DEFINE_integer('num_episodes', 10, 'Number of episodes for training/testing')
flags.DEFINE_string('network', 'dqn', 'Type of network (e.g., DQN)')
flags.DEFINE_string('mode', 'train', 'Mode of operation: train or test')
flags.DEFINE_float('eps_start', 1.0, 'Starting epsilon for exploration')
flags.DEFINE_float('eps_end', 0.1, 'Ending epsilon for exploration')
flags.DEFINE_integer('eps_decay', 83000, 'Epsilon decay rate')
flags.DEFINE_integer('target_update', 3000, 'Target network update frequency')
flags.DEFINE_string('network_file', '', 'Path to pretrained network weights')
flags.DEFINE_float('gamma', 0.95, 'Discount factor for rewards')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training')
flags.DEFINE_bool('use_sgd', True, 'Use SGD optimizer')

device = "cuda" if torch.cuda.is_available() else "cpu"

time = str(datetime.now()).split('.')[0].split(' ')[0]
time = time.replace('-', '')

# Visualization setup
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Average Queue Length per Episode", fontsize=16)
ax.set_xlabel("Episode", fontsize=14)
ax.set_ylabel("Average Queue Length", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
line, = ax.plot([], [], label="Avg Queue Length", marker='o', linestyle='-', linewidth=2)
trendline, = ax.plot([], [], label="Trend (Rolling Avg)", color='orange', linewidth=2)
ax.legend(fontsize=12)
queue_lengths = []

def update_plot():
    """Updates the real-time plot."""
    line.set_data(range(len(queue_lengths)), queue_lengths)
    
    # Add rolling average trendline
    if len(queue_lengths) > 5:
        rolling_avg = np.convolve(queue_lengths, np.ones(5) / 5, mode='valid')
        trendline.set_data(range(len(rolling_avg)), rolling_avg)
    else:
        trendline.set_data([], [])
    
    ax.set_xlim(0, max(1, len(queue_lengths)))
    ax.set_ylim(0, max(queue_lengths) + 5)
    plt.draw()
    plt.pause(0.01)

def main(argv):
    del argv  # Unused
    env = SumoEnv(
        net_file=FLAGS.net_file,
        route_file=FLAGS.route_file,
        skip_range=FLAGS.skip_range,
        simulation_time=FLAGS.simulation_time,
        yellow_time=FLAGS.yellow_time,
        delta_rs_update_time=FLAGS.delta_rs_update_time,
        reward_fn=FLAGS.reward_fn,
        mode=FLAGS.mode,
        use_gui=FLAGS.use_gui,
    )
    replay_buffer = ReplayBuffer(capacity=20000)

    agent = None
    if FLAGS.network == 'dqn':
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        agent = DqnAgent(
            FLAGS.mode, replay_buffer, FLAGS.target_update, FLAGS.gamma, FLAGS.use_sgd, 
            FLAGS.eps_start, FLAGS.eps_end, FLAGS.eps_decay, input_dim, output_dim, 
            FLAGS.batch_size, FLAGS.network_file
        )

    for episode in range(FLAGS.num_episodes):
        initial_state = env.reset()
        env.train_state = initial_state
        done = False
        invalid_action = False
        total_queue = 0
        step_count = 0

        while not done:
            state = env.compute_state()
            action = agent.select_action(state, replay_buffer.steps_done, invalid_action)
            next_state, reward, done, info = env.step(action)
            total_queue += env.get_queue_length()  # Replace with actual method to get queue length
            step_count += 1

            if info['do_action'] is None:
                invalid_action = True
                continue
            invalid_action = False

            if FLAGS.mode == 'train':
                replay_buffer.add(env.train_state, env.next_state, reward, info['do_action'])
                if not agent.update_gamma:
                    agent.learn()
                else:
                    agent.learn_gamma()

        avg_queue = total_queue / max(1, step_count)  # Avoid division by zero
        queue_lengths.append(avg_queue)
        update_plot()

        env.close()
        if FLAGS.mode == 'train' and episode != 0 and episode % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f'weights/weights_{time}_{episode}.pth')

        print(f"Episode {episode}:")
        print(f"  Avg Queue Length: {avg_queue:.2f}")
        print(f"  Epsilon Threshold: {FLAGS.eps_end + (FLAGS.eps_start - FLAGS.eps_end) * math.exp(-1. * replay_buffer.steps_done / FLAGS.eps_decay):.2f}")
        print(f"  Learn Steps: {agent.learn_steps}")
        print(f"  Gamma: {agent.gamma}")

    print("Training completed.")
    plt.ioff()
    
    # Save the final plot
    plt.savefig(f"average_queue_plot_{time}.png")
    plt.show()

if __name__ == '__main__':
    app.run(main)
