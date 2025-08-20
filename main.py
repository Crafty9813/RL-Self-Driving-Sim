import pygame
import numpy as np
import collections

from car_env import CarEnv
from q_table_utils import save_q_table, load_q_table

FPS = 60
lr = 0.2
discount_f = 0.9

model_path = "q_table.pkl"

testing_mode = True # MODIFY THIS IF YOUR TRAINING OR TESTING

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    env = CarEnv()

    exploration_r = 1.0
    min_exploration = 0.02
    exploration_decay = 0.995

    try:
        q_table = load_q_table(model_path, len(env.action_space))
        print("Succesfully loaded Q-table! :D")
    except FileNotFoundError:
        q_table = collections.defaultdict(lambda: np.zeros(len(env.action_space)))
        print("New Q-table created")

    running = True
    episode_num = 0

    while running:
        episode_num += 1
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True

            if testing_mode:
                action = np.argmax(q_table[state]) # EXPLOIT ONLY IF TESTING
            else:
                # ---- TRAINING (epsilon greedy method) ----
                if np.random.rand() < exploration_r:
                    action = np.random.choice(env.action_space) # Choose random action for exploration
                else:
                    action = np.argmax(q_table[state]) # Exploit known action

            next_state, reward, done, _ = env.step(action)
            total_reward+=reward
            
            if not testing_mode:
                #---- Q-table updates (ONLY DURING TRAINING) ----
                best_next = np.max(q_table[next_state])
                q_table[state][action] += lr * (reward + discount_f * best_next - q_table[state][action])

            state = next_state

            env.render(screen)
            clock.tick(FPS)

        # IF TRAINING DECAY EXPLORATION RATE SO LATER ON STILL EXPLOITS
        if not testing_mode:
            exploration_r = max(min_exploration, exploration_r * exploration_decay)

        print(f"Episode: {episode_num}, reward: {total_reward:.2f}")

    if not testing_mode:
        save_q_table(q_table, model_path)
        print("Training complete! Q-table saved.")
    else:
        print("Testing done, Q-table not modified.")

    pygame.quit()

if __name__ == "__main__":
    main()
