import pygame
import numpy as np
import collections

from car_env import CarEnv
from q_table_utils import save_q_table, load_q_table

FPS = 60
lr = 0.2
discount_f = 0.9

model_path = "q_table.pkl"

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    env = CarEnv()

    exploration_r = 1.0
    min_exploration = 0.02
    exploration_decay = 0.98

    try:
        q_table = load_q_table(model_path, len(env.action_space))
        print("Succesfully loaded Q-table! :D")
    except FileNotFoundError:
        q_table = collections.defaultdict(lambda: np.zeros(len(env.action_space)))
        print("New Q-table created")

    running = True

    while running:
        state = env.reset()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True

            # epsilon greedy method
            if np.random.rand() < exploration_r:
                action = np.random.choice(env.action_space) # Choose random action for exploration
            else:
                action = np.argmax(q_table[state]) # Exploit known action

            next_state, reward, done, _ = env.step(action)
            

            #---- Q-learning updates ----
            best_next = np.max(q_table[next_state])
            q_table[state][action] += lr * (reward + discount_f * best_next - q_table[state][action])

            state = next_state

            env.render(screen)
            clock.tick(FPS)

        exploration_r = max(min_exploration, exploration_r * exploration_decay)

    save_q_table(q_table, model_path)
    print("Training complete! Q-table saved.")

    pygame.quit()

if __name__ == "__main__":
    main()
