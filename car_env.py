import pygame
import math
import numpy as np

CAR_WIDTH = 40
CAR_HEIGHT = 20
TURN_ANGLE = 7
SPEED = 2

class CarEnv:
    def __init__(self, width = 800, height = 600):
        self.width = width
        self.height = height
        self.action_space = [0, 1, 2] # 0 = forward, 1 = left, 2 = right

        self.goal = pygame.Vector2(70, 70)

        self.reset()

    def reset(self):
        self.pos = pygame.Vector2(self.width - 50, self.height - 50)
        self.angle = 180
        self.episode_done = False
        self.prev_distance = self.pos.distance_to(self.goal)

        return self._get_state()

    def step(self, action):
        if self.episode_done:
            return self._get_state(), 0, True, {}
        
        self._apply_action(action)
        self._move_forward()

        # ORIGINAL
        # reward: -0.01, progress*0.15, +=20, -10

        reward = -0.01 # Helps w/ stalling so agent must move

        distance = self.pos.distance_to(self.goal)
        progress = self.prev_distance - distance
        reward += progress * 0.11 # Progress reward weight

        self.prev_distance = distance

        if distance < 25:
            reward += 20
            self.episode_done = True

        if not self._within_bounds():
            self.episode_done = True
            reward = -10 # Penalty for going oob

        return self._get_state(), reward, self.episode_done, {}

    def _apply_action(self, action):
        # 0 = forward, 1 = left, 2 = right
        if action == 1:
            self.angle -= TURN_ANGLE
        elif action == 2:
            self.angle += TURN_ANGLE

        self.angle = self.angle % 360 # Normalize so within 0 to 360 deg

    def _move_forward(self):
        rad = math.radians(self.angle)
        direction = pygame.Vector2(math.cos(rad), math.sin(rad))
        self.pos += direction * SPEED

    def _within_bounds(self):
        return 0 < self.pos.x < self.width and 0 < self.pos.y < self.height

    def _get_state(self):
        #return (self.pos.x, self.pos.y, self.angle)

        # Make pos more broad to fit single states so less memory & easier control
        # 25px & 30 deg cells
        return (
            int(self.pos.x // 25),
            int(self.pos.y // 25),
            int(self.angle % 360 // 30),
        )

    def render(self, screen):
        screen.fill((40, 40, 40))

        car = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
        #pygame.draw.rect(car, (255, 255, 0), car.get_rect(), border_radius = min(CAR_WIDTH, CAR_HEIGHT) // 4)
        car.fill((255, 255, 0))

        rotated = pygame.transform.rotate(car, -self.angle)
        rect = rotated.get_rect(center=self.pos)

        pygame.draw.circle(screen, (0, 255, 0), (int(self.goal.x), int(self.goal.y)), 10)

        screen.blit(rotated, rect.topleft)
        pygame.display.flip()
