import pygame
import math

CAR_WIDTH = 40
CAR_HEIGHT = 20
TURN_ANGLE = 7
SPEED = 2.5

class CarEnv:
    def __init__(self, width = 800, height = 600):
        self.width = width
        self.height = height
        self.action_space = [0, 1, 2] # 0 = forward, 1 = left, 2 = right
        self.reset()

    def reset(self):
        self.pos = pygame.Vector2(self.width / 2, self.height / 2)
        self.angle = 0
        self.done = False
        return self._get_state()

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}
        
        self._apply_action(action)
        self._move_forward()

        # REWARD TUNING
        reward = -0.01 # Helps w/ stalling so agent must move
        if not self._within_bounds():
            self.done = True
            reward = -5 # Penalty for going oob
        else:
            reward += 0.2 # Rewards/sec = FPS * this I think

        return self._get_state(), reward, self.done, {}

    def _apply_action(self, action):
        # 0 = forward, 1 = left, 2 = right
        if action == 1:
            self.angle -= TURN_ANGLE
        elif action == 2:
            self.angle += TURN_ANGLE

        self.angle = self.angle % 360 # Normalize so within 0 to 360

    def _move_forward(self):
        rad = math.radians(self.angle)
        direction = pygame.Vector2(math.cos(rad), math.sin(rad))
        self.pos += direction * SPEED

    def _within_bounds(self):
        return 0 < self.pos.x < self.width and 0 < self.pos.y < self.height

    def _get_state(self):
        #return (self.pos.x, self.pos.y, self.angle)

        # Make pos more broad to fit single states so less memory & easier control
        # 25px & 20 deg cells
        return (
            int(self.pos.x // 25),
            int(self.pos.y // 25),
            int(self.angle % 360 // 20)
        )

    def render(self, screen):
        screen.fill((10, 10, 10))

        car = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
        car.fill((255, 255, 0))

        rotated = pygame.transform.rotate(car, -self.angle)
        rect = rotated.get_rect(center=self.pos)

        screen.blit(rotated, rect.topleft)
        pygame.display.flip()
