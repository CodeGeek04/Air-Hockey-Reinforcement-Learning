import pygame
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()

# Window dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 750

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddle attributes
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20
PADDLE_SPEED = 7

# Puck attributes
PUCK_RADIUS = 15
PUCK_SPEED_X = 4
PUCK_SPEED_Y = 4

# Create the game window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Two-Player Air Hockey")

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Two actions: left or right
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fc(x)

class QLearningAgent:
    def __init__(self, learning_rate=0.00005, discount_factor=0.99, exploration_prob=1.0, exploration_decay=0.998):
        self.model = QNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay

    def select_action(self, state):
        if random.random() < self.exploration_prob:
            return random.randint(0, 1) # Random action
        with torch.no_grad():
            q_values = self.model(torch.FloatTensor(state))
            return torch.argmax(q_values).item() # Greedy action

    def train(self, state, action, reward, next_state, done):
        with torch.no_grad():
            target_q_values = self.model(torch.FloatTensor(next_state))
            max_target_q_value = torch.max(target_q_values)

        q_values = self.model(torch.FloatTensor(state))
        target_q_value = reward + (1 - done) * self.discount_factor * max_target_q_value
        target_q_values = q_values.clone().detach()
        target_q_values[action] = target_q_value

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if done:
            self.exploration_prob *= self.exploration_decay

agent1 = QLearningAgent()
agent2 = QLearningAgent()

def calculate_state_vector(paddle_x, puck_x, puck_y, puck_dx, puck_dy):
    state_vector = [0] * 6
    
    # Check puck position relative to paddle
    if paddle_x + PADDLE_WIDTH // 2 < puck_x:
        state_vector[0] = 1  # Ball is on the right
    else:
        state_vector[1] = 1  # Ball is on the left
        
    # Check puck direction
    if puck_dy < 0:  # Ball is moving upwards
        if puck_dx > 0:
            state_vector[2] = 1  # Ball is moving north-east
        else:
            state_vector[3] = 1  # Ball is moving north-west
    else:  # Ball is moving downwards
        if puck_dx > 0:
            state_vector[4] = 1  # Ball is moving south-east
        else:
            state_vector[5] = 1  # Ball is moving south-west
            
    return state_vector


def play(agent1, agent2):
    clock = pygame.time.Clock()
    running = True
    player1_score = 0
    player2_score = 0

    puck_x = WINDOW_WIDTH // 2
    puck_y = WINDOW_HEIGHT // 2
    puck_dx = random.choice([-1, 1]) * PUCK_SPEED_X
    puck_dy = random.choice([-1, 1]) * PUCK_SPEED_Y

    player1_x = WINDOW_WIDTH // 2 - PADDLE_WIDTH // 2
    player2_x = WINDOW_WIDTH // 2 - PADDLE_WIDTH // 2
    player1_y = 10
    player2_y = WINDOW_HEIGHT - PADDLE_HEIGHT - 10

    hits = 0
    hits_lst = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get pressed keys
        reward_1 = 0
        reward_2 = 0
        
        # Player 1 movement
        state_vector1 = calculate_state_vector(player1_x, puck_x, puck_y, puck_dx, puck_dy)
        action1 = agent1.select_action(state_vector1)
        if action1 == 0:
            player1_x -= PADDLE_SPEED
        else:
            player1_x += PADDLE_SPEED

        # Player 2 movement
        state_vector2 = calculate_state_vector(player2_x, puck_x, puck_y, puck_dx, puck_dy)
        action2 = agent2.select_action(state_vector2)
        if action2 == 0:
            player2_x -= PADDLE_SPEED
        else:
            player2_x += PADDLE_SPEED

        # Update puck position based on velocity
        puck_x += puck_dx
        puck_y += puck_dy

        # Paddle boundary limits
        player1_x = max(0, min(WINDOW_WIDTH - PADDLE_WIDTH, player1_x))
        player2_x = max(0, min(WINDOW_WIDTH - PADDLE_WIDTH, player2_x))

        # Puck boundary limits and collision with walls
        if puck_x <= 0 or puck_x >= WINDOW_WIDTH:
            puck_dx *= -1
        if puck_y <= 0 or puck_y >= WINDOW_HEIGHT:
            puck_dy *= -1

        # Puck collision with paddles
        if (
            player1_y + PADDLE_HEIGHT >= puck_y - PUCK_RADIUS
            and player1_x <= puck_x <= player1_x + PADDLE_WIDTH
        ):
            puck_dy *= -1.01
            reward_1 = 10
            hits += 1

        if (
            player2_y <= puck_y + PUCK_RADIUS
            and player2_x <= puck_x <= player2_x + PADDLE_WIDTH
        ):
            puck_dy *= -1.01
            reward_2 = 10
            hits += 1

        # Drawing
        screen.fill(WHITE)
        pygame.draw.rect(screen, BLACK, (player1_x, player1_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(screen, BLACK, (player2_x, player2_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.circle(screen, BLACK, (puck_x, puck_y), PUCK_RADIUS)

        if puck_y <= 0:
            player2_score += 1
            reward_1 = -30
            hits_lst.append(hits)
            hits = 0
            puck_x = WINDOW_WIDTH // 2
            puck_y = WINDOW_HEIGHT // 2
            puck_dx = random.choice([-1, 1]) * PUCK_SPEED_X
            puck_dy = random.choice([-1, 1]) * PUCK_SPEED_Y
        elif puck_y >= WINDOW_HEIGHT:
            player1_score += 1
            reward_2 = -30
            hits_lst.append(hits)
            hits = 0
            puck_x = WINDOW_WIDTH // 2
            puck_y = WINDOW_HEIGHT // 2
            puck_dx = random.choice([-1, 1]) * PUCK_SPEED_X
            puck_dy = random.choice([-1, 1]) * PUCK_SPEED_Y

        # Display scores
        font = pygame.font.Font(None, 36)
        score_text = font.render(
            f"Player 1: {player1_score}   Player 2: {player2_score}", True, BLACK
        )
        screen.blit(score_text, (10, 10))
        
        if player1_score == 5:
            reward_1 = 50
            running = False

        if player2_score == 5:
            reward_2 = 50
            running = False
        
        pygame.display.flip()
        next_state_vector1 = calculate_state_vector(player1_x, puck_x, puck_y, puck_dx, puck_dy)
        agent1.train(state_vector1, action1, reward_1, next_state_vector1, running)
        next_state_vector2 = calculate_state_vector(player2_x, puck_x, puck_y, puck_dx, puck_dy)
        agent2.train(state_vector2, action2, reward_2, next_state_vector2, running)

        if not running:
            return hits_lst

        # Limit frame rate
        clock.tick(60)

for i in range(100):
    hits_lst = play(agent1, agent2)
    print("ROUND :", i, ":", hits_lst)
pygame.quit()
sys.exit()
