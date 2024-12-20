import numpy as np
from PIL import Image
import cv2
import pickle

size = 10

no_episodes = 25000
move_penalty = 1
enemy_penalty = 300
food_reward = 25
epsilon = 0.9
epsilon_decay = 0.9998  # Every episode will be epsilon*EPS_DECAY
show_every = 3000  # how often to play through env visually.

start_q_table = None  # None or Filename

learning_rate = 0.1
discount = 0.95

player_n = 1  # player key in dict
food_n = 2  # food key in dict
enemy_n = 3  # enemy key in dict

# the dict!
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > size - 1:
            self.x = size - 1
        if self.y < 0:
            self.y = 0
        elif self.y > size - 1:
            self.y = size - 1


if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for i in range(-size + 1, size):
        for ii in range(-size + 1, size):
            for iii in range(-size + 1, size):
                for iiii in range(-size + 1, size):
                    q_table[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(4)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

# can look up from Q-table with: print(q_table[((-9, -2), (3, 9))]) for example

episode_rewards = []

for episode in range(no_episodes):
    player = Blob()
    food = Blob()
    enemy = Blob()
    if episode % show_every == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player - food, player - enemy)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        # Take the action!
        player.action(action)

        #### MAYBE ###
        enemy.move()
        food.move()
        ##############

        if player.x == enemy.x and player.y == enemy.y:
            reward = -enemy_penalty
        elif player.x == food.x and player.y == food.y:
            reward = food_reward
        else:
            reward = -move_penalty

        # first we need to obs immediately after the move.
        new_obs = (player - food, player - enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == food_reward:
            new_q = food_reward
        else:
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((size, size, 3), dtype=np.uint8)  # starts an rbg of our size
            env[food.x][food.y] = d[food_n]  # sets the food location tile to green color
            env[player.x][player.y] = d[player_n]  # sets the player tile to blue
            env[enemy.x][enemy.y] = d[enemy_n]  # sets the enemy location to red
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            if reward == food_reward or reward == -enemy_penalty:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == food_reward or reward == -enemy_penalty:
            break

    # print(episode_reward)
    epsilon *= epsilon_decay
