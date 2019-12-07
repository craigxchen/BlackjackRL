'''
Blackjack environment for reinforcement learning agent
'''
import numpy as np
import random

class Blackjack:
    def __init__(self):
        self.action_space = [0, 1] # hit = 1, stand = 0
        self.state_space = [(x, y, True) for x in range(12,22) for y in range(1,11)] + [(x, y, False) for x in range(4,22) for y in range(1, 11)]
        
    def step(self, action):
        assert action in self.action_space
        if action == 1: # hit  
            self.player.append(self.deal_card())
            if self.is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else: # stay
            done = True
            self.dealer_plays()
            reward = self.winner(self.score(self.player), self.score(self.dealer))
        return self.get_playerstate(), reward, done 
    
    def reset(self):
        '''
        also functions as start
        '''
        self.player = [self.deal_card(), self.deal_card()]
        self.dealer = [self.deal_card(), self.deal_card()]
        
        return self.get_playerstate()
    
    def dealer_plays(self):
        # hit on soft 17
        if self.sum_hand(self.dealer) == 17 and self.usable_ace(self.dealer):
                self.dealer.append(self.deal_card())
            
        while self.sum_hand(self.dealer) < 17:
            self.dealer.append(self.deal_card())
        return
    
    def future_states(self, state):
        curr_hand = self.player
        new_hands = [curr_hand + [card] for card in list(range(1,11)) + 3*[10]]
        states = [(self.sum_hand(x), state[1], state[2]) for x in new_hands]
        rewards = []
        for s in states:
            if self.is_bust(self.player):
                rewards.append(-1)
            else:
                rewards.append(0)
        return states, rewards
       
    # all methods below were taken from OpenAI Gym's blackjack environment
    # ALL credit for code below goes to OpenAI 
    @staticmethod
    def winner(player, dealer):
        return (player > dealer) - (dealer > player)
    
    @staticmethod
    def deal_card():
        return random.choice(list(range(1,11)) + 3*[10])

    def get_playerstate(self):
        return (self.sum_hand(self.player), self.dealer[0], self.usable_ace(self.player))
    
    @staticmethod
    def usable_ace(hand):  # Does this hand have a usable ace?
        return 1 in hand and sum(hand) + 10 <= 21
    
    def sum_hand(self, hand):  # Return current hand total
        if self.usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)
    
    def is_bust(self, hand):  # Is this hand a bust?
        return self.sum_hand(hand) > 21
    
    def score(self, hand):  # What is the score of this hand (0 if bust)
        return 0 if self.is_bust(hand) else self.sum_hand(hand)
