from typing import Tuple
import numpy as np
import time
import os


class Congkak:
    def __init__(self, display: bool) -> None:
        self.board = np.full(16, 7, np.int8)
        self.board[0], self.board[8] = 0, 0
        self.display = display

    def two_agent_step(self, n: int, m: int) -> Tuple[np.ndarray, float, float,
                                                      bool, int]:
        n = self.offset_action(n, 1)
        m = self.offset_action(m, 2)
        assert 9 <= n <= 15
        assert 1 <= m <= 7

        score1 = self.board[0]
        score2 = self.board[8]
        step1 = 0
        step2 = 0
        end1 = False
        end2 = False
        while not (end1 and end2):
            balls1 = 0
            if not end1:
                balls1 = self.board[n]
                self.board[n] = 0
            balls2 = 0
            if not end2:
                balls2 += self.board[m]
                self.board[m] = 0
            while balls1 + balls2 > 0:
                if balls1 > 0:
                    n = (n + 1) % 16
                    if n == 8:
                        n = (n + 1) % 16
                    self.board[n] += 1
                    balls1 -= 1
                    step1 += 1
                    if balls1 == 0:
                        if self.board[n] == 1 or n == 0:
                            end1 = True
                        else:
                            balls1 += self.board[n]
                            self.board[n] = 0
                if balls2 > 0:
                    m = (m + 1) % 16
                    if m == 0:
                        m = (m + 1) % 16
                    self.board[m] += 1
                    balls2 -= 1
                    step2 += 1
                    if balls2 == 0:
                        if self.board[m] == 1 or m == 8:
                            end2 = True
                        else:
                            balls2 += self.board[m]
                            self.board[m] = 0
                if self.display:
                    os.system("clear")
                    print(f"Player 1 balls: {balls1} Player 2 balls: {balls2}")
                    print(self)
                    time.sleep(0.2)

        reward1 = self.board[0] - score1
        reward2 = self.board[8] - score2

        if n == 0 and m == 8:
            info = 0
        elif n == 0 and m != 8:
            info = 1
        elif n != 0 and m == 8:
            info = 2
        else:
            info = 0 * (step1 == step2) + 1 * (step1 > step2) + 2 * (step1 <
                                                                     step2)

        return self.board.copy().astype(np.float32), reward1, reward2, \
                self.get_done(), info

    def step(self, n: int, player: int) -> Tuple[np.ndarray, float, bool, int]:
        n = self.offset_action(n, player)
        assert (1 <= n <= 7) or (9 <= n <= 15)
        
        if player == 1:
            player_home = 0
            skip_index = 8
            enemy = 2
        else:
            player_home = 8
            skip_index = 0
            enemy = 1

        score = self.board[player_home]
        end = False
        can_eat = False
        while not end:
            balls = self.board[n]
            self.board[n] = 0
            while balls > 0:
                n = (n + 1) % 16
                if n == skip_index:
                    n = (n + 1) % 16
                if n == player_home:
                    can_eat = True
                self.board[n] += 1
                balls -= 1
                if balls == 0 and (self.board[n] == 1 or n == player_home):
                    end = True
                if self.display:
                    os.system("clear")
                    print(f"Player {player} balls: {balls}")
                    print(self)
                    time.sleep(0.2)

        if can_eat and (((1 <= n <= 7) and player == 2) or ((9 <= n <= 15) and
                                                            player == 1)):
            target_entry = 16 - n
            self.board[player_home] += self.board[target_entry] + 1
            self.board[target_entry] = 0
            self.board[n] = 0
            if self.display:
                os.system("clear")
                print(self)
                time.sleep(0.2)

        reward = self.board[player_home] - score
        info = player if n == player_home else enemy

        return self.board.copy().astype(np.float32), reward, self.get_done(), \
                info

    def get_done(self) -> bool:
        return not (self.have_valid_action(1) and self.have_valid_action(2))

    def reset(self, display: bool) -> np.ndarray:
        self.__init__(display)
        return self.board.copy().astype(np.float32)

    def sample_valid_action(self, player: int) -> int:
        actions = []
        if player == 1:
            for i, entry in enumerate(self.board[9:16]):
                if entry != 0:
                    actions.append(i)
        else:
            for i, entry in enumerate(self.board[1:8]):
                if entry != 0:
                    actions.append(i)
        if len(actions) == 0:
            raise Exception(f"Player {player} dont have any valid action.")
        return np.random.choice(actions)

    def have_valid_action(self, player: int) -> bool:
        if player == 1:
            for entry in self.board[9:16]:
                if entry != 0:
                    return True
        else:
            for entry in self.board[1:8]:
                if entry != 0:
                    return True
        return False

    def is_valid_action(self, action: int, player: int) -> bool:
        return self.board[self.offset_action(action, player)] != 0

    @staticmethod
    def offset_action(n: int, player: int) -> int:
        return n + 9 if player == 1 else n + 1

    def __str__(self) -> str:
        output = f"Player 1 Home: {self.board[0]} Player 2 Home: {self.board[8]}"
        output += '\n'
        for i in range(1, 8):
            output += f"{self.board[i]} "
        output += '\n'
        for i in range(15, 8, -1):
            output += f"{self.board[i]} "
        return output
