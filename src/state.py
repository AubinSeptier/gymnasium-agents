import copy

class OthelloState:
    def __init__(self, board=None, current_player=1):
        
        if board is None:
            # Initialize a standard Othello board or an empty one
            self.board = [[0]*8 for _ in range(8)]
            self.board[3][3] = -1
            self.board[3][4] = 1
            self.board[4][3] = 1
            self.board[4][4] = -1
        else:
            self.board = board
        self.current_player = current_player

    def clone(self):
        return OthelloState(board=copy.deepcopy(self.board),
                            current_player=self.current_player)

    def get_legal_actions(self):

        moves = []
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == 0 and self.is_legal_move(r, c):
                    moves.append((r, c))
        if not moves:
            moves.append(None)  
        return moves

    def is_legal_move(self, row, col):
        
        if self.board[row][col] != 0:
            return False
        return True  

    def move(self, action):

        new_state = self.clone()
        if action is None:
            
            new_state.current_player = -self.current_player
            return new_state

        row, col = action
        new_state.board[row][col] = self.current_player
        new_state.current_player = -self.current_player
        return new_state

    def is_game_over(self):
        if any(a is not None for a in self.get_legal_actions()):
            return False
        self.current_player = -self.current_player
        has_moves = any(a is not None for a in self.get_legal_actions())
        self.current_player = -self.current_player
        return not has_moves

    def game_result(self):
        black = sum(cell == 1 for row in self.board for cell in row)
        white = sum(cell == -1 for row in self.board for cell in row)
        if black > white: return 1
        elif white > black: return -1
        else: return 0
