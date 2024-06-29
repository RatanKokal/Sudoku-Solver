from make_predictions import predict_sudoku
import argparse

# convert to class structure for easy access of mat from other python file
class SudokuSolver:
    def __init__(self):
        self.mat = [[0]*9]*9

    def get_mat(self):
        return self.mat

    def set_mat(self, mat):
        self.mat = mat

    def predict_sudoku(self, image_path, model_path):
        self.mat = predict_sudoku(image_path, model_path)

    # Check if the sudoku is solvable
    def is_solvable(self):
        valid = True
        for i in range(9):
            for j in range(9):
                if self.mat[i][j] == 0:
                    continue
                else:
                    if not self.is_valid(i, j, self.mat[i][j]):
                        valid = False
                        break
        return valid
    
    # Make corrections to the sudoku in the format row, column, number
    def make_corrections(self):
        print('Any corrections to be made? If yes, enter the row, column and number to be replaced. Else, enter -1')
        try:
            row, col, num = map(int, input().split())
            row -= 1
            col -= 1
            assert row >= 0 and row < 9 and col >= 0 and col < 9 and num >= 1 and num <= 9
            self.mat[row][col] = num
        except:
            return
        print(self.mat)
        if not self.is_solvable():
            print('Invalid Sudoku. Please look carefully for changes to be made')
        self.make_corrections()

    # Solve using backtracking
    def solve(self):
        for i in range(9):
            for j in range(9):
                if self.mat[i][j] == 0:
                    for num in range(1, 10):
                        if self.is_valid(i, j, num):
                            self.mat[i][j] = num
                            if self.solve():
                                return True
                            self.mat[i][j] = 0
                    return False
        return True

    # Check if the number to be placed is valid
    def is_valid(self, row, col, num):
        for i in range(9):
            if i != col and self.mat[row][i] == num or self.mat[i][col] == num and i != row:
                return False
        start_row = row - row % 3
        start_col = col - col % 3
        for i in range(3):
            for j in range(3):
                if i + start_row != row and j + start_col != col and self.mat[i + start_row][j + start_col] == num:
                    return False
        return True
    

if __name__ == '__main__':
    sudoku = SudokuSolver()

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', 
                        help = 'Path to image of sudoku(default = \'./assets/sudoku_images/sudoku.jpg\')',
                        type = str,
                        default = './assets/sudoku_images/sudoku.jpg')
    
    parser.add_argument('--model_path',
                        help = 'Path to model(default = \'./assets/models/model.keras\')',
                        type = str,
                        default = './assets/models/model.keras')
    
    args = parser.parse_args()

    # Predict sudoku
    sudoku.predict_sudoku(args.image_path, args.model_path)

    # Corrections
    print(sudoku.get_mat())
    if not sudoku.is_solvable():
        print('Invalid Sudoku. Please look carefully for changes to be made')
    sudoku.make_corrections()
    while not sudoku.is_solvable():
        print(sudoku.get_mat())
        print('Invalid Sudoku. Please look carefully for changes to be made')
        sudoku.make_corrections()
    
    # Solve and print
    sudoku.solve()
    print(sudoku.get_mat())
    
