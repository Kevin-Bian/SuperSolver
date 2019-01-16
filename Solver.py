def check_empty_grid(grid, lst):
    for row in range(0,9):
        for column in range(0,9):
            if grid[row][column] == 0:
                lst[0]=row 
                lst[1]=column 
                return True
    return False

def in_row_or_column(grid, row, column, num, row_or_column):
    if (row_or_column == "row"):
        for index in range(9): 
            if(grid[row][index] == num): 
                return True
        return False
    else:
        for index in range(9): 
            if(grid[index][column] == num): 
                return True
        return False

def in_box(grid, row, column, num):
    for box_row in range(0,3):
        for box_column in range(0,3):
            if grid[box_row + row][box_column + column] == num:
                return True
    return False

def allowed_spot(grid, row, column, num):
    not_used_in_row = not in_row_or_column(grid, row, column, num, "row")
    not_used_in_column = not in_row_or_column(grid, row, column, num, "column")
    not_used_in_box = not in_box(grid, row - row % 3, column - column % 3, num)
    return not_used_in_row and not_used_in_column and not_used_in_box

def solve(grid):
    lst=[0,0]
    if (not check_empty_grid(grid, lst)):
        return True
    row = lst[0]
    column = lst[1]
    for num in range(1,10):
        if (allowed_spot(grid, row, column, num)):
            grid[row][column] = num
            if (solve(grid)):
                return grid
            grid[row][column] = 0
    return False


#Algorithm inspired by https://www.geeksforgeeks.org/sudoku-backtracking-7/

