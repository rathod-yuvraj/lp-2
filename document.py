class Node:
    def __init__(self,data,level,fval):
        """ Initialize the node with the data, level of the node and the calculated fvalue """
        self.data = data
        self.level = level
        self.fval = fval

    def generate_child(self):
        """ Generate child nodes from the given node by moving the blank space
            either in the four directions {up,down,left,right} """
        x,y = self.find(self.data,'_')
        """ val_list contains position values for moving the blank space in either of
            the 4 directions [up,down,left,right] respectively. """
        val_list = [[x,y-1],[x,y+1],[x-1,y],[x+1,y]]
        children = []
        for i in val_list:
            child = self.shuffle(self.data,x,y,i[0],i[1])
            if child is not None:
                child_node = Node(child,self.level+1,0)
                children.append(child_node)
        return children
        
    def shuffle(self,puz,x1,y1,x2,y2):
        """ Move the blank space in the given direction and if the position value are out
            of limits the return None """
        if x2 >= 0 and x2 < len(self.data) and y2 >= 0 and y2 < len(self.data):
            temp_puz = []
            temp_puz = self.copy(puz)
            temp = temp_puz[x2][y2]
            temp_puz[x2][y2] = temp_puz[x1][y1]
            temp_puz[x1][y1] = temp
            return temp_puz
        else:
            return None
            

    def copy(self,root):
        """ Copy function to create a similar matrix of the given node"""
        temp = []
        for i in root:
            t = []
            for j in i:
                t.append(j)
            temp.append(t)
        return temp    
            
    def find(self,puz,x):
        """ Specifically used to find the position of the blank space """
        for i in range(0,len(self.data)):
            for j in range(0,len(self.data)):
                if puz[i][j] == x:
                    return i,j


class Puzzle:
    def __init__(self,size):
        """ Initialize the puzzle size by the specified size,open and closed lists to empty """
        self.n = size
        self.open = []
        self.closed = []

    def accept(self):
        """ Accepts the puzzle from the user """
        puz = []
        for i in range(0,self.n):
            temp = input().split(" ")
            puz.append(temp)
        return puz

    def f(self,start,goal):
        """ Heuristic Function to calculate hueristic value f(x) = h(x) + g(x) """
        return self.h(start.data,goal)+start.level

    def h(self,start,goal):
        """ Calculates the different between the given puzzles """
        temp = 0
        for i in range(0,self.n):
            for j in range(0,self.n):
                if start[i][j] != goal[i][j] and start[i][j] != '_':
                    temp += 1
        return temp
        

    def process(self):
        """ Accept Start and Goal Puzzle state"""
        print("Enter the start state matrix \n")
        start = self.accept()
        print("Enter the goal state matrix \n")        
        goal = self.accept()

        start = Node(start,0,0)
        start.fval = self.f(start,goal)
        """ Put the start node in the open list"""
        self.open.append(start)
        print("\n\n")
        while True:
            cur = self.open[0]
            print("")
            print("  | ")
            print("  | ")
            print(" \\\'/ \n")
            for i in cur.data:
                for j in i:
                    print(j,end=" ")
                print("")
            """ If the difference between current and goal node is 0 we have reached the goal node"""
            if(self.h(cur.data,goal) == 0):
                break
            for i in cur.generate_child():
                i.fval = self.f(i,goal)
                self.open.append(i)
            self.closed.append(cur)
            del self.open[0]

            """ sort the opne list based on f value """
            self.open.sort(key = lambda x:x.fval,reverse=False)


puz = Puzzle(3)
puz.process()
# ////////////////////////////////////////////////////////////////////////
class Node:
    def __init__(self,data,level,fval):
        """ Initialize the node with the data, level of the node and the calculated fvalue """
        self.data = data
        self.level = level
        self.fval = fval

    def generate_child(self):
        """ Generate child nodes from the given node by moving the blank space
            either in the four directions {up,down,left,right} """
        x,y = self.find(self.data,'_')
        """ val_list contains position values for moving the blank space in either of
            the 4 directions [up,down,left,right] respectively. """
        val_list = [[x,y-1],[x,y+1],[x-1,y],[x+1,y]]
        children = []
        for i in val_list:
            child = self.shuffle(self.data,x,y,i[0],i[1])
            if child is not None:
                child_node = Node(child,self.level+1,0)
                children.append(child_node)
        return children
        
    def shuffle(self,puz,x1,y1,x2,y2):
        """ Move the blank space in the given direction and if the position value are out
            of limits the return None """
        if x2 >= 0 and x2 < len(self.data) and y2 >= 0 and y2 < len(self.data):
            temp_puz = []
            temp_puz = self.copy(puz)
            temp = temp_puz[x2][y2]
            temp_puz[x2][y2] = temp_puz[x1][y1]
            temp_puz[x1][y1] = temp
            return temp_puz
        else:
            return None
            

    def copy(self,root):
        """ Copy function to create a similar matrix of the given node"""
        temp = []
        for i in root:
            t = []
            for j in i:
                t.append(j)
            temp.append(t)
        return temp    
            
    def find(self,puz,x):
        """ Specifically used to find the position of the blank space """
        for i in range(0,len(self.data)):
            for j in range(0,len(self.data)):
                if puz[i][j] == x:
                    return i,j


class Puzzle:
    def __init__(self,size):
        """ Initialize the puzzle size by the specified size,open and closed lists to empty """
        self.n = size
        self.open = []
        self.closed = []

    def accept(self):
        """ Accepts the puzzle from the user """
        puz = []
        for i in range(0,self.n):
            temp = input().split(" ")
            puz.append(temp)
        return puz

    def f(self,start,goal):
        """ Heuristic Function to calculate hueristic value f(x) = h(x) + g(x) """
        return self.h(start.data,goal)+start.level

    def h(self,start,goal):
        """ Calculates the different between the given puzzles """
        temp = 0
        for i in range(0,self.n):
            for j in range(0,self.n):
                if start[i][j] != goal[i][j] and start[i][j] != '_':
                    temp += 1
        return temp
        

    def process(self):
        """ Accept Start and Goal Puzzle state"""
        print("Enter the start state matrix \n")
        start = self.accept()
        print("Enter the goal state matrix \n")        
        goal = self.accept()

        start = Node(start,0,0)
        start.fval = self.f(start,goal)
        """ Put the start node in the open list"""
        self.open.append(start)
        print("\n\n")
        while True:
            cur = self.open[0]
            print("")
            print("  | ")
            print("  | ")
            print(" \\\'/ \n")
            for i in cur.data:
                for j in i:
                    print(j,end=" ")
                print("")
            """ If the difference between current and goal node is 0 we have reached the goal node"""
            if(self.h(cur.data,goal) == 0):
                break
            for i in cur.generate_child():
                i.fval = self.f(i,goal)
                self.open.append(i)
            self.closed.append(cur)
            del self.open[0]

            """ sort the opne list based on f value """
            self.open.sort(key = lambda x:x.fval,reverse=False)


puz = Puzzle(3)
puz.process()
# //////////////////////////////////////////////////////
INF = 9999999
# number of vertices in graph
N = 5
#creating graph by adjacency matrix method
G = [[0, 19, 5, 0, 0],[19, 0, 5, 9, 2],[5, 5, 0, 1, 6],[0, 9, 1, 0, 1],[0, 2, 6, 1, 0]]
selected_node = [0, 0, 0, 0, 0]
no_edge = 0
selected_node[0] = True
# printing for edge and weight print("Edge : Weight\n")
while (no_edge < N - 1):
    minimum = INF
    a = 0
    b = 0
    for m in range(N):
        if selected_node[m]:
            for n in range(N):
                if ((not selected_node[n]) and G[m][n]):
                    # not in selected and there is an edge
                    if minimum > G[m][n]:
                        minimum = G[m][n]
                        a = m
                        b = n
                        print(str(a) + "-" + str(b) + ":" + str(G[a][b]))
                        selected_node[b] = True
                        no_edge += 1

OUTPUT
0-1:19
0-2:5
1-4:2
2-3:1
# //////////////////////////////////////////////////////////
""" Assignment No – 05 (Group B) Program to an elementary catboat for Simple
Question and Answering Application """
print("Simple Question and Answering Program")
print("=====================================")
print(" You may ask any one of these questions")
print("Hi")
print("How are you?")
print("Are you working?")
print("What is your name?")
print("what did you do yesterday?")
print("Quit")
while True:
    question = input("Enter one question from above list:")
    question = question.lower()
    if question in ['hi']:
         print("Hello")
    elif question in ['how are you?','how do you do?']:
         print("I am fine")
    elif question in ['are you working?','are you doing any job?']:
         print("yes. I'am working in KLU")
    elif question in ['what is your name?']:
         print("My name is Emilia")
         name=input("Enter your name?")
         print("Nice name and Nice meeting you",name)
    elif question in ['what did you do yesterday?']:
         print("I saw Bahubali 5 times")
    elif question in ['quit']:
         break
    else:
         print("I don't understand what you said")

OUTPUT
Simple Question and Answering Program
=====================================
 You may ask any one of these questions
Hi
How are you?
Are you working?
What is your name?
what did you do yesterday?
Quit
Enter one question from above list:Hi
Hello
Enter one question from above list:HOW are you?
I am fine
Enter one question from above list:
# //////////////////////////////////////////////////
from copy import deepcopy

class puzzle:
    def __init__ (self, starting, parent):
        self.board = starting
        self.parent = parent
        self.f = 0
        self.g = 0
        self.h = 0

    def manhattan(self):
        inc = 0
        h = 0
        for i in range(3):
            for j in range(3):
                h += abs(inc-self.board[i][j])
            inc += 1
        return h


    def goal(self):
        inc = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != inc:
                    return False
                inc += 1
        return True

    def __eq__(self, other):
        return self.board == other.board

def move_function(curr):
    curr = curr.board
    for i in range(3):
        for j in range(3):
            if curr[i][j] == 0:
                x, y = i, j
                break
    q = []
    if x-1 >= 0:
        b = deepcopy(curr)
        b[x][y]=b[x-1][y]
        b[x-1][y]=0
        succ = puzzle(b, curr)
        q.append(succ)
    if x+1 < 3:
        b = deepcopy(curr)
        b[x][y]=b[x+1][y]
        b[x+1][y]=0
        succ = puzzle(b, curr)
        q.append(succ)
    if y-1 >= 0:
        b = deepcopy(curr)
        b[x][y]=b[x][y-1]
        b[x][y-1]=0
        succ = puzzle(b, curr)
        q.append(succ)
    if y+1 < 3:
        b = deepcopy(curr)
        b[x][y]=b[x][y+1]
        b[x][y+1]=0
        succ = puzzle(b, curr)
        q.append(succ)

    return q

def best_fvalue(openList):
    f = openList[0].f
    index = 0
    for i, item in enumerate(openList):
        if i == 0: 
            continue
        if(item.f < f):
            f = item.f
            index  = i

    return openList[index], index

def AStar(start):
    openList = []
    closedList = []
    openList.append(start)

    while openList:
        current, index = best_fvalue(openList)
        if current.goal():
            return current
        openList.pop(index)
        closedList.append(current)

        X = move_function(current)
        for move in X:
            ok = False   #checking in closedList
            for i, item in enumerate(closedList):
                if item == move:
                    ok = True
                    break
            if not ok:              #not in closed list
                newG = current.g + 1 
                present = False

                #openList includes move
                for j, item in enumerate(openList):
                    if item == move:
                        present = True
                        if newG < openList[j].g:
                            openList[j].g = newG
                            openList[j].f = openList[j].g + openList[j].h
                            openList[j].parent = current
                if not present:
                    move.g = newG
                    move.h = move.manhattan()
                    move.f = move.g + move.h
                    move.parent = current
                    openList.append(move)

    return None


#start = puzzle([[2,3,6],[0,1,8],[4,5,7]], None)
start = puzzle([[5,2,8],[4,1,7],[0,3,6]], None)
# start = puzzle([[0,1,2],[3,4,5],[6,7,8]], None)
#start = puzzle([[1,2,0],[3,4,5],[6,7,8]], None)
result = AStar(start)
noofMoves = 0

if(not result):
    print ("No solution")
else:
    print(result.board)
    t=result.parent
    while t:
        noofMoves += 1
        print(t.board)
        t=t.parent
print ("Length: " + str(noofMoves))

OUTPUT:
[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
[[1, 0, 2], [3, 4, 5], [6, 7, 8]]
[[1, 4, 2], [3, 0, 5], [6, 7, 8]]
[[1, 4, 2], [3, 5, 0], [6, 7, 8]]
[[1, 4, 2], [3, 5, 8], [6, 7, 0]]
[[1, 4, 2], [3, 5, 8], [6, 0, 7]]
[[1, 4, 2], [3, 5, 8], [0, 6, 7]]
[[1, 4, 2], [0, 5, 8], [3, 6, 7]]
[[0, 4, 2], [1, 5, 8], [3, 6, 7]]
[[4, 0, 2], [1, 5, 8], [3, 6, 7]]
[[4, 5, 2], [1, 0, 8], [3, 6, 7]]
[[4, 5, 2], [0, 1, 8], [3, 6, 7]]
[[0, 5, 2], [4, 1, 8], [3, 6, 7]]
[[5, 0, 2], [4, 1, 8], [3, 6, 7]]
[[5, 2, 0], [4, 1, 8], [3, 6, 7]]
[[5, 2, 8], [4, 1, 0], [3, 6, 7]]
[[5, 2, 8], [4, 1, 7], [3, 6, 0]]
[[5, 2, 8], [4, 1, 7], [3, 0, 6]]
[[5, 2, 8], [4, 1, 7], [0, 3, 6]]
Length: 18
# ///////////////////////////////////////////////////////////////
# DFS algorithm in Python


# DFS algorithm
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)

    print(start)

    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited


graph = {'0': set(['1', '2']),
         '1': set(['0', '3', '4']),
         '2': set(['0']),
         '3': set(['1']),
         '4': set(['2', '3'])}

dfs(graph, '0')
 
OUTPUT:
0
1
3
4
2
2
{'0', '1', '2', '3', '4'}


# BFS algorithm in Python


import collections

# BFS algorithm
def bfs(graph, root):

    visited, queue = set(), collections.deque([root])
    visited.add(root)

    while queue:

        # Dequeue a vertex from queue
        vertex = queue.popleft()
        print(str(vertex) + " ", end="")

        # If not visited, mark it as visited, and
        # enqueue it
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)


if __name__ == '__main__':
    graph = {0: [1, 2], 1: [2], 2: [3], 3: [1, 2]}
    print("Following is Breadth First Traversal: ")
    bfs(graph, 0)

OUTPUT:
Following is Breadth First Traversal: 
0 1 2 3 
# ////////////////////////////////////////////////////////////
""" Python3 program to solve N Queen Problem 
using Branch or Bound """

N = 8

""" A utility function to print solution """
def printSolution(board): 
	for i in range(N): 
		for j in range(N): 
			print(board[i][j], end = " ") 
		print() 

""" A Optimized function to check if 
a queen can be placed on board[row][col] """
def isSafe(row, col, slashCode, backslashCode, 
		rowLookup, slashCodeLookup, 
					backslashCodeLookup): 
	if (slashCodeLookup[slashCode[row][col]] or
		backslashCodeLookup[backslashCode[row][col]] or
		rowLookup[row]): 
		return False
	return True

""" A recursive utility function 
to solve N Queen problem """
def solveNQueensUtil(board, col, slashCode, backslashCode, 
					rowLookup, slashCodeLookup, 
					backslashCodeLookup): 
						
	""" base case: If all queens are 
	placed then return True """
	if(col >= N): 
		return True
	for i in range(N): 
		if(isSafe(i, col, slashCode, backslashCode, 
				rowLookup, slashCodeLookup, 
				backslashCodeLookup)): 
					
			""" Place this queen in board[i][col] """
			board[i][col] = 1
			rowLookup[i] = True
			slashCodeLookup[slashCode[i][col]] = True
			backslashCodeLookup[backslashCode[i][col]] = True
			
			""" recur to place rest of the queens """
			if(solveNQueensUtil(board, col + 1, 
								slashCode, backslashCode, 
								rowLookup, slashCodeLookup, 
								backslashCodeLookup)): 
				return True
			
			""" If placing queen in board[i][col] 
			doesn't lead to a solution,then backtrack """
			
			""" Remove queen from board[i][col] """
			board[i][col] = 0
			rowLookup[i] = False
			slashCodeLookup[slashCode[i][col]] = False
			backslashCodeLookup[backslashCode[i][col]] = False
			
	""" If queen can not be place in any row in 
	this column col then return False """
	return False

""" This function solves the N Queen problem using 
Branch or Bound. It mainly uses solveNQueensUtil()to 
solve the problem. It returns False if queens 
cannot be placed,otherwise return True or 
prints placement of queens in the form of 1s. 
Please note that there may be more than one 
solutions,this function prints one of the 
feasible solutions."""
def solveNQueens(): 
	board = [[0 for i in range(N)] 
				for j in range(N)] 
	
	# helper matrices 
	slashCode = [[0 for i in range(N)] 
					for j in range(N)] 
	backslashCode = [[0 for i in range(N)] 
						for j in range(N)] 
	
	# arrays to tell us which rows are occupied 
	rowLookup = [False] * N 
	
	# keep two arrays to tell us 
	# which diagonals are occupied 
	x = 2 * N - 1
	slashCodeLookup = [False] * x 
	backslashCodeLookup = [False] * x 
	
	# initialize helper matrices 
	for rr in range(N): 
		for cc in range(N): 
			slashCode[rr][cc] = rr + cc 
			backslashCode[rr][cc] = rr - cc + 7
	
	if(solveNQueensUtil(board, 0, slashCode, backslashCode, 
						rowLookup, slashCodeLookup, 
						backslashCodeLookup) == False): 
		print("Solution does not exist") 
		return False
		
	# solution found 
	printSolution(board) 
	return True

# Driver Code 
solveNQueens() 


output:
1 0 0 0 0 0 0 0 
0 0 0 0 0 0 1 0 
0 0 0 0 1 0 0 0 
0 0 0 0 0 0 0 1 
0 1 0 0 0 0 0 0 
0 0 0 1 0 0 0 0 
0 0 0 0 0 1 0 0 
0 0 1 0 0 0 0 0 
True
# ////////////////////////////////////////////////////////////////////////////////
import heapq

class GameState:
    def __init__(self, position, cost=0):
        self.position = position
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

class AStar:
    def __init__(self, start, goal, obstacles):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.open_list = []
        self.closed_set = set()
        self.parent = {}
        self.g_score = {start: 0}

    def heuristic(self, current):
        # A simple Manhattan distance heuristic for illustration purposes
        return abs(current[0] - self.goal[0]) + abs(current[1] - self.goal[1])

    def get_neighbors(self, current):
        neighbors = []
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Possible moves: right, left, down, up

        for move in moves:
            neighbor = (current[0] + move[0], current[1] + move[1])

            # Check if the neighbor is within the bounds of the grid and not an obstacle
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] != '*':
                neighbors.append(neighbor)

        return neighbors

    def run(self):
        heapq.heappush(self.open_list, GameState(self.start, self.heuristic(self.start)))

        while self.open_list:
            current_state = heapq.heappop(self.open_list)

            if current_state.position == self.goal:
                return self.reconstruct_path(current_state.position)

            self.closed_set.add(current_state.position)

            for neighbor in self.get_neighbors(current_state.position):
                if neighbor in self.closed_set:
                    continue

                tentative_g_score = self.g_score[current_state.position] + 1

                if neighbor not in self.open_list or tentative_g_score < self.g_score[neighbor]:
                    self.g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor)
                    heapq.heappush(self.open_list, GameState(neighbor, f_score))
                    self.parent[neighbor] = current_state.position

        return None  # No path found

    def reconstruct_path(self, current):
        path = [current]
        while current in self.parent:
            current = self.parent[current]
            path.append(current)
        return path[::-1]

# Example usage:
grid = [
    "S . * . .",
    ". * . . .",
    ". * * * .",
    ". . . . G"
]

start_position = (0, 0)
goal_position = (3, 4)
obstacle_positions = [(0, 2), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

a_star = AStar(start_position, goal_position, obstacle_positions)
result = a_star.run()

if result:
    print("Shortest Path:", result)
else:
    print("No path found.")
# ///////////////////////////////////////////////////////////
import heapq

class GameState:
    def __init__(self, position, cost=0):
        self.position = position
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

class AStar:
    def __init__(self, start, goal, obstacles):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.open_list = []
        self.closed_set = set()
        self.parent = {}
        self.g_score = {start: 0}

    def heuristic(self, current):
        # A simple Manhattan distance heuristic for illustration purposes
        return abs(current[0] - self.goal[0]) + abs(current[1] - self.goal[1])

    def get_neighbors(self, current):
        neighbors = []
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Possible moves: right, left, down, up

        for move in moves:
            neighbor = (current[0] + move[0], current[1] + move[1])

            # Check if the neighbor is within the bounds of the grid and not an obstacle
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] != '*':
                neighbors.append(neighbor)

        return neighbors

    def run(self):
        heapq.heappush(self.open_list, GameState(self.start, self.heuristic(self.start)))

        while self.open_list:
            current_state = heapq.heappop(self.open_list)

            if current_state.position == self.goal:
                return self.reconstruct_path(current_state.position)

            self.closed_set.add(current_state.position)

            for neighbor in self.get_neighbors(current_state.position):
                if neighbor in self.closed_set:
                    continue

                tentative_g_score = self.g_score[current_state.position] + 1

                if neighbor not in self.open_list or tentative_g_score < self.g_score[neighbor]:
                    self.g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor)
                    heapq.heappush(self.open_list, GameState(neighbor, f_score))
                    self.parent[neighbor] = current_state.position

        return None  # No path found

    def reconstruct_path(self, current):
        path = [current]
        while current in self.parent:
            current = self.parent[current]
            path.append(current)
        return path[::-1]

# Example usage:
grid = [
    "S . * . .",
    ". * . . .",
    ". * * * .",
    ". . . . G"
]

start_position = (0, 0)
goal_position = (3, 4)
obstacle_positions = [(0, 2), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

a_star = AStar(start_position, goal_position, obstacle_positions)
result = a_star.run()

if result:
    print("Shortest Path:", result)
else:
    print("No path found.")
# ///////////////////////////////////////////////////////////////////////
