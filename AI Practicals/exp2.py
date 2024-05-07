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
