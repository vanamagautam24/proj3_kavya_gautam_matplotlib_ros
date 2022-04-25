import math
import numpy as np
import matplotlib.pyplot as plt

from heapq import heappush, heappop

class Arena():
    def __init__(self, width=40, height=25, r=1, c=1, threshold=0.01,
                 thetaStep=30, actions=None, wheelLength=1,
                 wheelRadius=1):
        self.threshold = threshold
        self.r = r
        self.c = c
        self.thetaStep = thetaStep
        self.visited_ = np.zeros([int(height/threshold) + 1, int(width/threshold) + 1, 4])
        self.Actions = np.zeros([int(height/threshold) + 1, int(width/threshold) + 1])
        self.x_val = []
        self.y_val = []
        self.plotData_A = []
        self.selected_action = []
        plt.ion()
        self.fig, self.ax = plt.subplots()
        plt.axis('square')
        self.DrawMap()
        self.actions = actions
        self.rad_w = wheelRadius
        self.wheelLength = wheelLength

    def DrawMap(self):
        centX, centY, radii = 30, 18.5, 4
        circle_1_X = [centX+radii*math.cos(i)
                      for i in np.arange(0, 2*3.14, 0.01)]
        circle_1_Y = [centY+radii*math.sin(i)
                      for i in np.arange(0, 2*3.14, 0.01)]
        p1 = [[10.5-0, 10.0-0], [3.6-0, 18.5-0],
              [11.5-0, 21.0-0], [8.0-0, 18.0-0], [10.5-0, 10.0-0]]
        xs, ys = zip(*p1)
        self.ax.plot(xs, ys,'g')
        p2 = [[16.5-0, 12.0-0], [20.0-0, 14.0-0], [23.5-0, 12.0-0],
              [23.5-0, 8.0-0], [20.0-0, 6.0-0], [16.5-0, 8.0-0], [16.5-0, 12.0-0]]
        xs, ys = zip(*p2)
        self.ax.plot(xs, ys,'g')
        self.ax.plot(circle_1_X, circle_1_Y,'g')
        self.ax.set_xlim(0, 40)
        self.ax.set_ylim(0, 25)
        pass
    def Draw_(self, Xi, Yi, Thetai, rpm1, rpm2, color="blue", lw=0.5):
        t = 0
        dt = 0.1
        Xn = Xi
        Yn = Yi
        tdash = 3.14 * Thetai / 180
        x_, x_n, y_s, y_n = [], [], [], []
        while t < 1:
            t = t + dt
            Xs = Xn
            Ys = Yn
            Xn += 0.5 * self.rad_w * (rpm1 + rpm2) * math.cos(tdash) * dt
            Yn += 0.5 * self.rad_w * (rpm1 + rpm2) * math.sin(tdash) * dt
            tdash += (self.rad_w / self.wheelLength) * (rpm2 - rpm1) * dt
            x_.append(Xs)
            x_n.append(Xn)
            y_s.append(Ys)
            y_n.append(Yn)
        self.ax.plot([x_, x_n], [y_s, y_n], color=color, linewidth=lw)
    def DrawPath(self, path, trackIndex):
        print(len(trackIndex), len(path))
        for i in range(len(path)):
            actionIndex = int(trackIndex[i])
            self.Draw_(path[i][1], path[i][2], path[i][3], self.actions[actionIndex][0], self.actions[actionIndex][1], color="red", lw=1.2)
            plt.pause(0.00001)
        plt.ioff()
        plt.show(block=False)
        pass

    def DrawExplore(self):
        for i in range(len(self.x_val)):
            actionIndex = self.selected_action[i]
            self.Draw_(self.x_val[i], self.y_val[i], self.plotData_A[i], self.actions[actionIndex][0], self.actions[actionIndex][1])
            if i % 100 == 0:
                plt.pause(0.000001)
        pass

    def InsideMap(self, i, j):
        return(not (i < (5 - self.r - self.c) and
                i > (-0 + self.r + self.c) and
                j < (5 - self.r - self.c) and
                j > (-0+self.r + self.c)))

    def IsObstacle(self, i, j):
        p1 = [[10.5-0, 10.0-0], [3.6-0, 18.5-0],
              [11.5-0, 21.0-0], [8.0-0, 18.0-0], [10.5-0, 10.0-0]]
        pm1 = [((p1[2][1] - p1[1][1])/(p1[2][0] - p1[1][0])), ((p1[3][1] - p1[2][1]) / (p1[3][0] - p1[2][0])), ((p1[1][1] - p1[4][1]) /
                                                                                                                (p1[1][0] - p1[4][0])), ((p1[4][1] - p1[3][1]) / (p1[4][0] - p1[3][0])), ((p1[3][1] - p1[1][1]) / (p1[3][0] - p1[1][0]))]
        p2 = [[16.5-0, 12.0-0], [20.0-0, 14.0-0], [23.5-0, 12.0-0],
              [23.5-0, 8.0-0], [20.0-0, 6.0-0], [16.5-0, 8.0-0], [16.5-0, 12.0-0]]
        pm2 = (p2[2][1] - p2[6][1]) / (p2[2][0] - p2[6][0])
        clearance = 0.1
        if (i < clearance) or (i > 40-clearance) or (j < 0) or (j > 25-clearance):
            return False
        elif(((pm1[0] * i - pm1[0]*p1[2][0] + p1[2][1] + clearance - j) >= 0) and ((pm1[1] * i - pm1[1]*p1[2][0] + p1[2][1] - clearance - j) <= 0) and ((pm1[4] * i - pm1[4] * p1[3][0] + p1[3][1] - j) <= 0)) or (((pm1[2] * i - pm1[2]*p1[4][0] + p1[4][1] - clearance - j) <= 0) and ((pm1[3] * i - pm1[3]*p1[4][0] + p1[4][1] + clearance - j) >= 0) and not((pm1[4] * i - pm1[4] * p1[3][0] + p1[3][1] - j) <= 0)):
            return False
        elif(((-pm2 * i - (-pm2 * p2[4][0]) + p2[4][1] - clearance - j) <= 0) and
             ((pm2 * i - (pm2 * p2[4][0]) + p2[4][1] - clearance - j) <= 0) and
             ((-pm2 * i - (-pm2 * p2[1][0]) + p2[1][1] + clearance - j) >= 0) and
             ((pm2 * i - (pm2 * p2[1][0]) + p2[1][1] + clearance - j) >= 0) and
             ((p2[2][0] + clearance - i) >= 0) and
             ((p2[6][0] - clearance - i) <= 0)):
            return False
        # Circle
        elif (((i - 30) ** 2 + (j - 18.5) ** 2) <= (4 + self.r + self.c) ** 2):
            return False
        else:
            return True

    def MatId(self, node):
        x, y, a = node[1], node[2], node[3]
        shiftx, shifty = 5, 0
        x += shiftx
        y = abs(shifty + y)
        i = int(round(y/self.threshold))
        j = int(round(x/self.threshold))
        k = int(round(a/self.thetaStep))
        return i, j

    def IsVisited(self, node):
        i, j = self.MatId(node)
        return self.visited_[i, j, 3] != 0

    def GetVisited(self, node):
        i, j = self.MatId(node)
        return self.visited_[i, j, :], self.Actions[i, j]

    def SetVisited(self, node, parent, action):
        i, j = self.MatId(node)
        self.Actions[i, j] = action
        self.selected_action.append(action)
        self.x_val.append(parent[1])
        self.y_val.append(parent[2])
        self.plotData_A.append(parent[3])
        self.visited_[i, j, :] = np.array(parent)
        return




class Astar():
    def __init__(self, initial, goal, thetaStep=30, stepSize=1, goalThreshold=0.1,
                 width=40, height=25, threshold=0.5, r=0.1, c=0.1, wheelLength=0.038,
                 Ur=2, Ul=2, wheelRadius=2, dt=0.1, dtheta=0, showExploration=0, showPath=1):
        self.initial = initial
        self.goal = goal
        self.nodeData = []
        self.Data = []
        self.thetaStep = thetaStep
        self.dt = dt
        self.dtheta = dtheta
        self.rob_wheel = wheelRadius
        self.wheelLength = wheelLength
        self.Ur = Ur
        self.Ul = Ul
        self.stepSize = stepSize
        self.goalThreshold = goalThreshold
        self.path = []
        self.trackIndex = []
        self.goalReach = False
        self.moves = [[0, self.Ur],
                        [self.Ul, 		 0],
                        [0, self.Ul],
                        [self.Ur, 		 0],
                        [self.Ul, self.Ur],
                        [self.Ur, self.Ul],
                        [self.Ur, self.Ur],
                        [self.Ul, self.Ul]]
        self.actions = []
        self.obstacle = Arena(width, height, r=r, c=c, threshold=threshold,
                              actions=self.moves, wheelLength=self.wheelLength,
                              wheelRadius=self.rob_wheel)
        self.showExploration = showExploration
        self.showPath = showPath    

    def SetUp(self):
        if not self.obstacle.IsObstacle(self.goal[0], self.goal[1]):
            print("Invalid Goal")
            return False
        elif not self.obstacle.IsObstacle(self.initial[0], self.initial[1]):
            print("Invalid Start")
            return False
        else:
            cost = math.sqrt(
                (self.initial[0] - self.goal[0])**2 + (self.initial[1] - self.goal[1])**2)
            heappush(self.Data, [cost, self.initial[0],
                     self.initial[1], self.initial[2], 0])
            self.nodeData.append(
                [self.initial[0], self.initial[1], self.initial[2], 0])
            return True
    def GetMoves(self, currentN):
        self.actions = []
        idx = 0
        for m in self.moves:
            t = 0
            dt = 0.1
            x, y, A = currentN[1], currentN[2], currentN[3]
            A = 3.14*A/180.0
            costtc = 0
            for i in range(10):
                t = t+dt
                xnew = 0.5*(self.rob_wheel) * \
                    (m[0]+m[1])*math.cos(A)*dt
                ynew = 0.5*(self.rob_wheel) * \
                    (m[0]+m[1])*math.sin(A)*dt
                x += xnew
                y += ynew
                A += (self.rob_wheel/self.wheelLength) * \
                    (m[1]-m[0])*dt
                costtc += math.sqrt(xnew**2 + ynew**2)
            A = 180 * (A) / 3.14
            self.actions.append([x, y, A, costtc, idx])
            idx += 1
        return
    def Backtrack(self, current):
        track = []
        trackIndex = []
        curr = current[:4]
        track.append(curr)
        trackIndex.append(0)
        while curr[1:] != self.initial:
            l, ind = self.obstacle.GetVisited(curr)
            curr = list(l)
            track.append(curr)
            trackIndex.append(ind)
        print("Generating path")
        track.reverse()
        trackIndex.reverse()
        return track, trackIndex

    def GetPath(self):
        if self.SetUp():
            while len(self.Data) > 0:
                presentNode = heappop(self.Data)
                previousCost, ctc_old = presentNode[0], presentNode[4]
                if ((presentNode[1] - self.goal[0])**2 + (presentNode[2] - self.goal[1])**2 <= (self.goalThreshold)**2):
                    self.goalReach = True
                    print(" Goal Reached ")
                    self.path, self.trackIndex = self.Backtrack(presentNode)
                    if self.showExploration:
                        self.obstacle.DrawExplore()
                    if self.showPath:
                        self.obstacle.DrawPath(self.path, self.trackIndex)
                    return
                self.GetMoves(presentNode)
                for action in self.actions:
                    newNode = [0, action[0], action[1], action[2], 0]
                    ctc_new = ctc_old + action[3]
                    newNode[4] = ctc_new
                    costToGo = 1.3 * math.sqrt((newNode[1] - self.goal[0])** 2 + (newNode[2] - self.goal[1])**2)
                    if self.obstacle.IsObstacle(action[0], action[1]):
                        if not self.obstacle.IsVisited(newNode):
                            presentNode[0] = ctc_new
                            self.obstacle.SetVisited(
                                newNode, presentNode[:4], action[4])
                            newNode[0] = ctc_new + costToGo
                            heappush(self.Data, newNode)
                        else:
                            previousVisited, _ = self.obstacle.GetVisited(
                                newNode)
                            previousCost = previousVisited[0]
                            if previousCost > ctc_new:
                                presentNode[0] = ctc_new
                                self.obstacle.SetVisited(
                                    newNode, presentNode[:4], action[4])
        print(" Goal Unreachable")
        return


if __name__ == '__main__':
    x = float(input('Enter Start x coordinate: '))
    y = float(input('Enter Start y coordinate: '))
    theta_start = float(input('Enter theta start: '))
    start = [x, y, theta_start]
    x = float(input('Enter Goal x coordinate: '))
    y = float(input('Enter Goal y coordinate: '))
    end = [x, y, 0]

    # start = [5, 5, 0]
    # end = [20, 20, 0]
    robot_radius = float(0.177)
    clearance = float(2.1)
    StepSize = int(2)
    Threshold = float(0.01)
    GoalThreshold = float(0.1)
    print(start)
    print(end)
    wheelLength = float(0.320)
    rpm1 = float(input('Enter RPM R: '))
    prm2 = float(input('Enter RPM L: '))
    rad_w = float(0.038)
    weight = float(1.3)
    solver = Astar(start, end, stepSize=StepSize,
                   goalThreshold=GoalThreshold, threshold=Threshold,
                   r=robot_radius, c=clearance, wheelLength=wheelLength, Ur=rpm1, Ul=prm2, wheelRadius=rad_w,
                   showExploration=int(1), showPath=int(1))
    solver.GetPath()
    print(solver.trackIndex)
    l = wheelLength
    robot_radius = rad_w
    for idx in solver.trackIndex:
        ul, ur = solver.moves[int(idx)]
        vx = robot_radius*0.5*(ur+ul)
        rz = robot_radius*(ur-ul)/l
        print(ul, ur, vx, rz)
    print("Done")
