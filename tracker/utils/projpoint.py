import cv2 as cv
import math
import json
import numpy as np


class projecter():
    def __init__(self,path):
        data = self.read_json(path)
        # camera resolution 
        self.U = data['U'] # orizzontale
        self.V = data['V'] # verticale
        
        # focal length in mm
        self.f = data['f'] 

        # camera sensor width and height in mm
        self.s_w = data['sw'] 
        self.s_h = data['sh'] 

        # camera position in real-world referement system (y depth, z height)
        self.xc = data['xc']
        self.yc = data['yc']
        self.zc = data['zc']
        self.camera_position = np.array([
            self.xc,
            self.yc,
            self.zc
        ]).T


        # camera rotations (  * math.pi/180 to have them in rad)
        self.yaw =    data['thyaw']     #  * math.pi/180  
        self.roll =   data['throll']    #  * math.pi/180 
        self.pitch =  data['thpitch']   #  * np.pi/180 
        
        
        # transforming focal length in pixel unit
        self.fpux = int((self.f  * self.U/self.s_w))
        self.fpuy = int((self.f  * self.V/self.s_h))

        # fov
        self.fovh = 2 * math.atan(self.s_w/(2*self.f)) * (180/math.pi)
        self.fovv = 2 * math.atan(self.s_h/(2*self.f)) * (180/math.pi)
        
        # camera matrix / intrinsic matrix
        self.K = np.matrix([[self.fpux, 0, self.U/2],
                            [0, self.fpuy, self.V/2],
                            [0, 0, 1]
                            ],dtype=np.float64)

        # id,x1,y1,x2,y2; in real-world coordinates
        self.lines = data['lines']



    def read_json(self,path):
        try:
            with open(path,'r') as file:
                data = json.load(file)
                return data
        except Exception as e:
            print(e)
            print("FILE NOT FOUND, loading default camera.json")
            self.default = True
            return self.read_json("./config/configuration_example.txt")


    def get_rotations(self):
        """Convert Euler angles to rotation matrix"""
        # Create rotation matrix from Euler angles
        pitch   =   self.pitch
        roll    =   self.roll
        yaw     =   self.yaw


        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch), math.cos(pitch)]
        ])
        
        Ry = np.array([
            [math.cos(roll), 0, math.sin(roll)],
            [0, 1, 0],
            [-math.sin(roll), 0, math.cos(roll)]
        ])

        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])

        R = Rx @ Rz @ Ry

        R = np.vstack((R, [0, 0, 0]))  # Add the row at the bottom
        R = np.hstack((R, np.array([[0], [0], [0], [1]])))  # Add the column at the right

        return R 
    
    def proj_point(self,R,point):
        """ Point projection using homogenuos coordinates """
        t = np.eye(4)
        t[:3, 3] = self.camera_position

        point = np.array([point[0],point[1],point[2],1])
        point = (R @ t) @ point

        x = point[0]/point[1]
        y = point[2]/point[1]

        u = int(x * self.fpux + self.U/2)
        v = int(y * self.fpuy + self.V/2)

        return (u,v)       


    def get_projected_points_with_perpendiculars(self):
        if self.default:
            return {1: [(786, 2167),(2140, 2317),(1425, 2209),(1484, 2101)], 2: [(1982, 2375),(927, 2117),(1409, 2209),(1326, 2317)], \
            3: [(703, 2267),(1168, 2458),(977, 2375),(1093, 2275)]}

        proj_lines = {}

        R = self.get_rotations()


        for line in self.lines:
            l_id = line['id']            
            
            point_3d1  = np.array([line['x2'],line['y2'],0], dtype=np.float32)
            point_3d2  = np.array([line['x1'],line['y1'],0], dtype=np.float32)
            u,v = self.proj_point(R,point_3d1)
            u1,v1 = self.proj_point(R,point_3d2)
            
            arrow_start, arrow_end = self.get_perpendiculars(u,v,u1,v1)
            proj_lines[l_id] = [(u,v),(u1,v1),arrow_start,arrow_end]
        
        return proj_lines    

    def get_perpendiculars(self,lsx,lsy,lex,ley):
            center_point = (
                (lsx + lex) // 2,
                (lsy + ley) // 2
            )

            # Displacement vector
            dx = lex - lsx
            dy = ley - lsy

            # Normalized displacement vector
            length = (dx ** 2 + dy ** 2) ** 0.5
            if length != 0:
                dx /= length
                dy /= length

            # Equivalent to 90 degree rotation in 2D
            perp_dx = -dy
            perp_dy = dx

            
            arrow_length = 30
            arrow_start = (
                int(center_point[0]),
                int(center_point[1])
            )

            # translation plus scaling
            arrow_end = (
                int(center_point[0] + perp_dx * arrow_length),
                int(center_point[1] + perp_dy * arrow_length)
            )

            return arrow_start,arrow_end

    


if __name__ == '__main__':

    img = cv.imread("./test20241212.png")
    proiett = projecter('./camera.json')
    
    res = proiett.get_projected_points_with_perpendiculars()


    for line in res.items():
        line_id,points = line
        x_start, y_start = points[0]
        x_end, y_end = points[1]
        
        print(x_start,y_start)
        print(x_end,y_end)
        cv.putText(img,str(line[0]),line[1][0],1,1,0,2)
        cv.line(img, (x_start,y_start ) ,(x_end,y_end ) , (0, 255, 0), 2)
        cv.circle(img,(x_start,y_start),10,3,3,1)
        cv.circle(img,(860,350),10,3,3,1)
        cv.circle(img,(750,1050),10,3,3,1)
        cv.circle(img,(1500,820),10,3,3,1)
        cv.circle(img,(1550,415),10,3,3,1)
        cv.circle(img,(x_end,y_end),10,3,3,1)
        

    cv.imshow("image",img)
    cv.waitKey(0)
    cv.destroyAllWindows()