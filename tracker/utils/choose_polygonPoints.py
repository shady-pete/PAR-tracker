import argparse
from shapely.geometry.polygon import Polygon
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseEvent
import sys
import os

clicked_points = []

def onclick(event):
    if isinstance(event, MouseEvent) and event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        print(f"({x}, {y}),")
        clicked_points.append((x, y))
        ax.plot(x, y, 'ro')

        if len(clicked_points) > 1:
            x0, y0 = clicked_points[-2]
            x1, y1 = clicked_points[-1]
            ax.plot([x0, x1], [y0, y1], 'r-')

        fig.canvas.draw()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Helper tool to define polygon in an image.')
    parser.add_argument('--input', type=str, default="../../video.mp4",
                      help='Path to the input video file (default: ../video.mp4)')
    args = parser.parse_args()
    video_path= args.input
    video_path= os.path.abspath( os.path.join(os.path.dirname(__file__), video_path) )
    POLYGON= Polygon([
        (22, 343),
        (562, 205),
        (683, 285),
        (816, 240),
        (1239, 442),
        (862, 681),
        (78, 686),
        (25, 535),
        (21, 343)
    ])
    
    cap= cv.VideoCapture(video_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    success, image= cap.read()
    
    frame_width= int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height= int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    if not success:
        print("An error occurred")
        sys.exit(1)
    
    for index in range(len(POLYGON.exterior.xy[0])-1):
        pt1= (
            int( POLYGON.exterior.xy[0][index] ),
            int( POLYGON.exterior.xy[1][index] )
        )
        pt2= (
            int( POLYGON.exterior.xy[0][index+1] ),
            int( POLYGON.exterior.xy[1][index+1] )
        )
        
        cv.line(image, pt1, pt2, (0,255,0), 2)

    print("Points:")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    ax.axis('off')
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()