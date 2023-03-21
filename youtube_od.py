import cv2
import pafy
import streamlit as st
import yolov5
import torchvision
import torch 
import time 
import wget
     
@st.cache
def loadModel():
    start_dl = time.time()
    model_file = wget.download('https://github.com/Francode77/livestream-object-detection/blob/main/model/yolov5s.pt', out="model/")
    finished_dl = time.time()
    print(f"Model Downloaded, ETA:{finished_dl-start_dl}")
loadModel()


variable = None

st.title('Livestream object detection')

def set_variable(value):
    global variable
    variable = value

if variable:
    st.write("Variable is set to:", variable)
    # run code with variable

# Use GPU if available, else use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
boxes = torch.tensor([[0., 1., 2., 3.]]).to(device)
scores = torch.randn(1).to(device)
iou_thresholds = 0.5

# YoloV5

# load pretrained model locally
#model = yolov5.load('/model/yolov5s.pt')

# load pretrained model on streamlit
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5s.pt', force_reload=True) 
model.to(device)  

values=[]
videos=[]
x=int(0)

urls=['https://www.youtube.com/watch?v=hIJs5p7ND0g',
'https://www.youtube.com/watch?v=1-iS7LArMPA',
'https://www.youtube.com/watch?v=CvOB-Is_yYU']

videostream=urls[0]

for url in urls:
    # Use pafy to get videostream from youtube
    videos.append(pafy.new(url))
    # getting thumbnail of the video
    values.append(videos[x].thumb)
    x+=1

# Create a sidebar for overview    
x=int(0)
st.sidebar.title("Youtube streams") 
for value in values:
    st.sidebar.image(value)
    if st.sidebar.button(f"Stream {x}",x):
        set_variable(urls[x]) 
    x+=1

# Catch the stop button
if st.button('Stop'):
    st.stop()

# Wait for input, if a button is pressed, variable will have the value of the clicked button
if variable:

    videostream=variable
    video=pafy.new(videostream)

    # Print all available streams
    streams = video.allstreams 
    for s in streams:
        print(s)

    #best  = video.getbest() 
    #print (best.resolution, best.extension)  

    # Search for a resolution
    desired_resolution = None
    for s in video.allstreams:
        if s.resolution == '640x360':
            desired_resolution = s
            break

    if desired_resolution:
        print(desired_resolution.resolution)
        #best_resolution.download()
    else:
        print("No video stream matches the specified resolution")

    # Capture with OpenCV
    capture = cv2.VideoCapture(desired_resolution.url)
    capture.set(cv2.CAP_PROP_FPS, 10)
    fps = capture.get(cv2.CAP_PROP_FPS)

    # Print the frames in same window
    image_place = st.empty()
    capture = cv2.VideoCapture(desired_resolution.url)
    while True:
        check, frame = capture.read() 
        frame = cv2.resize(frame, (1280, 720), interpolation = cv2.INTER_LINEAR)
        results=model(frame)
        # Parse results
        predictions = results.pred[0]
        boxes = predictions[:, :4] # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5] 
        names=results.pandas().xyxy[0]['name'] # Pandas series
        objects=len(predictions)
        object_labels=names.value_counts().to_dict()

        # Draw the bounding boxes on the frame
        for box, score, category, name in zip(boxes, scores, categories,names):
            x1, y1, x2, y2 = box
            x1=int(x1)
            x2=int(x2)
            y1=int(y1)
            y2=int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(frame, (20, 5), (20 + 230, 5 + 30 * (len(object_labels)+2)+5), (0,0,0), -1)
            cv2.putText(frame, f"{name} {score.item():.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255, 255, 255), 3)
            cv2.putText(frame, f"FPS: {fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255, 255, 255), 3)
            cv2.putText(frame, f"Objects detected: {objects}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255, 255, 255), 3)
            object_label_counter=0
            for index,object_label in enumerate(object_labels):
                cv2.putText(frame, f"{object_label}: {object_labels[object_label]}", (30, 90+ object_label_counter), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255, 255, 255), 3)
                object_label_counter+=30
            
        # Show the frame with bounding boxes
        image_place.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        #time.sleep(.05)
    
    capture.release()
    cv2.destroyAllWindows() 

else:
    with st.empty(): 
        while True:
            st.write(f"⏳ Please choose a stream")
            time.sleep(0.31)
            st.write(f"⏳ Please choose a stream .")
            time.sleep(0.31)
            st.write(f"⏳ Please choose a stream ..")
            time.sleep(0.31)
            st.write(f"⏳ Please choose a stream ...")
            time.sleep(0.31)
            st.write(f"⏳ Please choose a stream ....")

st.text(torch.cuda.is_available())
st.text(torch.__version__)
st.text(torch.version.cuda)
st.text(torchvision.__version__)

"""True
1.13.1+cu117
11.7
0.14.1+cu117
"""

