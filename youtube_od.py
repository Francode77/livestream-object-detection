import cv2
import pafy
import streamlit as st
import yolov5
import torchvision
import torch 
import time 
import wget
import socket   

# Determine on which platform the app runs, for model loading
def check_host():
    hostname=socket.gethostname()   
    IPAddr=socket.gethostbyname(hostname)     
    if '192.168.' in IPAddr:
        host='local'
    else:
        host='streamlit_cloud'
    return host

def wait_area():
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

# Function to display video from button_click
def open_stream(x,stream_link,title):
    st.session_state.stream_link=stream_link
    st.subheader(f"Stream {x+1}")
    st.text(title)
    display_video(model,stream_link)

# Function to draw the sidebar
def draw_sidebar():
    thumbs=[]
    videos=[]
    titles=[] 
    urls=[
    'https://www.youtube.com/watch?v=hIJs5p7ND0g',
    'https://www.youtube.com/watch?v=1-iS7LArMPA',
    'https://www.youtube.com/watch?v=CvOB-Is_yYU'
    ]

    for x,url in enumerate(urls):
        # Use pafy to get videostream from youtube
        videos.append(pafy.new(url))

        # Getting thumbnail of the video
        thumbs.append(videos[x].thumb)
        titles.append(videos[x].title) 

    # Create a sidebar for overview    
    st.sidebar.title("Youtube streams") 
    for x, (thumb, title) in enumerate(zip(thumbs,titles)):
        st.sidebar.image(thumb) 
        st.sidebar.button(f"Stream {x+1}",on_click=open_stream,kwargs=dict(x=x,stream_link=urls[x],title=title))


# Initial run will not have model, so declare it here
if 'model' not in st.session_state:
    
    # Use GPU if available, else use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    boxes = torch.tensor([[0., 1., 2., 3.]]).to(device)
    scores = torch.randn(1).to(device)
    iou_thresholds = 0.5

    # YoloV5
    # Check whether the app is run locally or on streamlit cloud,
    # On streamlit cloud we have to download the model because of permission errors
    if check_host()=='local':
        # load pretrained model locally
        model = yolov5.load('/model/yolov5s.pt')
    else:
        # load pretrained model on streamlit
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5s.pt', force_reload=True) 
    #model.to(device)   
    st.session_state['model'] = model
    draw_sidebar()

# Load the model from session_state
model=st.session_state.model 

st.title('Livestream object detection')

def display_video(model,stream_url):
 
    draw_sidebar()

    videostream=stream_url
    video=pafy.new(videostream)
 
    # Search for a 640x360 resolution 
    desired_resolution = None
    for stream in video.allstreams:
        if stream.resolution == '640x360':
            desired_resolution = stream
            break  

    # Capture with OpenCV
    capture = cv2.VideoCapture(desired_resolution.url)
    capture.set(cv2.CAP_PROP_FPS, 10)
    fps = capture.get(cv2.CAP_PROP_FPS)

    # Display frames in same window
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

# Catch the stop button
if st.button('Stop'):
    st.stop()
 

