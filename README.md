# GuessExamGrade
This is basically a DETR model for facial expression capture and emotion recognition.

#For a faster setup follow:<br>
https://github.com/AdrianHuang2002/GuessExamPerformance/tree/master<br>
With python installed<br>
Step 1: <br>
Create a python virtual environment <br>
You can follow <br>
https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/

Step 2: <br>
git clone https://github.com/facebookresearch/detr.git <br>
Meta's end-to-end objection detection with transformer <br>
And their face detection model: <br>
https://drive.google.com/file/d/1F_flSdZQNCb3URiiZMpSrU_aQe7bktmS/view?usp=drive_link (put into detr directory) <br>

Step 3: <br>
move files from src into detr (not putting src the folder into detr) <br>
and go into detr

Step 4: <br>
install necessary libraries

Step 5: <br>
run <br>
python app.py <br>
in the terminal and then following the link you can locally access the webpage.
Note the result is not accurate, we didn't have time to train model on people's 
actual reactions on taking exams or checking answers. Feel free to play around with the numbers. 

Step 6: <br>
Since the model goes through videos frame-by-frame, it may take a few minutes <br>
or even longer depending on the size of your files <br>
Since the front-end webpage part is incomplete, it doesn't play the processed video after loading your videos.<br>
You can go to detr/static/processed to view the videos after evaluated by the <br>
model

