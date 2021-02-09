FROM tensorflow/tensorflow:devel-gpu
WORKDIR /root/
RUN apt update
RUN apt install -y git nano
RUN touch /root/.nanorc
RUN echo "set tabsize 4" >> ~/.nanorc
RUN echo "set tabstospaces" >> ~/.nanorc
RUN pip install 'imageio==2.4.0'
RUN  pip install ipython
RUN  pip install real_robots
RUN  pip install tf-agents[reverb]
RUN  pip install gast==0.3.3
RUN  pip install numpy==1.19.2
RUN  pip install six==1.15.0
RUN  pip install opencv-python

