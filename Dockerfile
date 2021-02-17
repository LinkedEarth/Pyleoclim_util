FROM continuumio/miniconda3
run conda create -n pyleoclim python=3.8
RUN /bin/bash -c "source activate pyleoclim && conda install numpy && conda install -c conda-forge cartopy && pip install git+https://github.com/LinkedEarth/Pyleoclim_util.git@Development"
RUN echo "source activate pyleoclim" > ~/.bashrc
ENV PATH /opt/conda/envs/pyleoclim/bin:$PATH
