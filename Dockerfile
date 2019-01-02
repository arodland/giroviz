FROM continuumio/miniconda3

RUN apt-get update && apt-get install unzip

RUN conda create -y -q -n my_cartopy_env -c conda-forge cartopy

ENV PATH /opt/conda/envs/my_cartopy_env/bin:$PATH

RUN echo "conda activate my_cartopy_env" >> ~/.bashrc

RUN conda install -y -q pandas -n my_cartopy_env

RUN pip install geojsoncontour

RUN wget -O /tmp/ne_110m_land.zip http://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip
WORKDIR /root/.local/share/cartopy/shapefiles/natural_earth/physical
RUN unzip /tmp/ne_110m_land.zip

WORKDIR /
COPY contour.py /
