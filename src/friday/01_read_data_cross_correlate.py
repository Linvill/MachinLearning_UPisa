import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.colors as colors
from obspy import read
import glob
import os



# load data
catalog = pd.read_csv('C:/Users/Linus/University of Pisa/Machine Learning in Geosciences - Class Materials\Day5_Project/catalogue_HQDD.csv')
stations = pd.read_csv('C:/Users/Linus/University of Pisa/Machine Learning in Geosciences - Class Materials\Day5_Project/station.csv')

catalog_lon_lat = catalog.to_numpy()
kmeans = KMeans(n_clusters=5).fit(catalog_lon_lat[:, [1, 2]])
y_kmeans = kmeans.predict(catalog_lon_lat[:, [1, 2]])
labels = kmeans.labels_


cmap = plt.cm.get_cmap('jet', max(labels)-min(labels)+1)
bounds = range(min(labels), max(labels)+2)
norm = colors.BoundaryNorm(bounds, cmap.N)

plt.figure(dpi=300)
plt.scatter(catalog_lon_lat[:, 2], catalog_lon_lat[:, 1], c=labels, s=50, cmap=cmap, norm=norm)
#plt.scatter(catalog_lon_lat[:, 2], catalog_lon_lat[:, 1], c=labels, alpha=0.3, edgecolors='none', cmap=cmap, norm=norm)
for index, row in stations.iterrows():
    plt.text(stations['longitude'][index], stations['latitude'][index], stations['id'][index])

plt.xlabel('lon [deg]')
plt.ylabel('lat [deg]')
plt.grid()
plt.xlim([-21.5, -21.3])
plt.ylim([63.9, 64.15])

cb = plt.colorbar(ticks=labels+0.5, orientation="horizontal")
cb.set_ticklabels(labels)
plt.scatter(stations['longitude'], stations['latitude'], marker="v")
#plt.legend((h_scatter), ('0', '1', '2', '3', '4'))
plt.show()

# get cluster 0 and do ccc on station '2C.THJ07', 'ON.HUMLI.00'
mask_catalog = labels == 1
mask_stations = (stations['id'] == '2C.THJ07.') | (stations['id'] == 'ON.HUMLI.00')
catalog_clus0 = catalog[mask_catalog]
stations_clus0 = stations[mask_stations]
plt.figure(dpi=300)
plt.scatter(catalog_clus0['longitude'], catalog_clus0['latitude'])
plt.scatter(stations_clus0['longitude'], stations_clus0['latitude'], marker='v')

for index, row in stations_clus0.iterrows():
    plt.text(stations_clus0['longitude'][index], stations_clus0['latitude'][index], stations_clus0['id'][index])

plt.xlabel('lon [deg]')
plt.ylabel('lat [deg]')

plt.grid()
plt.show()

# get waveforms of stations
path = 'C:/Users/Linus/University of Pisa/Machine Learning in Geosciences - Class Materials/Day5_Project/waveforms/'

folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

for index, folder in enumerate(folders):
    path_folder = os.listdir(path + folder)

    x = 2


mseed = '2C_2021-01-01T01_24_08.mseed'

st = read(path + folder + mseed)
st_2C_THJ07 = st.select(network='2C', station='THJ07', channel='EHZ')

mseed = 'OR_2021-01-01T01_24_08.mseed'
st = read(path + folder + mseed)
st_ON_HUMLI = st.select(network='ON', station='HUMLI', channel='HHZ')


def pre_prc_waveform(stream):
    stream.detrend('linear')
    stream.taper(max_percentage=0.05, type="hann")
    stream.filter("bandpass", freqmin=2, freqmax=30)



x = 2