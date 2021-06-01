# Ambulance CCTV Detection
Installing required library using:
```
!pip install -r requirements.txt
```

## Dataset Preparation
We are using Open-Source Framework to download the selected categorized Dataset. 

There are 4 categories that we used *categories* - (train, test, validation):
1. Ambulance - (338, 51, 12)
2. Bus - (1000, 247, 73)
3. Car - (1000, 1000, 1000)
4. Truck - (1000, 820, 269)

Dataset : [**Open Image Dataset v6**](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=segmentation&r=false)

Framework : [OIDv6](https://pypi.org/project/oidv6/).

Command Line That we used to aquire the selected categorized Dataset. 
```
!oidv6 downloader --dataset OIDv6/ --type_data all --classes Ambulance Bus Car Truck Van --limit 1000 --yes 
```

The Dataset will be Saved in the Following Structure
```
OIDv6/test/ambulance/labels
 ├── 📂test
 │ ├── 📂ambulance
 │ │ ├── 📂labels
 │ │ │ ├──image1_label.txt...image(n)_label.txt
 │ │ ├── image1.jpg...image(n).jpg
 │ ├── 📂bus
 │ ├── 📂car
 │ ├── 📂truck
 ├── 📂train
 ├── 📂validation
```

## Building the Model
 ### Training
 ### Validation
 ### Accuracy

## Testing


## Deployment
