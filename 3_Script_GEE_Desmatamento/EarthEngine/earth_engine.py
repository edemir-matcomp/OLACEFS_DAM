import ee
import pandas as pd
import unidecode
import time

#Open CSV with mapbiomas alerts
data = pd.read_csv("file_mapbiomas.csv")
data = data.values

# Initialize the Earth Engine
ee.Initialize()

#Only if it is necessary to change Earth Engine account
ee.Authenticate()

dates = [['2019-01-01', '2019-03-31'], ['2019-04-01', '2019-06-30'], ['2019-07-01', '2019-09-30'], ['2019-10-01', '2019-12-31'], ['2020-01-01', '2020-03-31']]

count = 1
exception_list = []
exception_list2 = []

for al in range(0, data.shape[0]):
    for day in dates:
        
        if count % 100 == 0:
            print("waiting...")
            f = open("exception_list_problem.txt","w+")
            for task in exception_list2:
                f.write(task.__repr__()+"\n")
            f.close()
            time.sleep(15*60)
        count += 1
        
        #Load the Sentinel-1 ImageCollection and filter to get images with VV and VH dual polarization
        sar = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.eq('instrumentMode', 
            'IW')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 
            'VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 
            'VH'))
    
        #Filter to get images from different look angles and filter dates
        vhAscending = sar.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).filterDate(day[0],day[1])
        vhDescending = sar.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')).filterDate(day[0],day[1])
        
        #Create a composite from means at different polarizations and look angles.
        image = ee.Image.cat(
        [ee.ImageCollection(vhAscending.select('VH').merge(vhDescending.select('VH'))).mean(),
        ee.ImageCollection(vhAscending.select('VV').merge(vhDescending.select('VV'))).mean()]).focal_median()
        
        #Get polygon from csv file
        llx = data[al][1]
        lly = data[al][2]
        urx = data[al][3]
        ury = data[al][4]

        #Get alert id and city name of the alert
        id_alerta = data[al][6]
        muni = data[al][7]
        muni = unidecode.unidecode(muni)
        
        #Create Earth Engine polygon
        polygon = list([(llx,lly),(llx,ury),(urx,ury),(urx,lly)])
        geometry = ee.Geometry.Polygon(polygon)
        region = polygon
        im = image.clip(geometry)
    
        file_name = str(id_alerta) + '_' + muni + '_' + day[0]
        print(file_name)
    
        task_config = { 
        'scale' : 10,
        'region': region,
        'folder':'GEE_mapbiomas',
        'fileFormat':'GeoTIFF',
        'skipEmptyTiles' : True, 
        'maxPixels' : 1e13
        }
        
        try:
            #Send the task for the engine.
            task = ee.batch.Export.image.toDrive(im, file_name, **task_config)
            task.start()
            print("Task sent ")
        except:
            print("Error sending this task")
            exception_list.append(task)
            exception_list2.append(file_name)
            print("Size exception_list: ", len(exception_list))
            

#Dealing with the exception list            
print("writing tasks list in a file...")
f = open("initial_list_problem.txt","w+")
for task in exception_list:
    f.write(task.__repr__()+"\n")
f.close()


list_round = []
flag_wait = 0

while len(exception_list) != 0:
    try:
        print(exception_list[0])
        exception_list[0].start()
        exception_list.pop(0)

        if flag_wait > 2:
            print("writing tasks list in a file...")
            f = open("final_list_problem.txt","w+")
            for task in exception_list:
                f.write(task.__repr__()+"\n")
            f.close()
            exit()

    except:
        print("using exception...")
        first = exception_list[0]
        exception_list.pop(0)
        exception_list.append(first)

        list_round.append(len(exception_list))

        if len(list_round) > 10:
            list_round.pop(0)

        if (len(set(list_round))==1):
            print("waiting...")
            print(list_round)
            time.sleep(60*60) 
            flag_wait += 1


print("script is done!")




        

