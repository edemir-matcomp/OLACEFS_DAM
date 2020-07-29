import shapefile
import pandas as pd
import numpy as np
#Reads shp file from MapBiomas and saves informations on a csv file

COLUMNS = ['lx', 'ly', 'ux', 'uy', 'ano', 'id_alerta', 'municipio', 'bioma']
df = pd.DataFrame(columns=COLUMNS)

cont = 0
with shapefile.Reader("dashboard_alerts-shapefile.shp") as shp:

    for feature in shp.shapeRecords():
        cont = cont+1
        # Class of type 'Record', containing:
        # 'AnoDetec', 'AreaHa', 'Bioma', 'DataDetec', 'Estado', 'IDAlerta', 'Municipio'
        rec = feature.record.as_dict()
        
        # Dict Keys for each polygon/feature
        #['bbox', 'parts', 'points', 'shapeType', 'shapeTypeName']
        shape = feature.shape
        
	    
        a1 = shape.bbox[0]
        a2 = shape.bbox[1]
        a3 = shape.bbox[2]
        a4 = shape.bbox[3]

        bio = rec['Bioma']
        id_a = rec['IDAlerta']
        muni = rec['Municipio']
        ano = rec['AnoDetec']

        df = pd.DataFrame(np.array([[a1, a2, a3, a4, ano, id_a, muni, bio]]), columns=['lx', 'ly', 'ux', 'uy', 'ano', 'id_alerta', 'municipio', 'bioma']).append(df, ignore_index=True)
    df.to_csv('file_mapbiomas.csv')

