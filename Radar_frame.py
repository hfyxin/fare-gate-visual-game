class RadarFrame:
   
    def __init__(self, data: 'list[dict[str, int or float]]'):
        """
        Radar frame object contains data for a defined frame interval in lists for each attribute
        param data: a list of dictionaries
        ex. [{
                'sensorId': 2,
                'x': -280.35359191052436,
                'y': 524.516705459526,
                'z': 875.3924645059872,
                'timestamp': 1663959822542
            }, ...]
        """
        self.sensorid = []
        self.x = []
        self.y = []
        self.z = []
        self.timestamp = []
        self.isStatic = [] # -1 default, 1 static, 0 not static. checked by static points class
        for item in data: 
            self.sensorid.append(item['sensorId'])
            self.x.append(item['x'])
            self.y.append(item['y'])
            self.z.append(item['z'])
            self.timestamp.append(item['timestamp'])
            self.isStatic.append(-1) # update in main program with static points class
    
    def __str__(self): 
        return f"""
        RadarFrame object with: sensorid: {self.sensorid}, \n
        x: {self.x}, \n
        y: {self.y}, \n
        z: {self.z}, \n
        timestamp: {self.timestamp}, \n
        is static: {self.isStatic}\n
        """
   
    # check if a specified sensor is empty      
    def is_empty(self, target_sensor_id=None) -> bool:  
         # if sensor id is not passed in, check all sensors
        if target_sensor_id is None:
            return len(self.sensorid) == 0
        else:
        # if argument specifies sensor id, check data within that sensor id only
            return not any(id == target_sensor_id for id in self.sensorid)

    # getter for points list in format to be used for display 
    def get_points_for_display(self, sensor_id) -> list:
        points_list = []
        for i, id in enumerate(self.sensorid):
            if id == sensor_id:
                points_list.append((self.x[i], self.y[i], self.isStatic[i]))
        return points_list