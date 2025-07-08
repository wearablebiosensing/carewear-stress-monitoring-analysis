# Importing Module
import wearablehrv

# downloading some example data
path = wearablehrv.data.download_data_and_get_path()
# downloading some example data
path = wearablehrv.data.download_data_and_get_path()
# Define the participant ID 
pp = "test" 
# Define your experimental conditions, for instance, sitting, standing, walking, and biking
conditions = ['sitting', 'standing', 'walking', 'biking'] 

# Define the devices you want to validate against the criterion. 
devices = ["kyto", "heartmath", "rhythm", "empatica", "vu"] 

# Redefine the name of the criterion device
criterion = "vu" 

# Read data, experimental events, and segment the continuous data into smaller chunks
data = wearablehrv.individual.import_data (path, pp, devices)
events = wearablehrv.individual.define_events (path, pp, conditions, already_saved= True, save_as_csv= False)
data_chopped = wearablehrv.individual.chop_data (data, conditions, events, devices)
wearablehrv.individual.visual_inspection (data_chopped, devices, conditions,criterion)
