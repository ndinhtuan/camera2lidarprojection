import rosbag
import pandas as pd

bag_file_path = "/media/photo/T7/data_mapathon/16082023/IKG_camera/scenario1/2023-08-16-08-57-55_0.bag"
bag = rosbag.Bag(bag_file_path, "r")

#list_timestamp = bag.read_messages(topics=["/trigger_timestamps"])
#list_timestamp = list(list_timestamp)

time_shift = 7828077.279999971
count = 1
data = {
        "image_name": [],
        "timestamp": []
        }
for (ltopic, lmsg, lt) in bag.read_messages(topics=["/trigger_timestamps"]):
    image_name = "image_{}".format(count)
    count += 1
    data["image_name"].append(image_name)
    data["timestamp"].append(lmsg.data+time_shift)

pd.DataFrame(data).to_csv("image_timestamp.csv")
