def revert_dict(coarse_to_fine_dict):
    fine_to_coarse_dict = dict()
    for key, value in coarse_to_fine_dict.items():
        for v in value:
            fine_to_coarse_dict[v] = key
    return fine_to_coarse_dict

CLASS_9_LIST = ['education', 'entertainment_arts_culture', 'facilities', 'financial', 'healthcare', 'public_service', 'sustenance', 'transportation', 'waste_management']

CLASS_118_LIST=['animal_boarding', 'animal_breeding', 'animal_shelter', 'arts_centre', 'atm', 'baby_hatch', 'baking_oven', 'bank', 'bar', 'bbq', 'bench', 'bicycle_parking', 'bicycle_rental', 'bicycle_repair_station', 'biergarten', 'boat_rental', 'boat_sharing', 'brothel', 'bureau_de_change', 'bus_station', 'cafe', 'car_rental', 'car_sharing', 'car_wash', 'casino', 'charging_station', 'childcare', 'cinema', 'clinic', 'clock', 'college', 'community_centre', 'compressed_air', 'conference_centre', 'courthouse', 'crematorium', 'dentist', 'dive_centre', 'doctors', 'dog_toilet', 'dressing_room', 'drinking_water', 'driving_school', 'events_venue', 'fast_food', 'ferry_terminal', 'fire_station', 'food_court', 'fountain', 'fuel', 'funeral_hall', 'gambling', 'give_box', 'grave_yard', 'grit_bin', 'hospital', 'hunting_stand', 'ice_cream', 'internet_cafe', 'kindergarten', 'kitchen', 'kneipp_water_cure', 'language_school', 'library', 'lounger', 'love_hotel', 'marketplace', 'monastery', 'motorcycle_parking', 'music_school', 'nightclub', 'nursing_home', 'parcel_locker', 'parking', 'parking_entrance', 'parking_space', 'pharmacy', 'photo_booth', 'place_of_mourning', 'place_of_worship', 'planetarium', 'police', 'post_box', 'post_depot', 'post_office', 'prison', 'pub', 'public_bath', 'public_bookcase', 'ranger_station', 'recycling', 'refugee_site', 'restaurant', 'sanitary_dump_station', 'school', 'shelter', 'shower', 'social_centre', 'social_facility', 'stripclub', 'studio', 'swingerclub', 'taxi', 'telephone', 'theatre', 'toilets', 'townhall', 'toy_library', 'training', 'university', 'vehicle_inspection', 'vending_machine', 'veterinary', 'waste_basket', 'waste_disposal', 'waste_transfer_station', 'water_point', 'watering_place']

CLASS_95_LIST = ['arts_centre', 'atm', 'baby_hatch', 'bank', 'bar', 'bbq', 'bench', 'bicycle_parking', 'bicycle_rental', 'bicycle_repair_station', 'biergarten', 'boat_rental', 'boat_sharing', 'brothel', 'bureau_de_change', 'bus_station', 'cafe', 'car_rental', 'car_sharing', 'car_wash', 'casino', 'charging_station', 'cinema', 'clinic', 'college', 'community_centre', 'compressed_air', 'conference_centre', 'courthouse', 'dentist', 'doctors', 'dog_toilet', 'dressing_room', 'drinking_water', 'driving_school', 'events_venue', 'fast_food', 'ferry_terminal', 'fire_station', 'food_court', 'fountain', 'fuel', 'gambling', 'give_box', 'grit_bin', 'hospital', 'ice_cream', 'kindergarten', 'language_school', 'library', 'love_hotel', 'motorcycle_parking', 'music_school', 'nightclub', 'nursing_home', 'parcel_locker', 'parking', 'parking_entrance', 'parking_space', 'pharmacy', 'planetarium', 'police', 'post_box', 'post_depot', 'post_office', 'prison', 'pub', 'public_bookcase', 'ranger_station', 'recycling', 'restaurant', 'sanitary_dump_station', 'school', 'shelter', 'shower', 'social_centre', 'social_facility', 'stripclub', 'studio', 'swingerclub', 'taxi', 'telephone', 'theatre', 'toilets', 'townhall', 'toy_library', 'training', 'university', 'vehicle_inspection', 'veterinary', 'waste_basket', 'waste_disposal', 'waste_transfer_station', 'water_point', 'watering_place']

CLASS_74_LIST = ['arts_centre', 'atm', 'bank', 'bar', 'bench', 'bicycle_parking', 'bicycle_rental', 'bicycle_repair_station', 'boat_rental', 'bureau_de_change', 'bus_station', 'cafe', 'car_rental', 'car_sharing', 'car_wash', 'charging_station', 'cinema', 'clinic', 'college', 'community_centre', 'conference_centre', 'courthouse', 'dentist', 'doctors', 'drinking_water', 'driving_school', 'events_venue', 'fast_food', 'ferry_terminal', 'fire_station', 'food_court', 'fountain', 'fuel', 'gambling', 'hospital', 'kindergarten', 'language_school', 'library', 'motorcycle_parking', 'music_school', 'nightclub', 'nursing_home', 'parcel_locker', 'parking', 'pharmacy', 'police', 'post_box', 'post_depot', 'post_office', 'pub', 'public_bookcase', 'recycling', 'restaurant', 'sanitary_dump_station', 'school', 'shelter', 'social_centre', 'social_facility', 'stripclub', 'studio', 'swingerclub', 'taxi', 'telephone', 'theatre', 'toilets', 'townhall', 'university', 'vehicle_inspection', 'veterinary', 'waste_basket', 'waste_disposal', 'waste_transfer_station', 'water_point', 'watering_place']

FEWSHOT_CLASS_55_LIST = ['arts_centre', 'atm', 'bank', 'bar', 'bench', 'bicycle_parking', 'bicycle_rental', 'boat_rental', 'bureau_de_change', 'bus_station', 'cafe', 'car_rental', 'car_sharing', 'car_wash', 'charging_station', 'cinema', 'clinic', 'college', 'community_centre', 'courthouse', 'dentist', 'doctors', 'drinking_water', 'driving_school', 'events_venue', 'fast_food', 'ferry_terminal', 'fire_station', 'fountain', 'fuel', 'hospital', 'kindergarten', 'library', 'music_school', 'nightclub', 'parking', 'pharmacy', 'police', 'post_box', 'post_office', 'pub', 'public_bookcase', 'recycling', 'restaurant', 'school', 'shelter', 'social_centre', 'social_facility', 'studio', 'theatre', 'toilets', 'townhall', 'university', 'vehicle_inspection', 'veterinary']


DICT_9to74 = {'sustenance':['bar','cafe','fast_food','food_court','pub','restaurant'],
    'education':['college','driving_school','kindergarten','language_school','library','music_school','school','university'],
    'transportation':['bicycle_parking','bicycle_repair_station','bicycle_rental','boat_rental',
    'bus_station','car_rental','car_sharing','car_wash','vehicle_inspection','charging_station','ferry_terminal',
        'fuel','motorcycle_parking','parking','taxi'],
    'financial':['atm','bank','bureau_de_change'],
    'healthcare':['clinic','dentist','doctors','hospital','nursing_home','pharmacy','social_facility','veterinary'],
    'entertainment_arts_culture':['arts_centre','cinema','community_centre',
        'conference_centre','events_venue','fountain','gambling',
        'nightclub','public_bookcase','social_centre','stripclub','studio','swingerclub','theatre'],
    'public_service':['courthouse','fire_station','police','post_box',
        'post_depot','post_office','townhall'],
    'facilities':['bench','drinking_water','parcel_locker','shelter',
'telephone','toilets','water_point','watering_place'],
    'waste_management':['sanitary_dump_station','recycling','waste_basket','waste_disposal','waste_transfer_station',]
                 }

DICT_74to9 = revert_dict(DICT_9to74)

DICT_9to95 = {
    'education':{'college','driving_school','kindergarten','language_school','library','toy_library','training','music_school','school','university'},
    'entertainment_arts_culture':{'arts_centre','brothel','casino','cinema','community_centre','conference_centre','events_venue','fountain','gambling','love_hotel','nightclub','planetarium','public_bookcase','social_centre','stripclub','studio','swingerclub','theatre'},
    'facilities':{'bbq','bench','dog_toilet','dressing_room','drinking_water','give_box','parcel_locker','shelter','shower','telephone','toilets','water_point','watering_place'},
    'financial':{'atm','bank','bureau_de_change'},
    'healthcare':{'baby_hatch','clinic','dentist','doctors','hospital','nursing_home','pharmacy','social_facility','veterinary'},
    'public_service':{'courthouse','fire_station','police','post_box','post_depot','post_office','prison','ranger_station','townhall'},
    'sustenance':{'bar','biergarten','cafe','fast_food','food_court','ice_cream','pub','restaurant',},
    'transportation':{'bicycle_parking','bicycle_repair_station','bicycle_rental','boat_rental','boat_sharing','bus_station','car_rental','car_sharing','car_wash','compressed_air','vehicle_inspection','charging_station','ferry_terminal','fuel','grit_bin','motorcycle_parking','parking','parking_entrance','parking_space','taxi'},
    'waste_management':{'sanitary_dump_station','recycling','waste_basket','waste_disposal','waste_transfer_station'}}

DICT_95to9 = revert_dict(DICT_9to95)

# DICT_95to9 = {
#     'college':'education',
#     'driving_school':'education',
#     'kindergarten':'education',
#     'language_school':'education',
#     'library':'education',
#     'toy_library':'education',
#     'training':'education',
#     'music_school':'education',
#     'school':'education',
#     'university':'education',

#     'arts_centre':'entertainment_arts_culture',
#     'brothel':'entertainment_arts_culture',
#     'casino':'entertainment_arts_culture',
#     'cinema':'entertainment_arts_culture',
#     'community_centre':'entertainment_arts_culture',
#     'conference_centre':'entertainment_arts_culture',
#     'events_venue':'entertainment_arts_culture',
#     'fountain':'entertainment_arts_culture',
#     'gambling':'entertainment_arts_culture',
#     'love_hotel':'entertainment_arts_culture',
#     'nightclub':'entertainment_arts_culture',
#     'planetarium':'entertainment_arts_culture',
#     'public_bookcase':'entertainment_arts_culture',
#     'social_centre':'entertainment_arts_culture',
#     'stripclub':'entertainment_arts_culture',
#     'studio':'entertainment_arts_culture',
#     'swingerclub':'entertainment_arts_culture',
#     'theatre':'entertainment_arts_culture',

#     'bbq': 'facilities',
#     'bench': 'facilities',
#     'dog_toilet': 'facilities',
#     'dressing_room': 'facilities',
#     'drinking_water': 'facilities',
#     'give_box': 'facilities',
#     'parcel_locker': 'facilities',
#     'shelter': 'facilities',
#     'shower': 'facilities',
#     'telephone': 'facilities',
#     'toilets': 'facilities',
#     'water_point': 'facilities',
#     'watering_place': 'facilities',

#     'atm': 'financial',
#     'bank': 'financial',
#     'bureau_de_change': 'financial',

#     'baby_hatch':'healthcare',
#     'clinic':'healthcare',
#     'dentist':'healthcare',
#     'doctors':'healthcare',
#     'hospital':'healthcare',
#     'nursing_home':'healthcare',
#     'pharmacy':'healthcare',
#     'social_facility':'healthcare',
#     'veterinary':'healthcare',

#     'courthouse': 'public_service',
#     'fire_station': 'public_service',
#     'police': 'public_service',
#     'post_box': 'public_service',
#     'post_depot': 'public_service',
#     'post_office': 'public_service',
#     'prison': 'public_service',
#     'ranger_station': 'public_service',
#     'townhall': 'public_service',

#     'bar': 'sustenance',
#     'biergarten': 'sustenance',
#     'cafe': 'sustenance',
#     'fast_food': 'sustenance',
#     'food_court': 'sustenance',
#     'ice_cream': 'sustenance',
#     'pub': 'sustenance',
#     'restaurant': 'sustenance',

#     'bicycle_parking': 'transportation',
#     'bicycle_repair_station': 'transportation',
#     'bicycle_rental': 'transportation',
#     'boat_rental': 'transportation',
#     'boat_sharing': 'transportation',
#     'bus_station': 'transportation',
#     'car_rental': 'transportation',
#     'car_sharing': 'transportation',
#     'car_wash': 'transportation',
#     'compressed_air': 'transportation',
#     'vehicle_inspection': 'transportation',
#     'charging_station': 'transportation',
#     'ferry_terminal': 'transportation',
#     'fuel': 'transportation',
#     'grit_bin': 'transportation',
#     'motorcycle_parking': 'transportation',
#     'parking': 'transportation',
#     'parking_entrance': 'transportation',
#     'parking_space': 'transportation',
#     'taxi': 'transportation',

#     'sanitary_dump_station': 'waste_management',
#     'recycling': 'waste_management',
#     'waste_basket': 'waste_management',
#     'waste_disposal': 'waste_management',
#     'waste_transfer_station': 'waste_management',
    
# }

# CLASS_9_LIST = ['sustenance', 'education', 'transportation', 'financial', 'healthcare', 'entertainment_arts_culture', 'public_service', 'facilities', 'waste_management']

# FINE_LIST = ['bar','biergarten','cafe','fast_food','food_court','ice_cream','pub','restaurant','college','driving_school','kindergarten','language_school','library','toy_library','training','music_school','school','university','bicycle_parking','bicycle_repair_station','bicycle_rental','boat_rental','boat_sharing','bus_station','car_rental','car_sharing','car_wash','compressed_air','vehicle_inspection','charging_station','ferry_terminal','fuel','grit_bin','motorcycle_parking','parking','parking_entrance','parking_space','taxi','atm','bank','bureau_de_change','baby_hatch','clinic','dentist','doctors','hospital','nursing_home','pharmacy','social_facility','veterinary','arts_centre','brothel','casino','cinema','community_centre','conference_centre','events_venue','fountain','gambling','love_hotel','nightclub','planetarium','public_bookcase','social_centre','stripclub','studio','swingerclub','theatre','courthouse','fire_station','police','post_box','post_depot','post_office','prison','ranger_station','townhall','bbq','bench','dog_toilet','dressing_room','drinking_water','give_box','parcel_locker','shelter','shower','telephone','toilets','water_point','watering_place','sanitary_dump_station','recycling','waste_basket','waste_disposal','waste_transfer_station']

# FINE_LIST = ['bar','biergarten','cafe','fast_food','food_court','ice_cream','pub','restaurant','college','driving_school','kindergarten','language_school','library','toy_library','training','music_school','school','university','bicycle_parking','bicycle_repair_station','bicycle_rental','boat_rental','boat_sharing','bus_station','car_rental','car_sharing','car_wash','compressed_air','vehicle_inspection','charging_station','ferry_terminal','fuel','grit_bin','motorcycle_parking','parking','parking_entrance','parking_space','taxi','atm','bank','bureau_de_change','baby_hatch','clinic','dentist','doctors','hospital','nursing_home','pharmacy','social_facility','veterinary','arts_centre','brothel','casino','cinema','community_centre','conference_centre','events_venue','fountain','gambling','love_hotel','nightclub','planetarium','public_bookcase','social_centre','stripclub','studio','swingerclub','theatre','courthouse','fire_station','police','post_box','post_depot','post_office','prison','ranger_station','townhall','bbq','bench','dog_toilet','dressing_room','drinking_water','give_box','parcel_locker','shelter','shower','telephone','toilets','water_point','watering_place','sanitary_dump_station','recycling','waste_basket','waste_disposal','waste_transfer_station','animal_boarding','animal_breeding','animal_shelter','baking_oven','childcare','clock','crematorium','dive_centre','funeral_hall','grave_yard','hunting_stand','internet_cafe','kitchen','kneipp_water_cure','lounger','marketplace','monastery','photo_booth','place_of_mourning','place_of_worship','public_bath','refugee_site','vending_machine']