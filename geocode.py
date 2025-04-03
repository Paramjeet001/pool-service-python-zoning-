import psycopg2

connection = psycopg2.connect(database="innovatemore", user="postgres", password="abcd", host="127.0.0.1", port=5432)

cursor = connection.cursor()

cursor.execute('''SELECT address,
    city,
    state,
    zip from public.addresses;''')

record = cursor.fetchall()
addresses = []
for i in record:
    a=""
    for j in i:
        a=a+" "+j
    a=a.strip()
    addresses.append(a)
# print(addresses)

from geopy.geocoders import Nominatim

#function to get coordinates
def get_coordinates(address):

    geolocator = Nominatim(user_agent="my_name")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return 0.0,0.0


# Fetch and print coordinates for each address
for address in addresses:
    coordinates = get_coordinates(address)
    print(coordinates)

for address in addresses:
    coordinates = get_coordinates(address)
    insert = '''
    insert into public.lat_long values (%s,%s)
    ''' 
    cursor.execute(insert,coordinates)
print("data successfully inserted")
connection.commit()
cursor.close()
connection.close()