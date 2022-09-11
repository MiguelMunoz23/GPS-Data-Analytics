#!/usr/bin/python3.8
"""
gps_data_analytics.py

This Python script performs the parsing of the raw data from a GPS, applies data analytics and visualizes those results
in a table, graphs and a map from Google Maps.

Authors:
+ Jenyfer Eugenia Toj López
+ Juan Carlos Aguilar Ramírez
+ Miguel Angel Muñoz Rizo

Institution: Universidad de Monterrey
Subject: Advanced Robotics
Lecturer: Dr. Andrés Hernández Gutiérrez

Date of creation: 07 September 2022
"""
from math import sqrt, sin, cos

import matplotlib.pyplot as plt
import numpy as np
from numpy import deg2rad
from prettytable import PrettyTable
import gmplot


def gps_data_parsing(txt):
    """
    This function is used to parse the data from the original file.
    :param txt: A .txt file with the raw data from the GPS.
    :return:
        + longitude: A list with the longitude values [°].
        + latitude: A list with the latitude values [°].
        + altitude: A list with the altitude values [m].
        + time: A list with the time values [string].
        + speed: A list with the speed values [m/s].
    """
    txt = open(txt)  # open the input txt file
    time = []
    latitude = []  # create a latitude list
    longitude = []  # create a longitude list
    altitude = []  # create a altitude list
    speed = []  # create a altitude list
    first = 0
    for line in txt.readlines():  # check each line of the txt file
        if first == 0:
            first += 1
            continue
        date_time = (line.split(',')[1])  # divide the line by ',' and take the date time
        time.append(date_time.split(' ')[1])  # divide the line by ',' and take the time
        latitude.append(float(line.split(',')[2]))  # divide the line by ',' and take the latitude
        longitude.append(float(line.split(',')[3]))  # divide the line by ',' and take the longitude
        altitude.append(float(line.split(',')[5]))  # divide the line by ',' and take the altitude
        speed.append(float(line.split(',')[7]))  # divide the line by ',' and take the speed
    # Return a list of latitude, longitude, altitude values of txt file
    return longitude, latitude, altitude, time, speed


def get_acceleration(speed):
    """
    This function computes the acceleration between two speed measurements.
    :param speed: A list containing all the speeds [m/s].
    :return:
        + accelerations: The acceleration list at each second.
        + average_acceleration: The average of the acceleration list.
        + max_acc: The maximum acceleration in the list.
    """
    accelerations = [0]
    for i in range(len(speed)):
        # GPS takes a sample every second, so delta_t = 1 [sec]
        # a = vf - vi / (tf - ti)
        if i == 0:
            continue
        accelerations.append(speed[i] - speed[i - 1])
    max_acc, min_acc = max(accelerations), min(accelerations)
    # Check the absolute values to determine the maximum acceleration
    if abs(min_acc) > abs(max_acc):
        max_acc = min_acc
    return accelerations, sum(accelerations) / len(accelerations), max_acc


def get_distance(longitude, latitude, altitude):
    """
    This function gets the total distance. It goes from the geodetic to the ECEF to the NED coordinate system and then
    measures the distance.
    :param longitude: A list containing all the longitude values [°].
    :param latitude: A list containing all the latitude values [°].
    :param altitude: A list containing all the altitude values [°].
    :return:
        + distance[-1]: The total distance.
        + distance: The list "distance" with the accumulated distance at each second.
    """
    # Define constants
    R_Ea = 6378137  # [m]
    f = 1 / 298.257223563
    R_Eb = R_Ea * (1 - f)
    e = sqrt(R_Ea ** 2 - R_Eb ** 2) / R_Ea

    # Create a 3 x n matrix with all the geodetic, ECEF and NED system coordinates
    geodetic_system = np.zeros((3, len(longitude)))
    ecef_system = np.zeros(shape=geodetic_system.shape)
    ned_system = np.zeros(shape=geodetic_system.shape)

    # Initialize necessary variables used in the for loop.
    R_ne, distance = 0, [0]

    # For each point, fill the 3 different system coordinates matrices
    for i in range(len(longitude)):
        # Define the current lon, lat and alt in radians
        lon, lat, alt = deg2rad(longitude[i]), deg2rad(latitude[i]), altitude[i]
        # Fill the geodetic_system matrix
        geodetic_system[:, i] = lon, lat, alt
        # Calculate Ne
        Ne = R_Ea / sqrt(1 - e ** 2 * sin(lat) ** 2)
        # Fill the ecef_system matrix
        ecef_system[:, i] = (Ne + alt) * cos(lat) * cos(lon),\
                            (Ne + alt) * cos(lat) * sin(lon), \
                            (Ne * (1 - e ** 2) + alt) * sin(lat)
        # If it's the first point
        if i == 0:
            # Calculate the R_ne matrix with the current lon, lat and alt values, which are the reference points
            R_ne = np.array([[(-sin(lat) * cos(lon)), (-sin(lat) * sin(lon)), cos(lat)],
                             [      -sin(lon),                cos(lon),           0   ],
                             [(-cos(lat) * cos(lon)), (-cos(lat) * sin(lon)), -sin(lat)]])
            continue
        # Calculate ned_system coordinates after the first point
        ned_system[:, i] = np.matmul(R_ne, (ecef_system[:, i] - ecef_system[:, 0]).reshape(3, 1)).reshape(1, 3)

        distance.append(np.linalg.norm(ned_system[:, i] - ned_system[:, i - 1]) + distance[i - 1])
    return distance[-1], distance


def gps_data_analytics(lon, lat, alt, time, speed):
    """
    This function analyzes the data from the GPS and returns some statistics.
    :param lon: A list containing all the longitude values [°].
    :param lat: A list containing all the latitude values [°].
    :param alt: A list containing all the altitude values [°].
    :param time: A list containing the time [string].
    :param speed: A list containing the speed [m/s].
    :return:
        + start_time: A string containing the start time and zone.
        + end_time: A string containing the end time and zone.
        + total_dist: The total distance traveled [m].
        + av_speed: The average speed [m/s].
        + max_speed: The maximum speed [m/s].
        + av_acc: The average acceleration [m/s²].
        + max_acc: The maximum speed [m/s²].
        + lowest_elevation: The minimum altitude point [m].
        + highest_elevation: The maximum altitude point [m].
        + elevation_diff: The maximum difference in altitudes [m].
        + dist_list: The list "distance" with the accumulated distance [m] at each second.
        + accelerations: The acceleration [m/s²] list at each second.
    """
    start_time, end_time = min(time) + ' T', max(time) + ' T'
    total_dist, dist_list = get_distance(lon, lat, alt)
    av_speed = max(speed) / len(speed)
    max_speed = max(speed)
    accelerations, av_acc, max_acc = get_acceleration(speed)
    lowest_elevation, highest_elevation = min(alt), max(alt)
    elevation_diff = highest_elevation - lowest_elevation

    return start_time, end_time, total_dist, av_speed, max_speed, av_acc, max_acc, lowest_elevation, \
           highest_elevation, elevation_diff, dist_list, accelerations


def get_graphs(distance, parameter, titles, graph_name):
    """
    This function is used to graph the results from the GPS statistics.
    :param distance: A list with the accumulated distance [m] at each second.
    :param parameter: A list with the parameters that are going to be graphed.
    :param titles: A list with the titles for the graph.
    :param graph_name: A list with the names for the files.
    :return: None.
    """
    plt.figure()
    plt.plot(distance, parameter)
    plt.title(titles[0])
    plt.xlabel(titles[1])
    plt.ylabel(titles[2])
    plt.grid()
    plt.savefig(f"graphs/{graph_name}")
    return None


def gps_report_generator(results, alt, speed):
    """
    This function is used to report the statistics obtained from the GPS.
    :param results: A tuple containing the start time, end time, total distance, average speed, maximum speed, average
                    acceleration, maximum acceleration, the lowest elevation, the highest elevation, elevation
                    difference, the list of the distances and the list of the accelerations in that order.
    :param alt: A list with the altitude values [m].
    :param speed: A list with the speed values [m/s].
    :return: None.
    """
    # Print the table using PrettyTable
    table = PrettyTable()
    table.field_names = ["GPS Statistics", "Value", "Units"]
    table.add_rows(
        [
            ["Start Time", results[0], "[H/M/S]"],
            ["End Time", results[1], "[H/M/S]"],
            ["Total Distance", f"{results[2]:.4f}", "[m]"],
            ["Average Speed", f"{results[3]:.6f}", "[m/s]"],
            ["Max Speed", results[4], "[m/s]"],
            ["Average Acceleration", f"{results[5]:.4f}", "[m/s²]"],
            ["Max Acceleration", f"{results[6]:.4f}", "[m/s²]"],
            ["Lowest Elevation", results[7], "[m]"],
            ["Highest Elevation", results[8], "[m]"],
            ["Elevation Difference", results[9], "[m]"]
        ]
    )
    print(table)

    # Compute the elevation change, take the first altitude as the reference
    alt_change = [alt_measurement - alt[0] for alt_measurement in alt]

    # Graph the results
    parameters = [alt, alt_change, speed, results[11]]
    titles = [["Elevation vs Distance", "Distance [m]", "Elevation [m]"],
              ["Elevation Change vs Distance", "Distance [m]", "Elevation Change [m]"],
              ["Speed vs Distance", "Distance [m]", "Speed [m/s]"],
              ["Acceleration vs Distance", "Distance [m]", "Acceleration [m/s²]"]]
    graphs_names = ["elevation-distance.png", "elevationChange-distance.png", "speed-distance.png",
                    "acceleration-distance.png"]

    for parameter, title, graph_name in zip(parameters, titles, graphs_names):
        get_graphs(results[10], parameter, title, graph_name)

    return None


def get_map(lon, lat):
    """
    This function is used to plot the GPS path in Google Maps.
    :param lon: A list containing the longitude values [°].
    :param lat: A list containing the latitude values [°].
    :return: None
    """
    # Create a map and specify the latitud and longitud center point
    gmap = gmplot.GoogleMapPlotter(25.6886933, -100.3693874, 14)
    # Create a marker at the beginning of the GPS measurements
    gmap.marker(lat[0], lon[0], color='red')
    # Plot all the coordinates in blue and draw the map
    gmap.plot(lat, lon, 'blue', edge_width=4)
    gmap.draw("map.html")
    return None


def main():
    """
    This main function calls all the necessary functions to implement the workflow of GPS data anayltics for vehicle
    navigation.
    :return: None
    """
    # Raw GPS data
    raw_data = 'datasets/20200919-204133.txt'
    # GPS data parsing
    lon, lat, alt, time, speed = gps_data_parsing(raw_data)
    # GPS data analytics
    results = gps_data_analytics(lon, lat, alt, time, speed)
    # Report Generator
    gps_report_generator(results, alt, speed)
    # Get map from Google Maps
    get_map(lon, lat)


if __name__ == '__main__':
    main()
