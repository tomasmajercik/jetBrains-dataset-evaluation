from matplotlib import pyplot as plt
import numpy

# !FUNCTIONS!

def generate_start_points(num_points, lower_interval_bound, upper_interval_bound):
    points_set = set() #use set so we have unique points

    while len(points_set) < num_points: #untill points_set is not full - based on num_points
        #generate random point (x,y)
        x = numpy.random.randint(lower_interval_bound, upper_interval_bound + 1)
        y = numpy.random.randint(lower_interval_bound, upper_interval_bound + 1)

        points_set.add((x, y)) #add x and y as a tuple into the set

    return numpy.array(list(points_set)) # convert set to numpy array and return
def generate_additional_points(default_points, num_additional_points, lower_interval_bound, upper_interval_bound, close_to_edge_penalty):
    points = default_points.copy() #copy numpy array and work with its copy

    for _ in range(num_additional_points):
        # choose random point from existing points
        random_index = numpy.random.randint(len(points)) 
        random_point = points[random_index]
        x, y = random_point

        # edit offset interval if too close to edge
        offset_interval = 100
        if  x < lower_interval_bound + 100 or x > upper_interval_bound-100 or y < lower_interval_bound + 100 or y > upper_interval_bound-100:
            offset_interval -= close_to_edge_penalty

        # generate offset
        X_offset = numpy.random.randint(-offset_interval, offset_interval)
        Y_offset = numpy.random.randint(-offset_interval, offset_interval)

        # set new point
        new_x = x+X_offset
        new_y = y+Y_offset

        points = numpy.append(points, numpy.array([[new_x, new_y]]), axis = 0) # append new point to the points
    return points

def visualise_clusters(clusters, medoids):
    # colors to choose from
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown", "cyan", "magenta", "lime", "indigo", "violet", "gold", "silver", "turquoise", "maroon", "navy", "coral", "teal"]
    plt.figure(figsize=(10, 10))

    # for each cluster, show one point on matplotlib
    for i in range(len(clusters)):
        cluster = clusters[i]
        if len(cluster) > 0:
            cluster_points = numpy.array(cluster)
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i % len(colors)], s=10, label='Cluster {}'.format(i+1))

    # plot centroids with larger size and different color
    medoids = numpy.array(medoids)
    plt.scatter(medoids[:, 0], medoids[:, 1], c='black', s=100, marker='x', label='Centroids', alpha=0.7)

    # aditional matplotlib settings
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.title("Divisive CLustering (centroid - K-Means)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(False)
    plt.show()

def euclidean_distance(point, other_point):
    # calculate euclidian distance based on formula 
    return numpy.sqrt(numpy.sum(((other_point - point) ** 2)))

def evaluate_cluster_average_mean(clusters, centroids, k, print_logs):
    ## if print_logs is False -> this function is used for calculation during divisive clustering
    ## else if print_logs is True -> this function is used in the end of code to evaluare average mean

    succesful_clusters = 0

    # calculate the sum of distances between each point in the cluster and the centroid of that cluster
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            # Sum up the distances from each point in the cluster to its centroid
            total_distance = sum(euclidean_distance(point, centroids[i]) for point in cluster)
            avg_distance = total_distance / len(cluster)
            if avg_distance <= 500:
                succesful_clusters += 1
            if print_logs:
                print(f"Average distance for cluster {i + 1}: {avg_distance:.2f}")
        else: 
            if print_logs:
                print(f"Cluster {i+1} was left empty")

    if print_logs:
        print(f"Number of cluster with average distance from centroid smaller than 500 is {succesful_clusters} from {k} possible")

    return succesful_clusters, k

def k_means(points, k, lower_interval_bound, upper_interval_bound):
    # choose k random centroids
    centroids = points[numpy.random.choice(points.shape[0], k, replace=False)]

    #asign points to the nearest centroid
    i = 1
    
    current_avg = 0
    previous_avg = 0

    no_update_count = 0
    average_distances = 0
    average_distances_count = 0
    while True:
        print(f"Iteration no.: {i}")
        i -=- 1

        clusters = [[] for _ in range(k)] #define k clusters
        distances_to_points = []
        max_distances = [0] * k

        # put each point to cluster
        for point in points:
            all_distances = [euclidean_distance(point, centroid) for centroid in centroids]

            average_distances += sum(all_distances)
            average_distances_count += len(all_distances)

            closest_centroid = numpy.argmin(all_distances)
            distances_to_points.append(all_distances[closest_centroid]) 
            clusters[closest_centroid].append(point)

            # Update max distance for this centroid
            max_distances[closest_centroid] = max(max_distances[closest_centroid], all_distances[closest_centroid])

        #recalculate centroids
        new_centroids = []
        for cluster in clusters:
            if len(cluster) > 0:  # Only calculate the mean for non-empty clusters
                new_centroids.append(numpy.mean(cluster, axis=0))
            else:
                # print("bububu")
                new_centroids.append([numpy.random.randint(lower_interval_bound, upper_interval_bound), numpy.random.randint(lower_interval_bound, upper_interval_bound)])

        if abs(previous_avg - current_avg) < 1: 
            no_update_count += 1
        else:
            no_update_count = 0  # Reset if assignments changed


        # Stop if there's no update for 3 iterations
        if no_update_count >= 3:
            break

        # Update centroids for the next iteration
        centroids = new_centroids
        previous_avg = current_avg

    return clusters, centroids
####

def divisive_clustering(points, lower_interval_bound, upper_interval_bound):

    clusters = [points] # all points are one cluster at first
    centroids = [numpy.mean(points, axis=0)]

    while True:
        # Evaluate the current clusters
        successful_clusters, total_clusters = evaluate_cluster_average_mean(clusters, centroids, len(clusters), False)
        
        # if each clusters average mean is less than 500, stop the clustering 
        if successful_clusters == total_clusters:
            print("All clusters have average less than 500.")
            break

        # find the biggest cluster by average distance to its centroid
        max_distance_cluster_index = -1
        max_distance_avg = -1
        for i, cluster in enumerate(clusters):
            if len(cluster) > 0:
                centroid = centroids[i]
                avg_distance = sum(euclidean_distance(point, centroid) for point in cluster) / len(cluster)
                if avg_distance > max_distance_avg:
                    max_distance_avg = avg_distance
                    max_distance_cluster_index = i

        # split the largest cluster if its average distance is above the threshold
        if max_distance_avg > 500:
            largest_cluster = clusters.pop(max_distance_cluster_index)
            centroids.pop(max_distance_cluster_index)  # Remove the centroid of the largest cluster

            # split the biggest cluster with k_means with centroid
            sub_clusters, sub_medoids = k_means(numpy.array(largest_cluster), 2, lower_interval_bound, upper_interval_bound)

            # add the split clusters back to the main list of clusters
            clusters.extend(sub_clusters)
            centroids.extend(sub_medoids)
        else:
            break  # stop if all clusters are below the threshold




    return clusters, centroids
 

# !START HERE!

# important variables
num_points = 20
lower_interval_bound = -5000
upper_interval_bound = 5000

num_additional_points = 40000
close_to_edge_penalty = 10

# inicialize first points
points = generate_start_points(num_points, lower_interval_bound, upper_interval_bound) 
# find additional points
additional_points = generate_additional_points(points, num_additional_points, lower_interval_bound, upper_interval_bound, close_to_edge_penalty) 

# start divisive clustering
clusters, centroids = divisive_clustering(additional_points, lower_interval_bound, upper_interval_bound)
# print the average mean 
evaluate_cluster_average_mean(clusters, centroids, len(centroids), True)
print("buliding visualisation...")
# visualise clustering
visualise_clusters(clusters, centroids)
