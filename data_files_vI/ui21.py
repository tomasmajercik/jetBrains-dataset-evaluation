from matplotlib import pyplot as plt
import numpy

# !FUNCTIONS!

def generate_start_points(num_points, lower_interval_bound, upper_interval_bound):
    points_set = set() # use set so we have unique points

    while len(points_set) < num_points: #untill points_set is not full - based on num_points
        #generate random point (x,y)
        x = numpy.random.randint(lower_interval_bound, upper_interval_bound + 1) 
        y = numpy.random.randint(lower_interval_bound, upper_interval_bound + 1)

        points_set.add((x, y)) #add x and y as a tuple into the set

    return numpy.array(list(points_set))  # convert set to numpy array and return
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

        # append new point to the points
        points = numpy.append(points, numpy.array([[new_x, new_y]]), axis = 0)

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

    # Plot centroids with larger size and different color
    medoids = numpy.array(medoids)
    plt.scatter(medoids[:, 0], medoids[:, 1], c='black', s=100, marker='x', label='Centroids', alpha=0.7)

    # aditional matplotlib settings
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.title("K-Means Clustering (medoid)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(False)
    plt.show()

def euclidean_distance(point, other_point):
    # calculate euclidian distance based on formula 
    return numpy.sqrt(numpy.sum(((other_point - point) ** 2)))

def k_means(points, k, lower_interval_bound, upper_interval_bound):
    # choose k random medoids 
    medoids = []
    for _ in range(k):
        medoids.append([numpy.random.randint(lower_interval_bound, upper_interval_bound), numpy.random.randint(lower_interval_bound, upper_interval_bound)])

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
        distances_to_medoids = []

        # put each point to cluster
        for point in points:
            all_distances = [euclidean_distance(point, medoid) for medoid in medoids]

            average_distances += sum(all_distances)
            average_distances_count += len(all_distances)

            closest_medoid = numpy.argmin(all_distances)
            distances_to_medoids.append(all_distances[closest_medoid])
            clusters[closest_medoid].append(point)

        # Calculate the average distance for this iteration
        if distances_to_medoids:
            average_distances = sum(distances_to_medoids)
            average_distances_count = len(distances_to_medoids)
            current_avg = average_distances / average_distances_count
            print(f"Average distance to closest medoid: {current_avg:.2f}")

        #recalculate medoids
        new_medoids = []
        for cluster in clusters:
            if len(cluster) > 0:
                cluster_mean = numpy.mean(cluster, axis=0)
                medoid = min(cluster, key=lambda point: euclidean_distance(point, cluster_mean))
                new_medoids.append(medoid)
            else:
                new_medoids.append(points[numpy.random.randint(len(points))])

        if no_update_count > 0:
            print(f"No update for {no_update_count}/3")

        if abs(previous_avg - current_avg) < 1:
            no_update_count += 1
        else:
            no_update_count = 0  # Reset if assignments have changed


        # Stop if there's no update for 3 iterations
        if no_update_count >= 3:
            print(f"    very small update for {no_update_count} iterations, not worth to continue")
            break

        # Update centroids for the next iteration
        medoids = new_medoids
        previous_avg = current_avg

    return clusters, medoids

def evaluate_cluster_average_mean(clusters, centroids, k):

    succesful_clusters = 0

    # calculate the sum of distances between each point in the cluster and the centroid of that cluster
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            # Sum up the distances from each point in the cluster to its centroid
            total_distance = sum(euclidean_distance(point, centroids[i]) for point in cluster)
            avg_distance = total_distance / len(cluster)
            if avg_distance <= 500:
                succesful_clusters += 1
            print(f"Average distance for cluster {i + 1}: {avg_distance:.2f}")
        else: 
            print(f"Cluster {i+1} was left empty")

    print(f"Number of cluster with average distance from centroid smaller than 500 is {succesful_clusters} from {k} possible")

    return
####

 
# !START HERE!

#important variables
num_points = 20
lower_interval_bound = -5000
upper_interval_bound = 5000

num_additional_points = 40000
close_to_edge_penalty = 10

k = 15 #number of clusters

#inicialize first points
points = generate_start_points(num_points, lower_interval_bound, upper_interval_bound) 
# find additional points
additional_points = generate_additional_points(points, num_additional_points, lower_interval_bound, upper_interval_bound, close_to_edge_penalty) 

clusters, medoids = k_means(additional_points, k, lower_interval_bound, upper_interval_bound)
evaluate_cluster_average_mean(clusters, medoids, k)
print("buliding visualisation...")
visualise_clusters(clusters, medoids)
