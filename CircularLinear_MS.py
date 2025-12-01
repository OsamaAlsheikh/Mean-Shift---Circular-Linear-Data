
"""
Mean Shift Clustering for Circular-Linear Data

This implementation follows the methodology described in:
Cheng, Y. (1995). "Mean Shift, Mode Seeking, and Clustering"
IEEE Transactions on Pattern Analysis and Machine Intelligence

Author: Osama Al Sheikh Ali
Date: November 2025
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import Tuple, List, Dict



class MeanShiftCircularLinear:
    """
    Mean Shift clustering algorithm for circular-linear data.
    
    The algorithm iteratively shifts each data point to the weighted mean
    of points in its neighborhood, where the neighborhood is defined by
    a kernel function. For circular-linear data, I use a product kernel
    that combines a von Mises kernel for the circular component (theta)
    and a Gaussian kernel for the linear component (r).
    """
        
    def __init__(self, 
                 bandwidth_theta: float = 0.5,
                 bandwidth_r: float = 1.0,
                 max_iter: int = 300,
                 tol: float = 1e-4):
        
        """
        Initialize Mean Shift algorithm for circular-linear data.
        
        Parameters:
        bandwidth_theta : Controls the width of the von Mises kernel.
        bandwidth_r : Controls the width of the Gaussian kernel.
        max_iter : Maximum number of iterations if convergence is not met
        tol : Convergence tolerance where the algorithm stops when the shift is below this threshold.
        """
        self.bandwidth_theta = bandwidth_theta
        self.bandwidth_r = bandwidth_r
        self.max_iter = max_iter
        self.tol = tol
        self.modes = None
        self.labels = None
        self.cluster_params = None
        self.iteration_counts = []
        self.total_time = 0.0
        

    # -------------------------
    # Important helper functions
    # -------------------------    
    def _circ_dist(self, theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
        """
        This function returns the shortest distance between two angles on a circle.

        Angles wrap around so a direct subtraction can give a misleading
        “long way around” distance. For example, 350 degree and 10 degrees look far apart
        if we subtract them normally (|350−10| = 340), but on a circle the
        shorter distance is actually just 20 degrees.

        So basically this just checks both directions and picks the smaller
        one, so the distance actually makes sense on a circle.
        """
        diff = np.abs(theta1 - theta2)
        return np.minimum(diff, 2 * np.pi - diff)
    
    
    def _circular_mean(self, angles: np.ndarray, weights: np.ndarray) -> float:
        """
        This function compute a weighted mean of angles while respecting their circular nature.

        Angles can't be averaged directly because they wrap at 2pi. For example,
        the average of 350 degrees and 10 degrees is not 180 degree but instead  around 0.

        To avoid this issue, the angles are averaged using the weights and then converted first to a unit vectors.
        (cos θ, sin θ) and then turned back into an angle with arctan2. 
        The result is wrapped into [0, 2pi].
        The Parameters are the angles which are in radians and the weights for each angle. 

        THIS FUNCTION is very much inspired the ideas and explanation explained in this video https://www.youtube.com/watch?v=2kIGEEzie1M

        """
        x = np.sum(weights * np.cos(angles))
        y = np.sum(weights * np.sin(angles))
        mean_angle = np.arctan2(y, x)
        return mean_angle % (2 * np.pi) #wrapping into [0, 2pi]
        


    # -------------------------
    # Kernels: von Mises & Gaussian
    # -------------------------
    def _von_mises_kernel(self, theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
        """
        Von mises is K(delta angle) = exp(kappa * cos(delta angle)) 
        where kappa = 1/bandwidth_theta^2 (concentration parameter)
        Important is normalization is unnecessary for weights because we normalize by sum later and dvision later.
        
        link : https://link.springer.com/article/10.1007/s00500-012-0802-z, https://www.youtube.com/watch?v=2kIGEEzie1M
        """

        kappa = 1.0 / (self.bandwidth_theta ** 2) # link = https://mc-stan.org/docs/functions-reference/circular_distributions.html#von-mises-distribution ,  https://en.wikipedia.org/wiki/Von_Mises_distribution
        delta_angle = self._circ_dist(theta1, theta2)
        return np.exp(kappa * np.cos(delta_angle))

    
    def _gaussian_kernel(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        Gaussian kernel for linear (radial) data.
        
        K(r1, r2) = exp(-||r1 - r2||^2 / (2 * bandwidth_r^2))
        """
        dist_sq = (r1 - r2) ** 2
        return np.exp(-dist_sq / (2 * self.bandwidth_r ** 2)) # I believe in Cheng, Y. (1995) it is only np.exp(-dist_sq) but online I found it to be like my implementation 
    



    # -------------------------
    # Mean-shift core algorithm
    # -------------------------    
    def _mean_shift_step(self, point: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Perform single mean shift iteration.
        
        The new position is the weighted mean of all data points,
        where weights are determined by the kernel function.
        
        m(x) = sum( K(x_i - x) * x_i )/ sum (K(x_i - x))

        Parameters:
        point : shape (2,) is equal to [theta, r]
        data : All data points with shape (n, 2) 
        """

        thetas = data[:, 0]
        rs = data[:, 1]

        #Compute total weight by multiplying two kernel together. 
        theta_kernel = self._von_mises_kernel(point[0],thetas)
        r_kernel = self._gaussian_kernel(point[1],rs)
        weights = theta_kernel * r_kernel

        if np.sum(weights) < 1e-10: # so If weights are almost 0
            return point
        weights = weights / np.sum(weights)

        theta_new = self._circular_mean(data[:, 0], weights)
        r_new = np.sum(weights *rs) #Linear Mean

        return np.array([theta_new, r_new]) #shape (2,)
    

    # Mode mergig and clustering
    # -------------------------
    def _mean_shift_modes(self, start_point: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Find mode by iteratively applying mean shift until convergence.
        Starting from a given point, repeatedly shift toward higher density
        regions until we reach maximum iteration number or we reached convergence (shift is less than the tolerance)

        This function returns the converged mode location  [theta, r]
        """
        point = start_point
        iteration_count = 0

        #loop over the max amount of iterations allowed if we dont reach the steady mode
        for iteration in range(self.max_iter): 
            # Perform mean shift step
            new_point = self._mean_shift_step(point, data)

            # Compute shift magnitude
            theta_shift = self._circ_dist(new_point[0], point[0])
            r_shift = np.abs(new_point[1] - point[1])

            #Check if we reached the tolerance level then we dont have to keep doing anything else
            shift_magnitude = np.sqrt((theta_shift/self.bandwidth_theta)**2 + (r_shift/self.bandwidth_r)**2)
            point = new_point
            iteration_count = iteration + 1

            # Check convergence
            if shift_magnitude < self.tol:
                break

        return point, iteration_count
    
    
    def _merge_modes(self, modes: np.ndarray, 
                    theta_threshold: float, 
                    r_threshold: float) -> np.ndarray:
        
        """ #TODO better docstring 
        Merge modes that are close in BOTH dimensions.
        A mode is considered duplicate if:
        - theta distance < theta_threshold  AND r distance < r_threshold"""
        
        if len(modes) == 0:
            return modes
        
        unique_modes = [modes[0]]
        
        for mode in modes[1:]: #All modes EXCEPT the first one (I already added that) in unique_modes
            is_unique = True
            
            for existing_mode in unique_modes:
                theta_dist = self._circ_dist(mode[0], existing_mode[0])
                r_dist = np.abs(mode[1] - existing_mode[1])
                
                # Check BOTH conditions separately
                if theta_dist < theta_threshold and r_dist < r_threshold: #if this condition is true it means this a mode we have already seen, thus do not consider it.
                    is_unique = False
                    break
            
            if is_unique: # this means that this is "new/first" mode never seen before.
                unique_modes.append(mode)
        
        return np.array(unique_modes)
    

    def _collect_modes(self, data: np.ndarray):
        """Run mean-shift for each point and merge modes."""
        n_points = len(data)
        modes_list = []
        self.iteration_counts = []

        for i, p in enumerate(data):
            # just a small progress print so I can see it's alive, print every 100 points
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{n_points}")

            mode, iters = self._mean_shift_modes(p, data)
            modes_list.append(mode)
            self.iteration_counts.append(iters) #collect iteration for each point

        modes_array = np.array(modes_list)

        #merge mechanism 
        theta_threshold = self.bandwidth_theta   # tuned manually in radians
        r_threshold = self.bandwidth_r         # tuned manually, in same units as r
        merged= self._merge_modes(modes_array, theta_threshold, r_threshold)
        return modes_array, merged

  

    def _mean_shift_full_clustering(self, point_modes: np.ndarray, merged_modes: np.ndarray):
        """Assign each point to its nearest merged mode using normalized distance."""
        #TODO better docstring


        labels = np.zeros(len(point_modes), dtype=int)

        for i, mode in enumerate(point_modes):
            min_mode_idx = 0
            least_dist = float('inf')

            for j, m in enumerate(merged_modes):
                # Normalize each dimension by its bandwidth
                theta_dist_norm = self._circ_dist(mode[0], m[0]) / self.bandwidth_theta
                r_dist_norm = abs(mode[1] - m[1]) / self.bandwidth_r
                
                
                d = np.sqrt(theta_dist_norm**2 + r_dist_norm**2) #Logic: Both are without dimensions so it safe to combine thme

                if d < least_dist:
                    least_dist = d
                    min_mode_idx = j

            labels[i] = min_mode_idx

        return labels



    def fit(self, data: np.ndarray):
        """
        This function as a whole implements the blurring process described in the paper which is that
        each data point is shifted to its local mode and points converging
        to the same mode form a cluster.
        """
        start_time = time.time()
        
        # 1. Find mode for each data point (blurring process) & merge modes
        point_modes, merged_modes = self._collect_modes(data)
        self.modes = merged_modes

        # 2. Assign labels (execute the clustering step)
        self.labels = self._mean_shift_full_clustering(point_modes, merged_modes)

        self.total_time = time.time() - start_time

        # 3. collect cluster stats
        self._cluster_stats(data)


    def _cluster_stats(self, data: np.ndarray):
        self.cluster_params = []
        for cluster_id in range(len(self.modes)):
            # Get points in this cluster
            mask = self.labels == cluster_id
            cluster_data = data[mask]
            
            # Compute means
            theta_mean = self._circular_mean(cluster_data[:, 0], np.ones(len(cluster_data)))
            r_mean = np.mean(cluster_data[:, 1])
            
            # Compute circular variance for theta
            # Circular variance = 1 - |mean resultant length|, link : https://www.ebi.ac.uk/thornton-srv/software/PROCHECK/nmr_manual/man_cv.html
            cos_sum = np.sum(np.cos(cluster_data[:, 0]))
            sin_sum = np.sum(np.sin(cluster_data[:, 0]))
            mean_resultant_length = np.sqrt(cos_sum**2 + sin_sum**2) / len(cluster_data)
            theta_var = 1 - mean_resultant_length
            
            # Compute variance for r
            r_var = np.var(cluster_data[:, 1])
            
            # Compute circular-linear correlation (Mardia & Jupp method) "Linear-Circular Correlation Coefficients and Rhythmometry" or the paper "Measures and Models for Angular Correlation and Angular-Linear Correlation"
            # Another Reference: https://jdblischak.github.io/fucci-seq/circ-simulation-correlation.html
            cos_theta = np.cos(cluster_data[:, 0])
            sin_theta = np.sin(cluster_data[:, 0])
            
            rxs = np.corrcoef(cluster_data[:, 1], sin_theta)[0, 1]
            rxc = np.corrcoef(cluster_data[:, 1], cos_theta)[0, 1]
            rcs = np.corrcoef(cos_theta, sin_theta)[0, 1]
            
            numerator = rxs**2 + rxc**2 - 2 * rxs * rxc * rcs
            denominator = 1 - rcs**2
            
            if denominator > 0:
                circ_lin_corr = np.sqrt(numerator / denominator)
            else:
                circ_lin_corr = 0.0
            
            #Save the reslts in a dictionary params
            params = {
                'mode': self.modes[cluster_id],
                'mean_theta': theta_mean,
                'mean_r': r_mean,
                'var_theta': theta_var,
                'var_r': r_var,
                'circ_lin_corr': circ_lin_corr,
                'n_points': len(cluster_data)
            }
            
            self.cluster_params.append(params)

        # Add iteration stats
        self.cluster_params.append({
            'avg_iterations': np.mean(self.iteration_counts),
            'median_iterations': np.median(self.iteration_counts),
            'max_iterations': np.max(self.iteration_counts),
            'total_time': self.total_time
        })


   
    # -------------------------
    # Plotting
    # -------------------------    

    def visualize(self, data: np.ndarray, save_path: str = None):
        """
        Visualize clusters in:
         1. Polar view
         2. Cartesian xy view 
         3. Cluster statistics
        """

        #Create the figure
        fig = plt.figure(figsize=(16, 5))



        # ----------------------------------------------------
        # (1) POLAR PLOT 
        # ----------------------------------------------------
        ax1 = plt.subplot(131, projection='polar')

        # Plot each cluster with different color
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.modes)))

        for cluster_id in range(len(self.modes)):
            mask = self.labels == cluster_id
            cluster_data = data[mask]

            ax1.scatter(cluster_data[:, 0], cluster_data[:, 1],
                        c=[colors[cluster_id]], alpha=0.6, s=30,
                        label=f'Cluster {cluster_id+1}')

            mode = self.modes[cluster_id]
            ax1.scatter(mode[0], mode[1],
                        c=[colors[cluster_id]], s=200, marker='*',
                        edgecolors='black', linewidths=2)

        ax1.set_title('Polar View: Clusters and Modes', fontsize=12, pad=20)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1))

        # ----------------------------------------------------
        # (2) CARTESIAN PLOT 
        # ----------------------------------------------------
        ax2 = plt.subplot(132)

        # Convert data to x, y
        x = data[:, 1] * np.cos(data[:, 0])
        y = data[:, 1] * np.sin(data[:, 0])

        # Convert modes to x, y
        mx = self.modes[:, 1] * np.cos(self.modes[:, 0])
        my = self.modes[:, 1] * np.sin(self.modes[:, 0])

        # Plot clusters
        for cluster_id in range(len(self.modes)):
            mask = self.labels == cluster_id
            ax2.scatter(
                x[mask], y[mask],
                s=30, alpha=0.7,
                color=colors[cluster_id],
                label=f"Cluster {cluster_id+1}"
            )

        # Plot merged modes
        ax2.scatter(mx, my, s=160, c='k', marker='X', label='Modes')

        # Aesthetics
        ax2.axhline(0, linewidth=0.5, color='gray')
        ax2.axvline(0, linewidth=0.5, color='gray')

        ax2.set_xlabel("x = r*cos(theta)", fontsize=11)
        ax2.set_ylabel("y = r*sin(theta)", fontsize=11)
        ax2.set_title("Cartesian View: Clusters and Modes", fontsize=12)

        ax2.legend(
            loc='lower left',
            bbox_to_anchor=(1.05, 0.5),
            borderaxespad=1,
            markerscale=0.8
        )

        ax2.grid(True, alpha=0.3)

        # ----------------------------------------------------
        # (3) CLUSTER STATISTICS
        # ----------------------------------------------------

        #creat a table 
        ax3 = plt.subplot(133)
        ax3.axis('off')

        #Start text
        stats_text = "Cluster Statistics\n" + "="*50 + "\n\n"

        for i, params in enumerate(self.cluster_params[:-1]):
            stats_text += (
                f"Cluster {i+1}:\n"
                f"  Size: {params['n_points']} points\n"
                f"  Mode: θ={params['mode'][0]:.3f}, r={params['mode'][1]:.3f}\n"
                f"  Mean: θ={params['mean_theta']:.3f}, r={params['mean_r']:.3f}\n"
                f"  Var(θ): {params['var_theta']:.4f}\n"
                f"  Var(r): {params['var_r']:.4f}\n"
                f"  Corr(θ,r): {params['circ_lin_corr']:.4f}\n\n"

            )
        
        # Add convergence stats
        conv_stats = self.cluster_params[-1]
        stats_text += (
            "="*50 + "\n"
            "Convergence:\n"
            f"  Avg iters: {conv_stats['avg_iterations']:.1f}\n"
            f"  Median: {conv_stats['median_iterations']:.0f}\n"
            f"  Max: {conv_stats['max_iterations']}\n"
            f"  Time: {conv_stats['total_time']:.5f}s\n"
        )

        ax3.text(
            0.1, 0.95, stats_text,
            transform=ax3.transAxes,
            fontsize=9,
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()






# -------------------------
# Main: process files individually using visualize()
# -------------------------
def main():

    results_dir = "ResultsFig"
    os.makedirs(results_dir, exist_ok=True)

    print(" Start processing the CSV files")

    filenames = ["data1.csv", "data2.csv", "data3.csv", "data4.csv"]

    for f in filenames:
        basename = os.path.splitext(os.path.basename(f))[0]
        print(f"\n Processing {basename}")

        # Load single file
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        data[:, 0] = data[:, 0] % (2 * np.pi)

        # Initialize MeanShift
        ms = MeanShiftCircularLinear(
            bandwidth_theta=0.15, #TUNED manually
            bandwidth_r=0.65,  #TUNED manually
            max_iter=300, #TUNED manually
            tol=1e-4 #TUNED manually
        )

        # Fit model
        ms.fit(data)


        #Save the plots in the ResultsFig folder
        save_path = os.path.join(results_dir, f"{basename}_visualization.png")
        ms.visualize(data, save_path=save_path)


if __name__ == "__main__":
    main()
