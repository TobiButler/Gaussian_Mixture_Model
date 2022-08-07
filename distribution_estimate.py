'''
Author: Tobi Butler
Last Edited: 08/05/2022
Purpose: This module defines the Guassian_Mixture_Model class. When instantiated, a Guassian_Mixture_Model object is capable of fitting a mixture model to some provided 
dataset of samples using the Expectation-Maximization algorithm. Once fit, the object can predict the log-likelihood of any given sample within in the same dimensional 
space as the samples used to fit the model. It can also predict the conditional expected value of some feature(s) given some other feature(s): E(Y|X). This way, the fitted 
distribution can be used for regression. To use an instance of this class for classification, when calling the fit() method, the user must provide a list-like object of 
labels which is will use to adjust responsibility values during the expectation step so that P(sample_from_class_i | sample_from_class_j) == 0. That is, samples can only 
be generated from other samples within the same class. Once the model has been fit using this method, functions (like log_likelihood()) will return num_classes values, 
one for each class, instead of one.
'''
# import modules from std library:
import os
import sys
import math
from copy import deepcopy
from ctypes import ArgumentError

# import modules from third party libraries:
import torch as t
import numpy as np
import scipy.integrate as integrate


"""
This class contains the methods and attributes to fit a guassian mixture model to a given dataset and can use the fitted model to determine likelihoods of given samples 
    or probabilities of some given interval. It can also compute the expected value of the distribution along any feature(s).
"""
class Guassian_Mixture_Model():
    """
        Parameters:
        ----------
        consistent_variance (bool): Determines whether each cluster of the fit guassian mixture model has the same covariance matrix

        covariance_matrix_type (str): Determines how detailed M.L.E. covariance matrices are. Must be one of "full", "diagonal", or "scalar". Default value is "full". 
        Here are there effects:
            "full" - covariance matrices will be represented by full num_features x num_features matrices. This allows each covariance matrix to capture feature 
                covariance measurements in addition to feature variances
            "diagonal" - covariance matrices will be represented by the diagonal elements of the covariance matrix. This captures just the different variances 
                of the features and not the covariances. Features are assumed to be independent. This requires less memory and tensor computations than "full" 
                covariance matrices.
            "scalar" - covariance matrices are represented by the average of the diagonal elements of the covariance matrix. This does not capture covariance 
                measurements and does not fully specify the variance of each feature. It does, however, require the least amount of memory and tensor computations.

        dtype (type): Determines the datatype with which tensor computations are done. Must be one of t.float64, t.float32, or t.float16.

        Returns: an instance of this class
        """
    def __init__(self, consistent_variance:bool = False, covariance_matrix_type = "full", dtype:type = t.float64) -> None:
        # call super to set self.dtype, that's all
        if dtype not in {t.float64, t.float32, t.float16}:
            raise ArgumentError("The provided argument \"dtype\" must be one of \"t.float64\", \"t.float32\", or \"t.float16\".")
        self.dtype = dtype
        
        if covariance_matrix_type not in {"full", "diagonal", "scalar"}: raise ArgumentError("Constructor argument \"covariance_type\" must be one of \"full\", \"diagonal\", or \"scalar\".")
        self.covariance_matrix_type = covariance_matrix_type
        self.consistent_variance = consistent_variance
        self.device = None # keeps track of which device class-wide tensors (like cluster_centers) are stored
        self.fitted = False # becomes True when the fit() method returns successfully. Certain class methods will raise errors until this becomes True
        self.num_classes = 1 # the number of 
    
    """
    Computes the log-likelihood of a provided dataset of samples from the estimated guassian mixture model. The method fit() must have been called before this method 
        will work error free. If fit() was provided a tensor of class labels when called most recently (for classification purposes), then this method will return a tensor
        with shape (num_unique_classes, num_samples), in which entry i,j is the log-likelihood of sample j being generated from class i. The classes will indexed in order of 
        their labels, ascending. If fit() was called without class labels, then this method will return a tensor of shape (1, num_samples), in which each entry is the 
        log-likelihood of the sample being generated from the guassian mixture model.
    """
    def log_likelihood(self, samples: t.Tensor):
        """
        Parameters:
        ----------
        samples (t.Tensor): Samples (along the first dimension) whose log-likelihood of being drawn from the fit() distribution will be returned.

        Returns:
        ----------
        t.Tensor[float]: The log-likelihoods of the provided samples being drawn from the fit guassian mixture model. If fit() was called with class labels, then the 
            returned log-likelihoods will be of shape (num_unique_classes, num_samples). Otherwise the returned log-likelihoods will be of shape (1, num_samples).
        """
        if not self.fitted: raise ArgumentError("You must fit this kernel density estimate before using it to compute the likelihood of observing some samples.")
        # check method arguments:
        if samples.dim() != 2: raise ArgumentError("The provided \"samples\" must be a 2 dimensional torch.Tensor of shape (num_samples, num_features).")
        if str(samples.device) != self.device or samples.dtype != self.dtype: samples = samples.to(device=self.device, dtype=self.dtype)
        # compute likelihoods
        if self.covariance_matrix_type == 'full':
            if self.constant_variance: 
                log_probs_front = (-1/2)*t.slogdet(self.bandwidths)[1] # scalar
                log_probs = log_probs_front + t.stack(([(-1/2) * t.sum(t.abs(t.matmul(self.cluster_centers-self.cluster_centers[n,:], t.linalg.inv(self.bandwidths)) * self.cluster_centers-self.cluster_centers[n,:]),dim=1) for n in range(len(self.cluster_centers))])) # K x K
            else: 
                log_probs_front = (-1/2) * t.slogdet(t.permute(self.bandwidths), (2,0,1))[:,None] # num_clusters x 1
                diffs = t.ones(size=(len(self.cluster_centers), self.cluster_centers.shape[1], len(self.cluster_centers))) * self.cluster_centers[:,:,None] - t.permute(self.cluster_centers[:,:,None], (2,1,0)) # num_clusters x num_features x num_clusters
                log_probs = log_probs_front + (-1/2) * t.einsum('ijk, ijk -> ki', t.matmul(t.permute(diffs, (2,0,1)), t.permute(t.linalg.inv(self.bandwidths), (2,0,1))), t.permute(diffs, (2,0,1))) # K x N
        elif self.covariance_matrix_type == 'diagonal': 
            if self.constant_variance: 
                log_probs_front = (-1/2) * t.sum(t.log(self.bandwidths)) # scalar
                log_probs = log_probs_front + t.stack(([(-1/2) + t.einsum('ij, ij -> i', ((self.cluster_centers-self.cluster_centers[n,:]) / self.bandwidths, self.cluster_centers-self.cluster_centers[n,:])) for n in range(len(self.cluster_centers))])) # K x N
            else: 
                log_probs_front = (-1/2) * t.sum(t.log(self.bandwidths), dim=1)[:,None] # K x 1
                diffs = t.ones(size=(len(self.cluster_centers), self.cluster_centers.shape[1], len(self.cluster_centers))) * self.cluster_centers[:,:,None] - t.permute(self.cluster_centers[:,:,None], (2,1,0)) # num_clusters x num_features x num_bandwidths
                log_probs = log_probs_front + (-1/2) * t.einsum('ijk, ijk -> ki', (diffs / self.bandwidths[None,:,:], diffs)) # K x N
        else: # self.covariance_matrix_type == 'scalar'
            if self.constant_variance: 
                log_probs_front = (-self.cluster_centers.shape[1]/2) * t.log(self.bandwidths) # scalar
                log_probs = log_probs_front * t.stack(([(-1/2) * t.einsum('ij, ij -> i', ((self.cluster_centers-self.cluster_centers[n,:]) / self.bandwidths, self.cluster_centers-self.cluster_centers[n,:])) for n in range(len(self.cluster_centers))])) # K x N
            else: 
                log_probs_front = (-self.cluster_centers.shape[1]/2) *  t.log(self.bandwidths) # K x 1
                diffs = t.ones(size=(len(self.cluster_centers), self.cluster_centers.shape[1], len(self.cluster_centers))) * self.cluster_centers[:,:,None] - t.permute(self.cluster_centers[:,:,None], (2,1,0)) # num_clusters x num_features x num_bandwidths
                log_probs = log_probs_front + (-1/2) * t.einsum('ijk, ijk -> ki', (diffs / self.bandwidths.T[None,:,:], diffs)) # K x N
        
        # shift the log-likelihoods by a constant so that they don't underflow when converted to linear space. This is the same as multiplying the linear likelihoods by a scalar.
        shift = 30 - t.max(log_probs, dim=0)[0]

        # compute log-likelihoods 
        if self.num_classes != 1:
            log_probs = []
            for label in t.unique(self.class_labels, sorted = True):
                class_indices = (self.class_labels == label)
                class_cluster_probs = self.cluster_probabilities[class_indices]
                class_cluster_probs = class_cluster_probs / t.sum(class_cluster_probs)
                log_probs.append(t.log(t.sum(class_cluster_probs * t.exp(log_probs[class_indices, :]),dim=0)) - shift) # list of 1 x num_samples tensors
            log_probs = t.stack([log_probs]) # num_classes x num_samples
        else: log_probs = t.log(t.sum(self.cluster_probabilities * t.exp(log_probs + shift), dim=0)) - shift # 1 x num_samples
        
        return log_probs
    
    '''
    This method approximates the distribution of the samples provided using a gaussian mixture model with cluster centers at each sample point. 
        This method's arguments allow the user to put various prior distributions over the dataset in order to maximize cross validation. 
        The model is fit using a version of the EM algorithm.
    '''
    def fit(self, samples:t.Tensor, class_labels:t.Tensor[int] = None, prior_cluster_probabilities:t.Tensor = None, initial_covariance_matrix:t.Tensor = None, 
        cluster_concentration:float = 1, cluster_probability_shift:float = 0, minimum_total_log_covariance:float = -np.inf, minimum_directional_covariance:float = -np.inf, 
        responsibility_concentration:float = 1, convergence_flexibility:int = 1, convergence_change:float = 1, print_status = False, 
        attempt_gpu = True):
        """
        Parameters:
        ----------
        samples (t.Tensor): The dataset of samples to which this method fits a gaussian mixture model. Has shape (num_samples, num_features)

        class_labels (t.Tensor[int]): An optional tensor of labels that correspond to the samples provided. Provide this argument if you 
            want to use the fit gaussian mixture model for classification.

        prior_cluster_probabilities (t.Tensor): A prior for the mixture model's cluster probabilities. A good reason to provide this argument 
            is if you are fitting a second+ mixture model to your dataset of samples, using the reciprocal of your previous probability 
            measurements as this prior.

        initial_covariance_matrix (t.Tensor): The initial covariance matrix used for the distribution at each cluster center during the em 
            algorithm. If None is provided, then the initial covariance matrix is taken to be twice the M.L.E. covariance matrix for a 
            single multivariate normal distribution given the dataset of samples.

        cluster_concentration (float): Part of the dirichlet process prior. During each iteration of the EM algorithm, cluster probabilities 
            are raised to this value before being normalized again. This brings cluster probabilities closer together or pushes them further 
            apart depending on the value used.
        
        cluster_probability_shift (float): Part of the dirichlet process prior. During each iteration of the EM algorithm, cluster probabilities 
            are raised to the power of themselves raised to this value, then they are normalized so that they sum to one. This also pushes cluster 
            probabilities closer together or further apart depending on its value, and can adjust which cluster probs become the largest/smallest.

        minimum_total_log_covariance (float): The minimum value to which the log-determinant of any cluster covariance matrix is allowed to reach.

        minimum_directional_covariance (float): The minimum amount of covariance allowed along a single direction at each cluster

        responsibility_concentration (float): The dirichlet process prior for the responsibilities computed during the expectation step of the EM algorithm. 
            Adjust this value to adjust how quickly covariance matrices change and the EM algorithm converges. Larger values will cause cluster covariance 
            matrices to change more slowly and smaller values will cause cluster covariance matrices to change more rapidly.

        convergence_flexibility (int): The number of times that the EM algorithm is allowed to converge until we stop iterating. 
            The default value is 1.

        convergence_change (float): The amount by which the complete-log-likelihood must change upon each iteration of the EM algorithm 
            before it converges. The default minimum change is 1.

        print-status (bool): Determines whether print statements are output to the console during fitting. This can be helpful when trying 
            to tune hyperparameters.

        Returns:
        ----------
        float: The complete-log-likelihood of the provided samples using the fit gaussian mixture model.
        """
        # check method arguments:
        if minimum_directional_covariance < 1e-12: raise ArgumentError("The fit() method argument \"minimum_directional_covariance\" must be greater than or equal to 1e-12. This is nearly the minimum value (of about 1e-14) that can be computed by pytorch's singular value decomposition, which is used to determine directional covariance.")
        if samples.dim() != 2: raise ArgumentError("The provided \"samples\" must be a 2 dimensional torch.Tensor of shape (num_samples, num_features).")
        if class_labels is not None:
            if len(class_labels) != len(samples): raise ArgumentError("You must provide the same number of class_labels as samples, one label for each sample.")
            self.class_labels = t.tensor(class_labels, dtype=int)
            self.num_classes = len(t.unique(class_labels))
        self.responsibility_concentration = responsibility_concentration

        # determine which device to use for computations
        if attempt_gpu and t.cuda.is_available(): self.device = "cuda"
        else: self.device = "cpu"
        
        # attempt em algorithm, catching memory errors, other errors, if they arise
        try: 
            # put tensors onto correct device and save samples for computing likelihoods (they are the density estimate's cluster centers)
            self.cluster_centers:t.Tensor = samples.to(device=self.device, dtype=self.dtype)
            if self.num_classes != 1 and (self.class_labels.dtype != self.dtype or str(self.class_labels.device) != self.device): self.class_labels = self.class_labels.to(device = self.device, dtype = self.dtype)
            
            # instantiate cluster center probs and variances
            if prior_cluster_probabilities is None: self.cluster_probabilities = t.ones(size = (samples.shape[0],1), device=self.device, dtype=self.dtype) / samples.shape[0] # N x 1
            else: 
                if prior_cluster_probabilities.dim() == 1: self.cluster_probabilities = prior_cluster_probabilities[:,None]
                elif prior_cluster_probabilities.dim() == 2: 
                    if prior_cluster_probabilities.shape[1] != 1: raise ArgumentError(f"The argument \"prior_cluster_probabilities\" provided to the fit() method must have shape (num_samples,) or (num_samples, 1). fit() was given an argument with shape ({prior_cluster_probabilities.shape})")
                    self.cluster_probabilities = prior_cluster_probabilities
                else: raise ArgumentError(f"The argument \"prior_cluster_probabilities\" provided to the fit() method must have shape (num_samples,) or (num_samples, 1). fit() was given an argument with shape ({prior_cluster_probabilities.shape})")
            
            # instantiate initial covariance matrix
            if initial_covariance_matrix is not None: 
                if initial_covariance_matrix.dtype is not self.dtype or str(initial_covariance_matrix.device) != self.device: initial_covariance_matrix = initial_covariance_matrix.to(device=self.device, dtype=self.dtype)
                bandwidth = initial_covariance_matrix
            else:
                mean = t.sum(self.cluster_centers*self.cluster_probabilities, dim=0)
                diffs = self.cluster_centers - mean
                bandwidth = 2 * t.matmul(diffs.T, diffs*self.cluster_probabilities)
            
            # check that the initial covariance matrix is not ill-conditioned
            if t.isinf(t.slogdet(bandwidth)[1]): raise ArgumentError("The initial mle covariance matrix of the given dataset is too large. Provide a smaller custom prior covariance matrix.  We recommend trying the average of the diagonals of the mle covariance matrix. You will have to compute this yourself..")
            if t.isneginf(t.slogdet(bandwidth)[1]): raise ArgumentError("The initial covariance matrix provided is too small. Provide a larger custom prior covariance matrix. We recommend trying the average of the diagonals of the mle covariance matrix. You will have to compute this yourself.")
            
            # adjust how covariance matrices are stored depending on the arguments passed
            if self.covariance_matrix_type == 'full': # most detailed option, keeps track of covariances in addition to feature variances
                if self.constant_variance: self.bandwidths = bandwidth # keep track of a single full covariance matrix with shape (num_features, num_features)
                else: # keep track of a full covariance matrix for every cluster center. This requires num_samples x num_features x num_features memory (too much for most datasets working on most machines)
                    self.bandwidths = t.ones(size=(bandwidth.shape[0], bandwidth.shape[0], len(self.cluster_centers)), device=self.device, dtype=self.dtype) * bandwidth[:,:,None] # gives tensor of shape (num_features, num_features * num_clusters)
            elif self.covariance_matrix_type == 'diagonal': # keeps track of just the variances of each feature, so features are assumed to be independent
                bandwidth = t.diag(bandwidth) # gets just the feature variances
                if self.constant_variance: self.bandwidths = bandwidth # has shape (num_features)
                else: self.bandwidths = t.ones(size=(len(self.cluster_centers), bandwidth.shape[0]), device=self.device, dtype=self.dtype) * bandwidth[None,:]  # gives tensor of shape (num_samples, n_features)
            else: # self.covariance_matrix_type == 'scalar': # This requires num_samples memory space
                bandwidth = t.mean(t.diag(bandwidth)) # average the diagonal covariance values to produce a single scalar.
                if self.constant_variance: self.bandwidths = bandwidth
                else: self.bandwidths = bandwidth * t.ones(size=(len(self.cluster_centers), 1), device=self.device, dtype=self.dtype) # gives tensor of dim (num_components x 1). Each row of sigma_inv corresponds to a single inverse covariance scalar for the i-th cluster
            
            if print_status: 
                print("initial bandwidth:")
                print(self.bandwidths)
                print("initial log-det: " + str(t.slogdet(self.bandwidths)[1]))
            
            # initial expectation step to set responsibilities and complete log likelihood
            responsibilities, L_comp_old = self._expectation(prior_cluster_probabilities, print_status) # K x N
            if print_status: print("First comp-log likelihood: " + str(L_comp_old))
            
            # check if any samples have a responsibility greater than 0.99. If so, that sample's likelihood is going to diverge too quickly, so we should increase "responsibility_concentration"
            if t.max(responsibilities) >= 0.99: raise ArgumentError("Sample responsibilities were ill-conditioned. Increase the \"responsibility_concentration\" argument and try again. We reccomend monitoring how quickly the responsibility values converge.")
            
            ###if (t.sum(t.abs(responsibilities - t.ones(size=(responsibilities.shape[0], 1), device=self.device, dtype=self.dtype) / responsibilities.shape[0]), dim=0)==0).any(): NOT USING THIS ANYMORE
            
            convergence_count = 0 # keeps track of how many times the em algorithm has converged
            count = 0 # keeps track of how many iterations the em algorithm has run for
            if self.constant_variance: converged_clusters = t.zeros(1).bool() # keeps track of how many cluster_centers have converged/had their covariance matrix reach a minimum determinant
            else: converged_clusters = t.zeros(self.cluster_centers.shape[0]).bool()
            while not t.all(converged_clusters):
                # compute cluster probabilities and apply dirichlet distribution prior:
                self.cluster_probabilities = (t.sum(responsibilities, dim=1) / len(self.cluster_centers))[:,None] # K x len(cluster_concentrations)
                self.cluster_probabilities = self.cluster_probabilities**(self.cluster_probabilities**cluster_probability_shift * cluster_concentration) # K x len(cluster_concentrations)
                self.cluster_probabilities = self.cluster_probabilities/t.sum(self.cluster_probabilities, dim=0) #* self.cluster_concentrations[:,0][None,None,:]/t.sum(self.cluster_concentrations[:,0]), dim=2)
                
                if print_status:
                    print("CLUSTER PROBS:")
                    print(self.cluster_probabilities)
                    print("Max cluster-prob: " + str(t.max(self.cluster_probabilities)))
                    print("Min cluster-prob: " + str(t.min(self.cluster_probabilities)))

                # maximization step: compute M.L.E. covariance matrices
                if self.constant_variance: # will compute the mle covariance matrix around each cluster center and then average them based on the original prior over the cluster centers
                    new_cov = t.zeros(size=self.bandwidths.size(), device=self.device, dtype=self.dtype)
                    for n in range(len(self.cluster_centers)):
                        temp_samples = self.cluster_centers - self.cluster_centers[n] # data centered around n-th sample
                        temp_cov = t.matmul(temp_samples.T, temp_samples*responsibilities[:,n][:,None])
                        if self.covariance_matrix_type == "diagonal": 
                            temp_cov = t.diag(temp_cov)
                        elif self.covariance_matrix_type == "scalar": 
                            temp_cov = t.mean(t.diag(temp_cov))
                        new_cov += temp_cov * prior_cluster_probabilities[n] # weight the covariance matrices by the original prior of the cluster centers
                    
                    # check that the average covariance matrix is not too small. Adjust it if it is.
                    new_cov = new_cov / minimum_directional_covariance
                    s,v,d = t.linalg.svd(new_cov)
                    v = t.maximum(v, t.tensor(1, device=self.device, dtype=self.dtype))
                    new_cov = t.matmul(s*v,d) * minimum_directional_covariance
                    
                    if t.slogdet(new_cov)[1] < minimum_total_log_covariance:
                        converged_clusters[0] = True
                        if t.isinf(t.slogdet(new_cov)[1]): 
                            new_cov = self.bandwidths
                        else:
                            new_cov = new_cov * t.exp((minimum_total_log_covariance-t.slogdet(new_cov)[1])/new_cov.shape[1])
                    self.bandwidths = new_cov
                else: pass # HERE NEED TO ADD THE NOT CONSISTANT_VARIANCE PART BACK IN. SHOULD BE EASY JUST DON"T AVERAGE
                    
                # END MAXIMIZATION STEP
                
                if print_status and self.consistent_variance:
                        print("New covariance matrix log-determinant: " + str(t.slogdet(self.bandwidths)[1]))
                
                # repeat expectation step
                responsibilities, L_comp_new = self._expectation(prior_cluster_probabilities, print_status)
                
                
                if print_status: print("New comp-log-likelihood: " + str(L_comp_new))
                if print_status: print(str(t.sum(converged_clusters.float().item())) + " cluster bandwidths have converged.")
                count += 1

                # check whether convergence conditions have been met
                if t.isinf(L_comp_new) or t.isnan(L_comp_new) or L_comp_new-L_comp_old < convergence_change:
                    convergence_count += 1
                    if convergence_count >= convergence_flexibility:
                        if print_status: print("EM algorithm has converged after " + str(count) + " iterations. Log-likelihood = " + str(L_comp_new.item()))
                        self.fitted = True
                        break
                else: convergence_count=0
                L_comp_old = L_comp_new
            
            if t.all(converged_clusters) and print_status: print("All cluster bandwidths converged to minimum value.")
            self.fitted=True
            return L_comp_new.item()
        except MemoryError: raise MemoryError(f"There is not enough space on device {self.device} to fit a mixture model with the dataset provided.")
        except OverflowError("OVERFLOW ERROR") as e: 
            print(e)
            return -math.inf
    # end fit()
    
    '''
    Approximate the area under the gaussian mixture model's pdf for one feature conditioned on values of the rest, using quadrature.
    '''
    def integrate(self, sample:t.Tensor, lower_bound:float, upper_bound:float, output_feature_index:int = 0, **kwargs):
        """
        Parameters:
        ----------
        sample (t.Tensor): The sample upon which one feature from the gaussian mixture model is conditioned. Should contain one less feature 
            than the samples to which this Gaussian_Mixture_Model was fit.

        lower_bound (float): The lower value from which integration of the desired feature begins.

        upper_bound (float): The upper value to which the desired feature is integrated

        output_feature_index (int): The index of the feature from this Gaussian_Mixture_Model's cluster_centers which is to be integrated 
            over conditioned on the provided sample. Default feature is that with index zero.
        Returns:
        ----------
        float: The probability of the desired feature being between the provided lower_bound and upper_bound conditioned on the provided sample.
        """
        if self.num_classes != 1: raise ArgumentError("This Guassin_Mixture_Model cannot be integrated since it has been fit for classification.")
        if sample.shape[1] != self.cluster_centers.shape[1]-1: raise ArgumentError("The sampled upon which the distribution is conditioned for integration must be one dimension smaller than the samples used to fit the gaussian mixture model.")
        if lower_bound >= upper_bound: raise ArgumentError("Lower bound must be less than the upper bound.")
        
        # convert distribution information from torch Tensors to numpy ndarrays
        cluster_centers = deepcopy(self.samples).detach().cpu().numpy()
        cluster_center_probs = deepcopy(t.flatten(self.cluster_probabilities)).detach().cpu().numpy()
        cluster_bandwidths = deepcopy(self.bandwidths).detach().cpu().numpy()
        
        # compute conditional mean and variance for the output feature given the sample provided
        conditional_means = np.zeros(size=(1,len(cluster_centers)))
        for n in range(len(self.cluster_centers)):
            conditional_means[n] = cluster_centers[n][output_feature_index] + np.dot(np.dot(cluster_bandwidths[output_feature_index,:output_feature_index:], np.linalg.inv(cluster_bandwidths[:output_feature_index:,:output_feature_index:])), sample-cluster_centers[n][:output_feature_index:])
        if self.consistent_variance: conditional_variances = cluster_bandwidths[output_feature_index,output_feature_index] - np.dot(np.dot(cluster_bandwidths[output_feature_index,:output_feature_index:], np.linalg.inv(cluster_bandwidths[:output_feature_index:,:output_feature_index:])), cluster_bandwidths[output_feature_index,:output_feature_index:])
        else: 
            conditional_variances = np.zeros(size=(1,len(cluster_centers)))
            for n in range(len(cluster_centers)):
                conditional_variances[n] = cluster_bandwidths[n][output_feature_index,output_feature_index] - np.dot(np.dot(cluster_bandwidths[n][output_feature_index,:output_feature_index:], np.linalg.inv(cluster_bandwidths[n][:output_feature_index:,:output_feature_index:])), cluster_bandwidths[n][output_feature_index,:output_feature_index:])

        # use substitution to convert indefinite integral to definite integral. We use y = log(x/(1-x)) which has inverse x = 1/(1+exp(-y))
        lower_bound = 1 / (1 + math.exp(-lower_bound))
        upper_bound = 1 / (1 + math.exp(-upper_bound))

        # define the function which we will integrate with respect to substituted variable
        def func(t):
            y = np.log(t/(1-t))
            dy = 1/(t-t**2)
            return np.sum((1/((2*np.pi*conditional_variances)**(1/2))) * np.exp((-1/2) * ((y-cluster_centers)**2) / conditional_variances) * cluster_center_probs) * dy
        
        # integrate using scipy library's quadrature
        return integrate.quadrature(func, lower_bound, upper_bound, **kwargs)[0]
    # end integrate()

    """
    Computes the expected value of a set of features from the fit gaussian mixture model given a sample of the remaining features upon which the model is 
        conditioned. Use this method for regression prediction by conditioning the distribution over your input features and computing the expected value 
        over the remaining (output) features.
    """
    def mean(self, samples:t.Tensor, features:list[int]=None, **kwargs) -> t.Tensor:
        """
        Parameters:
        ----------
        samples (t.Tensor): The samples upon which the gaussian mixture model will be conditioned. Has shape (num_samples, num_conditioning_features) 

        features (list[int]): The features along which the expected value will be computed.

        NOTE: The number of features in each sample provided and the number of feature indices in "features" must sum together to total the number of 
            features used to fit the gaussian mixture model, so we must have: samples.shape[1] + len(features) == self.cluster_centers.shape[1]
        
        Returns:
        ----------
        t.Tensor of shape (num_samples, num_features): The expected value(s) of the fit gaussian mixture model conditioned over the sample(s) provided.
        """
        # check method arguments
        if samples is None or features is None: return t.sum(self.cluster_probabilities*self.cluster_centers) # expected value of each cluster times its prob
        elif samples.shape[1] != len(features): raise ArgumentError(f"The number of features in each sample ({samples.shape[1]}) must match the number of indices provided in \"features\" ({len(features)})")
        if samples.dtype is not self.dtype or str(samples.device) != str(self.cluster_centers.device): samples = samples.to(device=str(self.cluster_centers.device), dtype=self.dtype)
        
        # compute list of conditional features that are opposite of the feature list provided
        conditional_features = sorted(list(set(range(0,self.cluster_centers.shape[1])) - set(features)))
        outputs = t.zeros(size=(len(samples), self.cluster_centers.shape[1]-len(features)), device=str(self.cluster_centers.device), dtype=self.dtype) # keeps track of running sum. Has shape (num_samples, num_conditional features)
        
        # below we compute conditional cluster probabilities with which we weight each cluster's conditional mean to produce the mean of the conditional gaussian mixture model.
        conditional_cluster_probs = []
        if self.constant_variance:
            if self.covariance_matrix_type is "full":
                
                # pull out the conditioned part of the covariance matrix
                covariance = self.bandwidths[features, features]

                # for each cluster center, compute conditional probabilities of all of the samples provided given that cluster
                for n in range(len(self.cluster_centers)): 
                    diffs = samples - self.cluster_centers[n,:]
                    probs = -(1/2) * t.slogdet(covariance)[1] + (-1/2) * t.sum(t.abs(t.matmul(diffs, t.linalg.inv(covariance)) * diffs),dim=1)
                    shift = 30 - t.max(probs, dim=0)[0]
                    probs = t.exp(probs + shift)
                    conditional_cluster_probs.append(probs)
                
                # weight conditional sample probs by prior cluster probs to form conditional cluster probs
                conditional_cluster_probs = self.cluster_probabilities * t.tensor(np.array(conditional_cluster_probs), device=str(self.cluster_centers.device), type=self.dtype).T # N x K
                conditional_cluster_probs = conditional_cluster_probs / t.sum(conditional_cluster_probs,dim=0)
                
                # sum posterior probabilities over the clusters
                for n in range(len(self.cluster_centers)): 
                    outputs += conditional_cluster_probs[:,n] * (self.cluster_centers[n,conditional_features] + t.matmul(samples-self.cluster_centers[n,features] / t.diag(self.bandwidths)[features], self.bandwidths[features, conditional_features])) # N x num_given_features
            if self.covariance_matrix_type is "diagonal":
                covariance = self.bandwidths[features]
                for n in range(len(self.cluster_centers)): 
                    diffs = samples - self.cluster_centers[n,:]
                    probs = -(1/2) * t.sum(t.log(covariance)) + (-1/2) * t.sum(t.abs(diffs**2 / covariance),dim=1)
                    shift = 30 - t.max(probs, dim=0)[0]
                    probs = t.exp(probs + shift)
                    conditional_cluster_probs.append(probs)
                conditional_cluster_probs = self.cluster_probabilities * t.tensor(np.array(conditional_cluster_probs), device=str(self.cluster_centers.device), type=self.dtype).T # N x K
                conditional_cluster_probs = conditional_cluster_probs / t.sum(conditional_cluster_probs,dim=0)
                for n in range(len(self.cluster_centers)):
                    outputs += conditional_cluster_probs[:,n] * (self.cluster_centers[n,conditional_features] + samples-self.cluster_centers[n,features] / self.bandwidths[features] * self.bandwidths[features, conditional_features])
            if self.covariance_matrix_type is "scalar":
                for n in range(len(self.cluster_centers)):
                    diffs = samples - self.cluster_centers[n,:]
                    probs = -(len(self.cluster_centers.shape[1])/2) * t.log(self.bandwidths) + (-1/2) * t.sum(t.abs(diffs**2 / self.bandwidths),dim=1)
                    shift = 30 - t.max(probs, dim=0)[0]
                    probs = t.exp(probs + shift)
                    conditional_cluster_probs.append(probs)
                conditional_cluster_probs = self.cluster_probabilities * t.tensor(np.array(conditional_cluster_probs), device=str(self.cluster_centers.device), type=self.dtype).T # N x K
                conditional_cluster_probs = conditional_cluster_probs / t.sum(conditional_cluster_probs,dim=0)
                for n in range(len(self.cluster_centers)):
                    outputs += conditional_cluster_probs[:,n] * self.cluster_centers[n,conditional_features] + samples-self.cluster_centers[n,features]
        else:
            if self.covariance_matrix_type is "full":
                conditional_cluster_probs = []
                for n in range(len(self.cluster_centers)):
                    covariance = self.bandwidths[features, features]
                    diffs = samples - self.cluster_centers[n,:]
                    probs = -(1/2) * t.slogdet(covariance)[1] + (-1/2) * t.sum(t.abs(t.matmul(diffs, t.linalg.inv(covariance)) * diffs),dim=1)
                    shift = 30 - t.max(probs, dim=0)[0]
                    probs = t.exp(probs + shift)
                    conditional_cluster_probs.append(probs)
                conditional_cluster_probs = self.cluster_probabilities * np.array(conditional_cluster_probs).T # N x K
                conditional_cluster_probs = conditional_cluster_probs / np.sum(conditional_cluster_probs,axis=0)
                for n in range(len(self.cluster_centers)):
                    outputs += conditional_cluster_probs[:,n] * (self.cluster_centers[n,conditional_features] + t.matmul(samples-self.cluster_centers[n,features] / t.diag(self.bandwidths)[features], self.bandwidths[features, conditional_features])) # N x num_given_features
            if self.covariance_matrix_type is "diagonal":
                for n in range(len(self.cluster_centers)):
                    outputs += self.cluster_probabilities[n] * self.cluster_centers[n,conditional_features] + samples-self.cluster_centers[n,features] / self.bandwidths[features] * self.bandwidths[features, conditional_features]
            if self.covariance_matrix_type is "scalar":
                for n in range(len(self.cluster_centers)):
                    outputs += self.cluster_probabilities[n] * self.cluster_centers[n,conditional_features] + samples-self.cluster_centers[n,features] / self.bandwidths * self.bandwidths
        return outputs
    # end mean()
    
    '''
    This method carries out the expectation step of the EM algorithm, using the current cluster probability values and cluster 
        covariances to compute responsibilities (posterior probabilities of cluster centers given samples).
    '''
    def _expectation(self, prior_cluster_probs:t.Tensor, print_status:bool):
        # compute conditional probabilities, P(sample_j | sample_i)
        if self.covariance_matrix_type == 'full':
            if self.constant_variance: 
                log_probs_front = (-1/2)*t.slogdet(self.bandwidths)[1] # scalar
                log_probs = log_probs_front + t.stack(([(-1/2) * t.sum(t.abs(t.matmul(self.cluster_centers-self.cluster_centers[n,:], t.linalg.inv(self.bandwidths)) * self.cluster_centers-self.cluster_centers[n,:]),dim=1) for n in range(len(self.cluster_centers))])) # K x K
            else: 
                log_probs_front = (-1/2) * t.slogdet(t.permute(self.bandwidths), (2,0,1))[:,None] # num_clusters x 1
                diffs = t.ones(size=(len(self.cluster_centers), self.cluster_centers.shape[1], len(self.cluster_centers))) * self.cluster_centers[:,:,None] - t.permute(self.cluster_centers[:,:,None], (2,1,0)) # num_clusters x num_features x num_clusters
                log_probs = log_probs_front + (-1/2) * t.einsum('ijk, ijk -> ki', t.matmul(t.permute(diffs, (2,0,1)), t.permute(t.linalg.inv(self.bandwidths), (2,0,1))), t.permute(diffs, (2,0,1))) # K x N
        elif self.covariance_matrix_type == 'diagonal': 
            if self.constant_variance: 
                log_probs_front = (-1/2) * t.sum(t.log(self.bandwidths)) # scalar
                log_probs = log_probs_front + t.stack(([(-1/2) + t.einsum('ij, ij -> i', ((self.cluster_centers-self.cluster_centers[n,:]) / self.bandwidths, self.cluster_centers-self.cluster_centers[n,:])) for n in range(len(self.cluster_centers))])) # K x N
            else: 
                log_probs_front = (-1/2) * t.sum(t.log(self.bandwidths), dim=1)[:,None] # K x 1
                diffs = t.ones(size=(len(self.cluster_centers), self.cluster_centers.shape[1], len(self.cluster_centers))) * self.cluster_centers[:,:,None] - t.permute(self.cluster_centers[:,:,None], (2,1,0)) # num_clusters x num_features x num_bandwidths
                log_probs = log_probs_front + (-1/2) * t.einsum('ijk, ijk -> ki', (diffs / self.bandwidths[None,:,:], diffs)) # K x N
        else: # self.covariance_matrix_type == 'scalar'
            if self.constant_variance: 
                log_probs_front = (-self.cluster_centers.shape[1]/2) * t.log(self.bandwidths) # scalar
                log_probs = log_probs_front * t.stack(([(-1/2) * t.einsum('ij, ij -> i', ((self.cluster_centers-self.cluster_centers[n,:]) / self.bandwidths, self.cluster_centers-self.cluster_centers[n,:])) for n in range(len(self.cluster_centers))])) # K x N
            else: 
                log_probs_front = (-self.cluster_centers.shape[1]/2) *  t.log(self.bandwidths) # K x 1
                diffs = t.ones(size=(len(self.cluster_centers), self.cluster_centers.shape[1], len(self.cluster_centers))) * self.cluster_centers[:,:,None] - t.permute(self.cluster_centers[:,:,None], (2,1,0)) # num_clusters x num_features x num_bandwidths
                log_probs = log_probs_front + (-1/2) * t.einsum('ijk, ijk -> ki', (diffs / self.bandwidths.T[None,:,:], diffs)) # K x N
            
        # check if bandwidths have overflowed
        if t.isneginf(log_probs_front).any(): # check if any bandwidths become too large.
            raise ArgumentError(f"Encountered too large of bandwidths in the expectation step (maximum bandwidth encountered is {t.max(t.abs(self.bandwidths)).item()}) causing underflow (zero probabilities). We recommend normalizing the samples (to which the distribution is being fit) into a smaller window.")
        if (t.isinf(t.abs(log_probs_front)).any()): # check if any bandwidths become too small.
            raise OverflowError(f"Encountered too small of bandwidths in the expectation step (minimum bandwidth encountered is {t.min(t.abs(self.bandwidths)).item()}) causing overflow (inf probabilities). We recommend increasing the \"minimum_variance\" argument or normalizing the samples (to which the distribution is being fit) into a larger window.")
        
        # convert log-likelihoods to linear space
        log_probs = log_probs - t.max(log_probs, dim=0)[0] # ensure that all log_probs are <= zero
        log_probs = log_probs / self.responsibility_concentration # bring probs closer together or push them further apart depending on self.responsibility_concentration
        log_probs += 30 - t.max(log_probs,dim=0)[0]
        probs = t.exp(log_probs)

        # normalize likelihoods to produce probabilities
        probs = probs/t.sum(probs,dim=0)

        if print_status:
            print("Posterior probabilities:")
            print(probs)
            print("Min prob: " + str(t.min(probs)))
            print("Max prob: " + str(t.max(probs)))
        
        # compute posterior probabilities, P(cluster_i | sample_j)
        responsibilities = self.cluster_probabilities*probs

        # set the responsibilities for different classes to zero
        if self.num_classes != 1:
            for label in t.unique(self.class_lables):
                label_indices = (self.class_labels == label)
                opposing_label_indices = ~label_indices
                responsibilities[label_indices, opposing_label_indices] = t.zeros(size=(len(label_indices), len(opposing_label_indices)), dtype = self.dtype, device = self.device)

        # set the sum of the responsibilities for each sample to reflect the samples prior probability
        responsibilities = responsibilities / t.sum(responsibilities, dim = 0)
        if prior_cluster_probs is not None: responsibilities = responsibilities * t.flatten(prior_cluster_probs) # K x N

        if print_status:
            print("Responsibilities:")
            print(responsibilities)
            print("Min responsibility: " + str(t.min(responsibilities)))
            print("Max responsibility: " + str(t.max(responsibilities)))

        L_comp = t.einsum("ij, ij -> ", responsibilities, t.log(probs))
        return responsibilities, L_comp # returns the responsibilities and the complete log likelihood
    # end _expectation()