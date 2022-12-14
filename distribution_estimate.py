'''
Author: Tobi Butler
Last Edited: 08/05/2022
Purpose: This module defines the Gaussian_Mixture_Model class. When instantiated, a Gaussian_Mixture_Model object is capable of fitting a mixture model to some provided 
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
import traceback

# import modules from third party libraries:
import torch as t
import numpy as np
import scipy.integrate as integrate
import psutil
import pynvml
import gc


"""
This class contains the methods and attributes to fit a gaussian mixture model to a given dataset and can use the fitted model to determine likelihoods of given samples 
    or probabilities of some given interval. It can also compute the expected value of the distribution along any feature(s).
"""
class Gaussian_Mixture_Model():
    """
        Parameters:
        ----------
        consistent_variance (bool): Determines whether each cluster of the fit gaussian mixture model has the same covariance matrix

        covariance_matrix_type (str): Determines how detailed M.L.E. covariance matrices are. Must be one of "full", "diagonal", or "scalar". Default value is "full". 
        Here are the effects:
            "full" - covariance matrices will be represented by full num_features x num_features matrices. This allows each covariance matrix to capture feature 
                covariance measurements in addition to feature variances
            "diagonal" - covariance matrices will be represented by the diagonal elements of the covariance matrix. This captures just the different variances 
                of the features and not the covariances. Features are assumed to be independent. This requires less memory and tensor computations than "full" 
                covariance matrices.
            "scalar" - covariance matrices are represented by the average of the diagonal elements of the covariance matrix. This does not capture covariance 
                measurements and does not fully specify the variance of each feature. It does, however, require the least amount of memory and tensor computations.

        limited_memory (bool): determines whether to compute covariance matrices for the clusters all at once in parallel or one at a time (in series). By default, this is True and 
            covariance matrices are computed in series.
        
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
        ##self.limited_memory = limited_memory
        self.device = None # keeps track of which device class-wide tensors (like cluster_centers) are stored
        self.fitted = False # becomes True when the fit() method returns successfully. Certain class methods will raise errors until this becomes True
        self.num_classes = 1 # the number of 
    
    """
    Computes the log-likelihood of a provided dataset of samples from the estimated gaussian mixture model. The method fit() must have been called before this method 
        will work error free. If fit() was provided a tensor of class labels when called most recently (for classification purposes), then this method will return a tensor
        with shape (num_unique_classes, num_samples), in which entry i,j is the log-likelihood of sample j being generated from class i. The classes will indexed in order of 
        their labels, ascending. If fit() was called without class labels, then this method will return a tensor of shape (1, num_samples), in which each entry is the 
        log-likelihood of the sample being generated from the gaussian mixture model.
    """
    def log_likelihood(self, samples: t.Tensor, attempt_gpu: bool = True):
        """
        Parameters:
        ----------
        samples (t.Tensor): Samples (along the first dimension) whose log-likelihood of being drawn from the fit() distribution will be returned.

        Returns:
        ----------
        t.Tensor[float]: The log-likelihoods of the provided samples being drawn from the fit gaussian mixture model. If fit() was called with class labels, then the 
            returned log-likelihoods will be of shape (num_unique_classes, num_samples). Otherwise the returned log-likelihoods will be of shape (1, num_samples).
        """
        # the distribution needs to have been fit() before log-likelihoods can be computed
        if not self.fitted: raise ArgumentError("You must fit this kernel density estimate before using it to compute the likelihood of observing some samples.")
        # the provided "samples" argument must match the shape of the distribution's cluster centers:
        if samples.dim() != 2: raise ArgumentError("The provided \"samples\" must be a 2 dimensional torch.Tensor of shape (num_samples, num_features).")
        ##if str(samples.device) != self.device or samples.dtype != self.dtype: samples = samples.to(device=self.device, dtype=self.dtype)
        # compute log-likelihoods
        log_probs = self._log_likelihoods(samples=samples, attempt_gpu=attempt_gpu)
        log_probs += (-self.cluster_centers.shape[1]/2)*math.log(2*math.pi) # multiply likelihoods by normalization constant so that entire distribution sums to 1.
        
        # shift the log-likelihoods by a constant so that they don't underflow when converted to linear space. This is the same as multiplying the linear likelihoods by a scalar.
        shift = 30 - t.max(log_probs, dim=0)[0]
        
        # compute log-likelihoods for classes. 
        if self.num_classes != 1:
            log_probs_list = []
            for label in t.unique(self.class_labels, sorted = True):
                class_indices = t.flatten(self.class_labels == label)
                class_cluster_probs = self.cluster_probabilities[class_indices,:]
                class_cluster_probs = class_cluster_probs / t.sum(class_cluster_probs)
                log_probs_list.append(t.log(t.sum(class_cluster_probs * t.exp(log_probs[class_indices,:] + shift),dim=0)) - shift) # list of num_classes tensors, each with shape: 1 x num_samples
            log_probs = t.stack(log_probs_list) # num_classes x num_samples
        else: log_probs = t.log(t.sum(self.cluster_probabilities * t.exp(log_probs + shift), dim=0)) - shift # 1 x num_samples
        
        return log_probs
    
    '''
    This method approximates the distribution of the samples provided using a gaussian mixture model with cluster centers at each sample point. 
        This method's arguments allow the user to put various prior distributions over the dataset in order to maximize cross validation. 
        The model is fit using a version of the EM algorithm.
    '''
    def fit(self, samples:t.Tensor, class_labels:t.Tensor = None, prior_cluster_probabilities:t.Tensor = None, initial_covariance_matrix:t.Tensor = None, 
        cluster_concentration:float = 1, minimum_total_log_covariance:float = -np.inf, minimum_directional_covariance:float = 1e-12, 
        responsibility_concentration:float = 1, convergence_flexibility:int = 1, convergence_change:float = 1, max_iterations:int = 10, print_status:bool = False,
        attempt_gpu:bool = True):
        """
        Parameters:
        ----------
        samples (t.Tensor): The dataset of samples to which this method fits a gaussian mixture model. Has shape (num_samples, num_features)

        class_labels (t.Tensor[int]): An optional tensor of labels that correspond to the samples provided. Provide this argument if you 
            want to use the fit gaussian mixture model for classification.

        prior_cluster_probabilities (t.Tensor): A prior for the mixture model's cluster probabilities. A good reason to provide this argument 
            is if you are fitting a second+ mixture model to your dataset of samples, using the reciprocal of your previous probability 
            measurements as this prior. Can be shape (num_cluster_probs), (1,num_cluster_probs), or (num_cluster_probs,1).

        initial_covariance_matrix (t.Tensor): The initial covariance matrix used for the distribution at each cluster center during the em 
            algorithm. If None is provided, then the initial covariance matrix is taken to be twice the M.L.E. covariance matrix for a 
            single multivariate normal distribution given the dataset of samples.

        cluster_concentration (float): Part of the dirichlet process prior. During each iteration of the EM algorithm, cluster probabilities 
            are raised to this value before being normalized again. This brings cluster probabilities closer together or pushes them further 
            apart depending on the value used.

        minimum_total_log_covariance (float): The minimum value to which the log-determinant of any cluster covariance matrix is allowed to reach.

        minimum_directional_covariance (float): The minimum amount of covariance allowed along a single direction at each cluster

        responsibility_concentration (float): The dirichlet process prior for the responsibilities computed during the expectation step of the EM algorithm. 
            Adjust this value to adjust how quickly covariance matrices change and the EM algorithm converges. Larger values will cause cluster covariance 
            matrices to change more slowly and smaller values will cause cluster covariance matrices to change more rapidly.

        convergence_flexibility (int): The number of times that the EM algorithm is allowed to converge until we stop iterating. 
            The default value is 1.

        convergence_change (float): The amount by which the complete-log-likelihood must change upon each iteration of the EM algorithm 
            before it converges. The default minimum change is 1.
        
        max_iterations (int): The maximum number of times that the em algorithm loop is allowed to run before converging. This prevents ill-conditioned hyperparameters 
            from causing the loop to run for a long time or forever.

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
            if class_labels.dtype is not t.int: raise ArgumentError("The class labels passed to the fit() method of the Gaussian_Mixture_Model must be a torch Tensor of type int.")
            self.num_classes = len(t.unique(class_labels))
            self.class_labels = class_labels.to(device = "cpu", dtype = t.int32)
        self.responsibility_concentration = responsibility_concentration

        # determine which device to use for computations
        ##if attempt_gpu and t.cuda.is_available(): self.device = "cuda"
        ##else: self.device = "cpu"
        # attempt em algorithm, catching memory errors, other errors, if they arise
        try: 
            # save samples for computing likelihoods (they are the density estimate's cluster centers)
            self.cluster_centers:t.Tensor = samples.to(device="cpu", dtype=self.dtype)
            
            # instantiate cluster center probs and variances
            if prior_cluster_probabilities is not None:
                if prior_cluster_probabilities.dim() == 1: prior_cluster_probabilities = prior_cluster_probabilities[:,None]
                elif prior_cluster_probabilities.dim() == 2: 
                    if prior_cluster_probabilities.shape[1] != 1: raise ArgumentError(f"The argument \"prior_cluster_probabilities\" provided to the fit() method must have shape (num_samples,) or (num_samples, 1). fit() was given an argument with shape ({prior_cluster_probabilities.shape})")
                else: raise ArgumentError(f"The argument \"prior_cluster_probabilities\" provided to the fit() method must have shape (num_samples,) or (num_samples, 1). fit() was given an argument with shape ({prior_cluster_probabilities.shape})")
                self.cluster_probabilities = prior_cluster_probabilities.to(device = "cpu")
                self.cluster_probabilities = self.cluster_probabilities**(1/cluster_concentration)
                self.cluster_probabilities = self.cluster_probabilities/t.sum(self.cluster_probabilities, dim=0) # num_samples x 1
            else: 
                self.cluster_probabilities = t.ones(size = (len(samples),1), device="cpu", dtype=self.dtype) / len(samples) # num_samples x 1
                prior_cluster_probabilities = 1
            
            # instantiate initial covariance matrix
            if initial_covariance_matrix is not None: 
                if attempt_gpu and t.cuda.is_available(): initial_covariance_matrix = initial_covariance_matrix.to(device="cuda", dtype=self.dtype)
                else: initial_covariance_matrix = initial_covariance_matrix.to(device="cpu", dtype=self.dtype)
            else:
                if attempt_gpu and t.cuda.is_available(): bandwidth = self._cuda_initial_bandwidth()
                else:
                    mean = t.sum(self.cluster_centers*self.cluster_probabilities, dim=0)
                    diffs = self.cluster_centers - mean
                    bandwidth = 2 * t.matmul(diffs.T, diffs*self.cluster_probabilities)
            
            # check that the initial covariance matrix is not too small
            s,v,d = t.linalg.svd(bandwidth)
            v = t.maximum(v, t.tensor(minimum_directional_covariance, device=bandwidth.device, dtype=self.dtype))
            bandwidth = t.matmul(s*v,d)
            if t.slogdet(bandwidth)[1] < minimum_total_log_covariance: bandwidth = bandwidth * t.exp((minimum_total_log_covariance-t.slogdet(bandwidth)[1])/self.cluster_centers.shape[1]) # if it has underflowed, must replace it with the previous estimate
            if str(bandwidth.device) != "cpu": bandwidth = bandwidth.cpu()

            # adjust how covariance matrices are stored depending on the arguments passed
            if self.covariance_matrix_type == 'full': # most detailed option, keeps track of covariances in addition to feature variances
                if self.consistent_variance: self.bandwidths = bandwidth # keep track of a single full covariance matrix with shape (num_features, num_features)
                else: # keep track of a full covariance matrix for every cluster center. This requires num_samples x num_features x num_features memory (too much for most datasets working on most machines)
                    ###self.bandwidths = t.ones(size=(len(self.cluster_centers), self.cluster_centers.shape[1], self.cluster_centers.shape[1]), device=self.device, dtype=self.dtype) * bandwidth[:,:,None] # gives tensor of shape (num_features, num_features * num_clusters)
                    self.bandwidths = bandwidth[None,:,:]
            elif self.covariance_matrix_type == 'diagonal': # keeps track of just the variances of each feature, so features are assumed to be independent
                bandwidth = t.diag(bandwidth) # gets just the feature variances
                if self.consistent_variance: self.bandwidths = bandwidth[None,:] # has shape (1,num_features)
                else: self.bandwidths = t.ones(size=(len(self.cluster_centers), self.cluster_centers.shape[1]), device="cpu", dtype=self.dtype) * bandwidth[None,:]  # gives tensor of shape (num_clusters, n_features)
            else: # self.covariance_matrix_type == 'scalar': # This requires num_samples memory space
                bandwidth = t.mean(t.diag(bandwidth)).unsqueeze(dim=0) # average the diagonal covariance values to produce a single scalar.
                if self.consistent_variance: self.bandwidths = bandwidth[None,:] # has shape (1,1)
                else: self.bandwidths = bandwidth * t.ones(size=(len(self.cluster_centers), 1), device="cpu", dtype=self.dtype) # gives tensor of dim (num_clusters x 1). Each row corresponds to a single inverse covariance scalar for the i-th cluster
            
            if print_status: 
                print("initial bandwidths:")
                print(self.bandwidths)
                if self.covariance_matrix_type == "full": print("initial log-dets: " + str(t.slogdet(self.bandwidths)[1]))
                elif self.covariance_matrix_type =="diagonal": print("initial log-dets: " + str(t.sum(t.log(self.bandwidths),dim=1)))
                else: print("initial log-dets: " + str(len(self.cluster_centers)*t.log(self.bandwidths)))
            
            # initial expectation step to set responsibilities and complete log likelihood
            responsibilities, L_comp_old = self._expectation(print_status, attempt_gpu) # responsibilities have shape num_samples x num_samples and sum to 1 along columns
            if print_status: print("First comp-log likelihood: " + str(L_comp_old))
            
            convergence_count = 0 # keeps track of how many times the em algorithm has converged
            count = 0 # keeps track of how many iterations the em algorithm has run for
            if self.consistent_variance: converged_clusters = t.zeros(1).bool() # keeps track of how many cluster_centers have converged/had their covariance matrix reach a minimum determinant
            else: converged_clusters = t.zeros(self.cluster_centers.shape[0]).bool()
            while True:
                # compute cluster probabilities and apply dirichlet distribution prior:
                self.cluster_probabilities = (t.sum(responsibilities, dim=1) / len(self.cluster_centers))[:,None] # K x len(cluster_concentrations)
                self.cluster_probabilities = self.cluster_probabilities * prior_cluster_probabilities # reapply the prior
                self.cluster_probabilities = t.log(self.cluster_probabilities) # convert to log space to avoid overflow
                self.cluster_probabilities = self.cluster_probabilities/cluster_concentration # K x len(cluster_concentrations)
                self.cluster_probabilities = t.exp(self.cluster_probabilities + 30 - t.max(self.cluster_probabilities))
                self.cluster_probabilities = self.cluster_probabilities/t.sum(self.cluster_probabilities, dim=0)
                
                if print_status:
                    print("CLUSTER PROBS:")
                    print(self.cluster_probabilities)
                    print("Max cluster-prob: " + str(t.max(self.cluster_probabilities)))
                    print("Min cluster-prob: " + str(t.min(self.cluster_probabilities)))

                # maximization step: compute M.L.E. covariance matrices
                converged_clusters = self._mle_bandwidths(responsibilities, minimum_directional_covariance, minimum_total_log_covariance, converged_clusters)
                
                if print_status:
                    if self.covariance_matrix_type == "full": print("New covariance matrix log-determinant:" + str(t.slogdet(self.bandwidths)[1]))
                    elif self.covariance_matrix_type =="diagonal": print("New covariance matrix log-determinant:" + str(t.sum(t.log(self.bandwidths),dim=1)))
                    else: print("New covariance matrix log-determinant:" + str(len(self.cluster_centers)*t.log(self.bandwidths)))
                
                # repeat expectation step
                responsibilities, L_comp_new = self._expectation(print_status)
                
                if print_status: print("New comp-log-likelihood: " + str(L_comp_new))
                if print_status: print(str(t.sum(converged_clusters.float()).item()) + " cluster bandwidths have converged.")
                count += 1

                # check whether convergence conditions have been met
                if t.isinf(L_comp_new) or t.isnan(L_comp_new) or L_comp_new-L_comp_old < convergence_change:
                    convergence_count += 1
                    if convergence_count >= convergence_flexibility:
                        if print_status: print("EM algorithm has converged after " + str(count) + " iterations. Log-likelihood = " + str(L_comp_new.item()))
                        break
                elif t.all(converged_clusters):
                    if print_status: print("All cluster bandwidths converged to minimum value. EM algorithm has converged after " + str(count) + " iterations. Log-likelihood = " + str(L_comp_new.item()))
                    break
                elif count >= max_iterations:
                    if print_status: print("EM algorithm ran for maximum allowed number of iterations: " + str(count) + ". Log-likelihood = " + str(L_comp_new.item()))
                    break
                else: convergence_count=0
                L_comp_old = L_comp_new
            
            self.fitted=True
            return L_comp_new.item()
        except MemoryError: raise MemoryError(f"There is not enough space on device {self.device} to fit a mixture model with the dataset provided.")
        except OverflowError: 
            traceback.print_exc()
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
        cluster_centers = deepcopy(self.cluster_centers).detach().cpu().numpy()
        cluster_center_probs = deepcopy(t.flatten(self.cluster_probabilities)).detach().cpu().numpy()
        cluster_bandwidths = deepcopy(self.bandwidths).detach().cpu().numpy()
        
        # compute conditional mean and variance for the output feature given the sample provided
        if self.limited_memory:
            conditional_means = np.zeros(size=(1,len(cluster_centers)))
            for n in range(len(self.cluster_centers)):
                conditional_means[n] = cluster_centers[n][output_feature_index] + np.dot(np.dot(cluster_bandwidths[output_feature_index,:output_feature_index:], np.linalg.inv(cluster_bandwidths[:output_feature_index:,:output_feature_index:])), sample-cluster_centers[n][:output_feature_index:])
            if self.consistent_variance: conditional_variances = cluster_bandwidths[output_feature_index,output_feature_index] - np.dot(np.dot(cluster_bandwidths[output_feature_index,:output_feature_index:], np.linalg.inv(cluster_bandwidths[:output_feature_index:,:output_feature_index:])), cluster_bandwidths[output_feature_index,:output_feature_index:])
            else: 
                conditional_variances = np.zeros(size=(1,len(cluster_centers)))
                for n in range(len(cluster_centers)):
                    conditional_variances[n] = cluster_bandwidths[n][output_feature_index,output_feature_index] - np.dot(np.dot(cluster_bandwidths[n][output_feature_index,:output_feature_index:], np.linalg.inv(cluster_bandwidths[n][:output_feature_index:,:output_feature_index:])), cluster_bandwidths[n][output_feature_index,:output_feature_index:])
        else:
            pass # haven't done yet
        
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

        features (list[int]): The features of which the provided samples are comprised.
        
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
        conditional_cluster_probs = [] # will be num_clusters x num_samples
        if self.consistent_variance:
            if self.covariance_matrix_type == "full":
                
                # pull out the portion of the covariance matrix corresponding to the provided features
                covariance = self.bandwidths[features,:][:,features]

                # for each cluster center, compute conditional probabilities of all of the samples provided given that cluster
                for n in range(len(self.cluster_centers)):
                    diffs = samples - self.cluster_centers[n,features] # num_samples x num_features
                    probs = -(1/2) * t.slogdet(covariance)[1] + (-1/2) * t.sum(t.abs(t.matmul(diffs, t.linalg.inv(covariance)) * diffs),dim=1)
                    conditional_cluster_probs.append(probs)
                
                # weight conditional sample probs by prior cluster probs to form conditional cluster probs
                conditional_cluster_probs = t.stack(conditional_cluster_probs) # num_clusters x num_samples
                conditional_cluster_probs = self.cluster_probabilities * t.exp(conditional_cluster_probs + 30 - t.max(conditional_cluster_probs, dim=0)[0]) # 1 x num_samples
                conditional_cluster_probs = conditional_cluster_probs/t.sum(conditional_cluster_probs,dim=0)
                
                # sum posterior probabilities over the clusters
                for n in range(len(self.cluster_centers)):
                    outputs += conditional_cluster_probs[n,:][:,None] * (self.cluster_centers[n,conditional_features][None,:] + t.matmul((samples-self.cluster_centers[n,features]) / t.diag(self.bandwidths)[features], self.bandwidths[features,:][:,conditional_features])) # num_samples x num_conditional_features

            if self.covariance_matrix_type == "diagonal":
                covariance = self.bandwidths[features]
                for n in range(len(self.cluster_centers)): 
                    diffs = samples - self.cluster_centers[n,features]
                    probs = -(1/2) * t.sum(t.log(covariance)) + (-1/2) * t.sum(t.abs(diffs**2 / covariance),dim=1)
                    conditional_cluster_probs.append(probs)
                conditional_cluster_probs = t.stack(conditional_cluster_probs) # num_clusters x num_samples
                conditional_cluster_probs = self.cluster_probabilities * t.exp(conditional_cluster_probs + 30 - t.max(conditional_cluster_probs, dim=0)[0]) # 1 x num_samples
                conditional_cluster_probs = conditional_cluster_probs/t.sum(conditional_cluster_probs,dim=0)
                for n in range(len(self.cluster_centers)):
                    outputs += conditional_cluster_probs[n,:][:,None] * (self.cluster_centers[n,conditional_features][None,:]  + (samples-self.cluster_centers[n,features]) / self.bandwidths[features] * self.bandwidths[features,:][:,conditional_features])
                    
            if self.covariance_matrix_type == "scalar":
                for n in range(len(self.cluster_centers)):
                    diffs = samples - self.cluster_centers[n, features]
                    probs = -(len(self.cluster_centers.shape[1])/2) * t.log(self.bandwidths) + (-1/2) * t.sum(t.abs(diffs**2 / self.bandwidths),dim=1)
                    conditional_cluster_probs.append(probs)
                conditional_cluster_probs = t.stack(conditional_cluster_probs) # num_clusters x num_samples
                conditional_cluster_probs = self.cluster_probabilities * t.exp(conditional_cluster_probs + 30 - t.max(conditional_cluster_probs, dim=0)[0]) # 1 x num_samples
                conditional_cluster_probs = conditional_cluster_probs/t.sum(conditional_cluster_probs,dim=0)
                for n in range(len(self.cluster_centers)):
                    outputs += conditional_cluster_probs[n,:][:,None] * self.cluster_centers[n,conditional_features][None,:]  + (samples-self.cluster_centers[n,features])
        else:
            if self.covariance_matrix_type == "full":
                conditional_cluster_probs = []
                for n in range(len(self.cluster_centers)):
                    diffs = samples - self.cluster_centers[n,features]
                    probs = -(1/2) * t.slogdet(self.bandwidths[n][features,:][:,features])[1] + (-1/2) * t.sum(t.abs(t.matmul(diffs, t.linalg.inv(self.bandwidths[n][features,:][:,features])) * diffs),dim=1)
                    conditional_cluster_probs.append(probs)
                conditional_cluster_probs = t.stack(conditional_cluster_probs) # num_clusters x num_samples
                conditional_cluster_probs = self.cluster_probabilities * t.exp(conditional_cluster_probs + 30 - t.max(conditional_cluster_probs, dim=0)[0]) # 1 x num_samples
                conditional_cluster_probs = conditional_cluster_probs/t.sum(conditional_cluster_probs,dim=0)
                for n in range(len(self.cluster_centers)):
                    outputs += conditional_cluster_probs[:,n][:,None] * (self.cluster_centers[n,conditional_features][None,:]  + t.matmul((samples-self.cluster_centers[n,features]) / t.diag(self.bandwidths[n])[features], self.bandwidths[n][features,:][:,conditional_features])) # N x num_given_features
            if self.covariance_matrix_type == "diagonal":
                for n in range(len(self.cluster_centers)):
                    diffs = samples - self.cluster_centers[n,features]
                    probs = -(1/2) * t.sum(t.log(self.bandwidths[n][features])) + (-1/2) * t.sum(t.abs(diffs**2 / self.bandwidths[n][features]),dim=1)
                    conditional_cluster_probs.append(probs)
                conditional_cluster_probs = t.stack(conditional_cluster_probs) # num_clusters x num_samples
                conditional_cluster_probs = self.cluster_probabilities * t.exp(conditional_cluster_probs + 30 - t.max(conditional_cluster_probs, dim=0)[0]) # 1 x num_samples
                conditional_cluster_probs = conditional_cluster_probs/t.sum(conditional_cluster_probs,dim=0)
                for n in range(len(self.cluster_centers)):
                    outputs += conditional_cluster_probs[n,:][:,None] * self.cluster_centers[n,conditional_features][None,:]  + (samples-self.cluster_centers[n,features]) / self.bandwidths[n][features] * self.bandwidths[n][features,:][:,conditional_features]
            if self.covariance_matrix_type == "scalar":
                for n in range(len(self.cluster_centers)):
                    diffs = samples - self.cluster_centers[n, features]
                    probs = -(len(self.cluster_centers.shape[1])/2) * t.log(self.bandwidths[n]) + (-1/2) * t.sum(t.abs(diffs**2 / self.bandwidths[n]),dim=1)
                    conditional_cluster_probs.append(probs)
                conditional_cluster_probs = t.stack(conditional_cluster_probs) # num_clusters x num_samples
                conditional_cluster_probs = self.cluster_probabilities * t.exp(conditional_cluster_probs + 30 - t.max(conditional_cluster_probs, dim=0)[0]) # 1 x num_samples
                conditional_cluster_probs = conditional_cluster_probs/t.sum(conditional_cluster_probs,dim=0)
                for n in range(len(self.cluster_centers)):
                    outputs += conditional_cluster_probs[n,:][:,None] * self.cluster_centers[n,conditional_features][None,:]  + (samples-self.cluster_centers[n,features]) / self.bandwidths[n]**2
        return outputs
    # end mean()
    
    """
    This method reduces the distribution to the desired number of clusters. These new clusters are centered at locations that are most likely to generate the distribution's cluster 
    centers (locations at which a single cluster can best represent the entire distribution). Each new cluster will be orthogonal to the others with respect to the likelihoods it 
    produces (the likelihoods generated by each cluster will be orthogonal). This allows the user to explore how many clusters are necessary to accurately represent the distribution 
    (dimensionality reduction).
    """
    def cluster(self, num_clusters, samples:t.Tensor = None):
        # produce matrix of likelihoods: i,j = P(sample j | cluster i)
        log_probs = self._log_likelihoods(samples=samples)
        
        # convert log-likelihoods to probabilities. We let the likelihoods sum to one to avoid underflow
        log_probs += 30 - t.max(log_probs)
        probs = t.exp(log_probs)
        probs = self.cluster_probabilities * probs
        likelihoods = probs/t.sum(probs)
        print("HERE ARE SOME LIKELIHOODS WHILE CLUSTERING:")
        print(likelihoods)
        print(t.max(likelihoods))
        print(t.min(likelihoods))
        
        # subtract the mean
        probs -= t.mean(probs,dim=1)[:,None]
        
        # compute principle components of the posterior likelihoods
        _,_,vt = t.linalg.svd(probs.T)
        principle_components = vt[:num_clusters,:]/t.sum(vt[:num_clusters,:],dim=1) # num_new_clusters x num_samples
        
        # create new cluster_centers and covariance matrices using the principle components as priors
        new_cluster_centers = t.matmul(principle_components, self.cluster_centers) # num_new_clusters x num_features
        diffs = samples[None,:,:] - new_cluster_centers[:,None,:] # num_new_clusters x num_samples x num_features
        new_cluster_bandwidths = t.matmul(t.permute(diffs,(0,2,1)), diffs*principle_components[:,:,None]) # num_new_clusters x num_features x num_features
        
        # compute posterior cluster probabilities using new distribution
        log_probs_front = (-1/2) * t.slogdet(new_cluster_bandwidths)[1][:,None] # num_new_clusters x 1
        log_probs = log_probs_front + (-1/2) * t.einsum('ijk, ijk -> ij', t.matmul(diffs, t.linalg.inv(new_cluster_bandwidths)), diffs) # num_new_clusters x num_samples
        log_probs += 30 - t.max(log_probs,dim=0)[0]
        probs = t.exp(log_probs) # i,j = P(sample j | cluster i)
        probs = probs/t.sum(probs,dim=0)
        new_cluster_probabilities = t.sum(probs, dim=1) / probs.shape[1]
        
        # create new gaussian mixture model using the new clusters
        new_distribution = Gaussian_Mixture_Model(consistent_variance = False, covariance_matrix_type = "full", dtype = t.float64)
        new_distribution.cluster_centers = new_cluster_centers
        new_distribution.bandwidths = new_cluster_bandwidths
        new_distribution.cluster_probabilities = new_cluster_probabilities
        new_distribution.device = self.device
        new_distribution.fitted = True
        
        return new_distribution
        
    
    '''
    This method carries out the expectation step of the EM algorithm, using the current cluster probability values and cluster 
        covariances to compute responsibilities (posterior probabilities of cluster centers given samples).
    '''
    def _expectation(self, print_status:bool, attempt_gpu:bool):
        log_probs = self._log_likelihoods(attempt_gpu)
            
        # convert log-likelihoods to linear space
        probs = log_probs - t.max(log_probs, dim=0)[0] # ensure that all log_probs are <= zero
        probs = probs / self.responsibility_concentration # bring probs closer together or push them further apart depending on self.responsibility_concentration
        probs = t.exp(probs + 30 - t.max(probs,dim=0)[0])
        probs = probs/t.sum(probs,dim=0)
        

        if print_status:
            print("conditional probabilities:")
            print(probs)
            print("Min prob: " + str(t.min(probs)))
            print("Max prob: " + str(t.max(probs)))
        
        # compute posterior probabilities, P(cluster_i | sample_j)
        responsibilities = self.cluster_probabilities*probs

        # set the responsibilities for different classes to zero
        if self.num_classes != 1:
            for label in t.unique(self.class_labels):
                label_indices = t.flatten((self.class_labels == label)) # (num_clusters, 1) with True where the j-th sample comes from the loop's current label
                opposing_label_indices = ~label_indices
                keepers = label_indices[:,None] * opposing_label_indices[None,:]
                responsibilities -= responsibilities*keepers

        responsibilities = responsibilities / t.sum(responsibilities, dim = 0)

        if print_status:
            print("Responsibilities:")
            print(responsibilities)
            print("Min responsibility: " + str(t.min(responsibilities)))
            print("Max responsibility: " + str(t.max(responsibilities)))
        
        L_comp = t.einsum("ij, ij -> ", responsibilities, log_probs)
        return responsibilities, L_comp # returns the responsibilities and the complete log likelihood
    # end _expectation()



###########################################################################################################################






    """
    Computes the maximum likelihood covariance matrices using the object's cluster centers and some passed responsibility values
    """
    def _mle_bandwidths(self, attempt_gpu:bool, responsibilities:t.Tensor, minimum_directional_covariance:float, minimum_total_log_covariance:float, converged_clusters:list[bool]):
        if attempt_gpu and t.cuda.is_available(): device = "cuda" # local reference
        else: device = "cpu"
        if self.device == "cuda":
            pynvml.nvmlInit()
            floats_available = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 2/ float(str(self.dtype)[-2:]) / 8)
        else:
            floats_available = int(psutil.virtual_memory()[4] / 2 / float(str(self.dtype)[-2:]) / 8)        

        # full covariance matrices:
        if self.covariance_matrix_type == "full":
            if self.consistent_variance: new_bandwidth = t.zeros(size=self.bandwidths.size(), device = "cpu") # placeholder
            # determine batch sizes
            feature_batch_size1 = int((floats_available-len(self.cluster_centers))/(2*len(self.cluster_centers)+1))
            if feature_batch_size1 > self.cluster_centers.shape[1]: feature_batch_size2 = int(feature_batch_size1/self.cluster_centers.shape[1])
            else: feature_batch_size2 = 1
            if feature_batch_size2 > self.cluster_centers.shape[1]: cluster_batch_size = int(feature_batch_size2/self.cluster_centers.shape[1])
            else: cluster_batch_size = 1
            if self.consistent_variance:
                # begin looping through clusters, features on the left, and features on the right
                for n in range(len(self.cluster_centers)/cluster_batch_size):
                    for m in range(self.cluster_centers.shape[1]/feature_batch_size2):
                        for l in range(self.cluster_centers.shape[1]/feature_batch_size1): 
                            diffs_left = self.cluster_centers[None,:,l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) # 1 x num_clusters x features_batch_size1
                            diffs_right = self.cluster_centers[None,:,m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) # 1 x num_clusters x feature_batch_size2
                            new_bandwidth[l*feature_batch_size1:(l+1)*feature_batch_size1, m*feature_batch_size2:(m+1)*feature_batch_size2] = t.sum(t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None]) * self.cluster_probabilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None], dim=0).cpu() # 1 x features_batch_size1 x features_batch_size2
                        # last iteration of l-loop
                        diffs_left = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) # 1 x num_clusters x features_batch_size1
                        diffs_right = self.cluster_centers[None,:,m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) # 1 x num_clusters x feature_batch_size2
                        new_bandwidth[int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):, m*feature_batch_size2:(m+1)*feature_batch_size2] = t.sum(t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None]) * self.cluster_probabilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None], dim=0).cpu() # 1 x features_batch_size1 x features_batch_size2
                    # last iteration of m-loop
                    for l in range(self.cluster_centers.shape[1]/feature_batch_size1):
                        diffs_left = self.cluster_centers[None,:,l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) # 1 x num_clusters x features_batch_size1
                        diffs_right = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) # 1 x num_clusters x feature_batch_size2
                        new_bandwidth[l*feature_batch_size1:(l+1)*feature_batch_size1, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):] = t.sum(t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None]) * self.cluster_probabilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None], dim=0).cpu() # 1 x features_batch_size1 x features_batch_size2
                    diffs_left = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) # 1 x num_clusters x features_batch_size1
                    diffs_right = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) # 1 x num_clusters x feature_batch_size2
                    new_bandwidth[int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):] = t.sum(t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None]) * self.cluster_probabilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None], dim=0).cpu() # 1 x features_batch_size1 x features_batch_size2
                # last iteration of n-loop, repeating m-loop and thereby l-loop
                for m in range(self.cluster_centers.shape[1]/feature_batch_size2):
                    for l in range(self.cluster_centers.shape[1]/feature_batch_size1):
                        diffs_left = self.cluster_centers[None,:,l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) # 1 x num_clusters x features_batch_size1
                        diffs_right = self.cluster_centers[None,:,m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) # 1 x num_clusters x feature_batch_size2
                        new_bandwidth[l*feature_batch_size1:(l+1)*feature_batch_size1, m*feature_batch_size2:(m+1)*feature_batch_size2] = t.sum(t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None]) * self.cluster_probabilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None], dim=0).cpu() # 1 x features_batch_size1 x features_batch_size2 
                    diffs_left = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) # 1 x num_clusters x features_batch_size1
                    diffs_right = self.cluster_centers[None,:,m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) # 1 x num_clusters x feature_batch_size2
                    new_bandwidth[int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):, m*feature_batch_size2:(m+1)*feature_batch_size2] = t.sum(t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None]) * self.cluster_probabilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None], dim=0).cpu() # 1 x features_batch_size1 x features_batch_size2
                for l in range(self.cluster_centers.shape[1]/feature_batch_size1):
                    diffs_left = self.cluster_centers[None,:,l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) # 1 x num_clusters x features_batch_size1
                    diffs_right = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) # 1 x num_clusters x feature_batch_size2
                    new_bandwidth[l*feature_batch_size1:(l+1)*feature_batch_size1, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):] = t.sum(t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None]) * self.cluster_probabilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None], dim=0).cpu() # 1 x features_batch_size1 x features_batch_size2
                diffs_left = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) # 1 x num_clusters x features_batch_size1
                diffs_right = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) # 1 x num_clusters x feature_batch_size2
                new_bandwidth[int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):] = t.sum(t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None]) * self.cluster_probabilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None], dim=0).cpu() # 1 x features_batch_size1 x features_batch_size2
                
                # check that new_bandwidth is not too small:
                if device is not "cpu": new_bandwidth = new_bandwidth.to(device=device)
                s,v,d = t.linalg.svd(new_bandwidth)
                v = t.maximum(v, t.tensor(minimum_directional_covariance, device=self.device, dtype=self.dtype))
                new_bandwidth = t.matmul(s*v,d)
                if t.slogdet(new_bandwidth)[1] < minimum_total_log_covariance: # average covariance matrix is too small, need to adjust it
                    converged_clusters[0] = True
                    self.bandwidths = new_bandwidth * t.exp((minimum_total_log_covariance-t.slogdet(new_bandwidth)[1])/self.cluster_centers.shape[1]).cpu()
            
            # repeat above with individual covariance matrices for each cluster. Uses the same code as above but with no placeholder. Computed variance/covariance values go straight into self.bandwidths
            else: 
                for n in range(len(self.cluster_centers)/cluster_batch_size): # loop through the cluster centers in batches
                    for m in range(self.cluster_centers.shape[1]/feature_batch_size2): # loop through cluster features in batches
                        for l in range(self.cluster_centers.shape[1]/feature_batch_size1): # loop through cluster features again in batches
                            diffs_left = self.cluster_centers[None,:,l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) # 1 x num_clusters x features_batch_size1
                            diffs_right = self.cluster_centers[None,:,m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) # 1 x num_clusters x feature_batch_size2
                            self.bandwidths[n*cluster_batch_size:(n+1)*cluster_batch_size,l*feature_batch_size1:(l+1)*feature_batch_size1, m*feature_batch_size2:(m+1)*feature_batch_size2] = t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None]).cpu() # 1 x features_batch_size1 x features_batch_size2
                        # last iteration of l-loop:
                        diffs_left = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) # 1 x num_clusters x features_batch_size1
                        diffs_right = self.cluster_centers[None,:,m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) # 1 x num_clusters x feature_batch_size2
                        self.bandwidths[n*cluster_batch_size:(n+1)*cluster_batch_size, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):, m*feature_batch_size2:(m+1)*feature_batch_size2] = t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None]).cpu() # 1 x features_batch_size1 x features_batch_size2
                    # last iteration of m-loop:
                    for l in range(self.cluster_centers.shape[1]/feature_batch_size1):
                        diffs_left = self.cluster_centers[None,:,l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) # 1 x num_clusters x features_batch_size1
                        diffs_right = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) # 1 x num_clusters x feature_batch_size2
                        self.bandwidths[n*cluster_batch_size:(n+1)*cluster_batch_size, l*feature_batch_size1:(l+1)*feature_batch_size1, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):] = t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None]).cpu() # 1 x features_batch_size1 x features_batch_size2
                    diffs_left = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) # 1 x num_clusters x features_batch_size1
                    diffs_right = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) # 1 x num_clusters x feature_batch_size2
                    self.bandwidths[n*cluster_batch_size:(n+1)*cluster_batch_size,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):] = t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None]).cpu() # 1 x features_batch_size1 x features_batch_size2
                # last iteration of n-loop:
                for m in range(self.cluster_centers.shape[1]/feature_batch_size2):
                    for l in range(self.cluster_centers.shape[1]/feature_batch_size1):
                        diffs_left = self.cluster_centers[None,:,l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) # 1 x num_clusters x features_batch_size1
                        diffs_right = self.cluster_centers[None,:,m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) # 1 x num_clusters x feature_batch_size2
                        self.bandwidths[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, l*feature_batch_size1:(l+1)*feature_batch_size1, m*feature_batch_size2:(m+1)*feature_batch_size2] = t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None]).cpu() # 1 x features_batch_size1 x features_batch_size2
                    diffs_left = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) # 1 x num_clusters x features_batch_size1
                    diffs_right = self.cluster_centers[None,:,m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, m*feature_batch_size2:(m+1)*feature_batch_size2].to(device=device) # 1 x num_clusters x feature_batch_size2
                    self.bandwidths[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):, m*feature_batch_size2:(m+1)*feature_batch_size2] = t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None]).cpu() # 1 x features_batch_size1 x features_batch_size2
                for l in range(self.cluster_centers.shape[1]/feature_batch_size1):
                    diffs_left = self.cluster_centers[None,:,l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, l*feature_batch_size1:(l+1)*feature_batch_size1].to(device=device) # 1 x num_clusters x features_batch_size1
                    diffs_right = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) # 1 x num_clusters x feature_batch_size2
                    self.bandwidths[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, l*feature_batch_size1:(l+1)*feature_batch_size1, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):] = t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None]).cpu() # 1 x features_batch_size1 x features_batch_size2
                diffs_left = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):].to(device=device) # 1 x num_clusters x features_batch_size1
                diffs_right = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):].to(device=device) # 1 x num_clusters x feature_batch_size2
                self.bandwidths[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size1)):, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size2)):] = t.matmul(t.permute(diffs_left,(0,2,1)), diffs_right*responsibilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None]).cpu() # 1 x features_batch_size1 x features_batch_size2
                
                # check that all covariance matrices are not too small:
                gc.collect()
                batch_size = int(floats_available/4*self.cluster_centers.shape[1]**2)
                for n in range(len(self.cluster_centers)/batch_size): # loop through the covariance matrices at each cluster in batches
                    new_bandwidths = self.bandwidths[n*batch_size:(n+1)*batch_size,:,:].to(device=device)
                    s,v,d = t.linalg.svd(new_bandwidths)
                    v = t.maximum(v, t.tensor(minimum_directional_covariance, device=self.device, dtype=self.dtype))
                    new_bandwidths = t.matmul(s*v,d)
                    converged_indices = t.slogdet(new_bandwidths)[1] < minimum_total_log_covariance
                    converged_clusters[n*converged_indices:(n+1)*batch_size] = True
                    new_bandwidths[converged_indices,:,:] = new_bandwidths[converged_indices,:,:] * t.exp((minimum_total_log_covariance-t.slogdet(new_bandwidths[converged_indices,:,:])[1])/self.cluster_centers.shape[1]).cpu()
                    self.bandwidths[n*batch_size:(n+1)*batch_size] = new_bandwidths
                # last iteration of n-loop:
                new_bandwidths = self.bandwidths[int(len(self.cluster_centers)-(len(self.cluster_centers)%batch_size)):,:,:].to(device=device)
                s,v,d = t.linalg.svd(new_bandwidths)
                v = t.maximum(v, t.tensor(minimum_directional_covariance, device=self.device, dtype=self.dtype))
                new_bandwidths = t.matmul(s*v,d)
                converged_indices = t.slogdet(new_bandwidths)[1] < minimum_total_log_covariance
                converged_clusters[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):] = True
                new_bandwidths[converged_indices,:,:] = new_bandwidths[converged_indices,:,:] * t.exp((minimum_total_log_covariance-t.slogdet(new_bandwidths[converged_indices,:,:])[1])/self.cluster_centers.shape[1]).cpu()
                self.bandwidths[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):] = new_bandwidths
        
        # using diagonal or scalar representations of covariance matrices
        else:
            # determine batch sizes
            feature_batch_size = int(floats_available/(2*len(self.cluster_centers)+1))
            if feature_batch_size > self.cluster_centers.shape[1]: cluster_batch_size = int(feature_batch_size/self.cluster_centers.shape[1])
            else: cluster_batch_size = 1
            if self.consistent_variance:
                new_bandwidth = t.zeros(size=(1,self.cluster_centers.shape[1]), device = "cpu") # placeholder
                # loop through clusters, features on the left, and features on the right
                for n in range(int(len(self.cluster_centers)/cluster_batch_size)):
                    for m in range(self.cluster_centers.shape[1]/feature_batch_size):
                        diffs = self.cluster_centers[None,:,m*feature_batch_size:(m+1)*feature_batch_size].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, m*feature_batch_size:(m+1)*feature_batch_size].to(device=device) # cluster_batch_size x num_clusters x feature_batch_size
                        new_bandwidth[m*feature_batch_size:(m+1)*feature_batch_size] = t.einsum("ijk,ijk->ik", diffs, diffs*responsibilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None]) * self.cluster_probabilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:].cpu()
                    # last iteration of the m-loop
                    diffs = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size)):].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size)):].to(device=device) # cluster_batch_size x num_clusters x feature_batch_size
                    new_bandwidth[int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size)):] = t.sum(t.einsum("ijk,ijk->ik", diffs, diffs*responsibilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None]) * self.cluster_probabilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:], dim=0).cpu()
                # last iteration of the n-loop
                for m in range(self.cluster_centers.shape[1]/feature_batch_size):
                    diffs = self.cluster_centers[None,:,m*feature_batch_size:(m+1)*feature_batch_size].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, m*feature_batch_size:(m+1)*feature_batch_size].to(device=device) # cluster_batch_size x num_clusters x feature_batch_size
                    new_bandwidth[m*feature_batch_size:(m+1)*feature_batch_size] = t.einsum("ijk,ijk->ik", diffs, diffs*responsibilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None]) * self.cluster_probabilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:].cpu()
                diffs = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size)):].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size)):].to(device=device) # cluster_batch_size x num_clusters x feature_batch_size
                new_bandwidth[int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size)):] += t.sum(t.einsum("ijk,ijk->ik", diffs, diffs*responsibilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None]) * self.cluster_probabilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:], dim=0).cpu()
                
                # check that new_bandwidth is not too small:
                if self.covariance_matrix_type == "scalar": self.bandwidths = t.exp(t.maximum(t.log(t.mean(new_bandwidths, dim=1)).cpu(), t.tensor(minimum_total_log_covariance, device=device, dtype=self.dtype)))
                else:
                    new_bandwidth = t.maximum(new_bandwidth, t.tensor(minimum_directional_covariance, device="cpu", dtype=self.dtype))
                    dets = t.sum(t.log(new_bandwidth),dim=1)
                    if (dets < minimum_total_log_covariance).any(): 
                        new_bandwidth[dets < minimum_total_log_covariance] = new_bandwidth[dets < minimum_total_log_covariance] * t.exp((minimum_total_log_covariance-dets[dets < minimum_total_log_covariance])/self.cluster_centers.shape[1])
                        converged_clusters[dets < minimum_total_log_covariance] = True
                self.bandwidths = new_bandwidths
            # repeat above but without summing the weighted covariance matrix representations
            
            else: # Here need to fix indexing below, especially when adding to new_bandwidths. Don't need to add, can just replace...
                for n in range(int(len(self.cluster_centers)/cluster_batch_size)):
                    new_bandwidths = t.zeros(size = (cluster_batch_size, self.cluster_centers.shape[1]), device="cpu", dtype = self.dtype)
                    for m in range(self.cluster_centers.shape[1]/feature_batch_size):
                        diffs = self.cluster_centers[None,:,m*feature_batch_size:(m+1)*feature_batch_size].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, m*feature_batch_size:(m+1)*feature_batch_size].to(device=device) # cluster_batch_size x num_clusters x feature_batch_size
                        new_bandwidths[:,n*cluster_batch_size:(n+1)*cluster_batch_size] = t.einsum("ijk,ijk->ik", diffs, diffs*responsibilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None]).cpu()
                    diffs = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size)):].to(device=device) - self.cluster_centers[n*cluster_batch_size:(n+1)*cluster_batch_size, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size)):].to(device=device) # cluster_batch_size x num_clusters x feature_batch_size
                    new_bandwidths[:,n*cluster_batch_size:(n+1)*cluster_batch_size] = t.einsum("ijk,ijk->ik", diffs, diffs*responsibilities[n*cluster_batch_size:(n+1)*cluster_batch_size,:,None]).cpu()
                    # make sure that covariance matrices are not too small
                    if self.covariance_matrix_type == "scalar": self.bandwidths[n*cluster_batch_size:(n+1)*cluster_batch_size] = t.exp(t.maximum(t.log(t.mean(new_bandwidths, dim=1)).cpu(), t.tensor(minimum_total_log_covariance, device=device, dtype=self.dtype)))
                    else:
                        new_bandwidths = t.maximum(new_bandwidths, t.tensor(minimum_directional_covariance, device=device, dtype=self.dtype))
                        dets = t.sum(t.log(new_bandwidths),dim=1)
                        if (dets < minimum_total_log_covariance).any(): 
                            new_bandwidths[dets < minimum_total_log_covariance] = new_bandwidths[dets < minimum_total_log_covariance] * t.exp((minimum_total_log_covariance-dets[dets < minimum_total_log_covariance])/self.cluster_centers.shape[1])
                            converged_clusters[dets < minimum_total_log_covariance] = True
                    self.bandwidths[n*cluster_batch_size:(n+1)*cluster_batch_size] = new_bandwidths
                # last iteration of n-loop:
                new_bandwidths = t.zeros(size = (int(len(self.cluster_centers)%cluster_batch_size), self.cluster_centers.shape[1]), device="cpu", dtype = self.dtype)
                for m in range(self.cluster_centers.shape[1]/feature_batch_size):
                    diffs = self.cluster_centers[None,:,m*feature_batch_size:(m+1)*feature_batch_size].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, m*feature_batch_size:(m+1)*feature_batch_size].to(device=device) # cluster_batch_size x num_clusters x feature_batch_size
                    new_bandwidths[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, m*feature_batch_size:(m+1)*feature_batch_size] = t.einsum("ijk,ijk->ik", diffs, diffs*responsibilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None]).cpu()
                diffs = self.cluster_centers[None,:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size)):].to(device=device) - self.cluster_centers[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):, None, int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size)):].to(device=device) # cluster_batch_size x num_clusters x feature_batch_size
                new_bandwidths[:,int(self.cluster_centers.shape[1]-(self.cluster_centers.shape[1]%feature_batch_size)):] = t.einsum("ijk,ijk->ik", diffs, diffs*responsibilities[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):,:,None]).cpu()
                # make sure that covariance matrices are not too small
                if self.covariance_matrix_type == "scalar": self.bandwidths[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):] = t.exp(t.maximum(t.log(t.mean(new_bandwidths, dim=1)).cpu(), t.tensor(minimum_total_log_covariance, device=device, dtype=self.dtype)))
                else:
                    new_bandwidths = t.maximum(new_bandwidths, t.tensor(minimum_directional_covariance, device=device, dtype=self.dtype))
                    dets = t.sum(t.log(new_bandwidths),dim=1)
                    if (dets < minimum_total_log_covariance).any(): 
                        new_bandwidths[dets < minimum_total_log_covariance] = new_bandwidths[dets < minimum_total_log_covariance] * t.exp((minimum_total_log_covariance-dets[dets < minimum_total_log_covariance])/self.cluster_centers.shape[1])
                        converged_clusters[dets < minimum_total_log_covariance] = True
                    self.bandwidths[int(len(self.cluster_centers)-(len(self.cluster_centers)%cluster_batch_size)):] = new_bandwidths
        return converged_clusters
        # end maximization step
        
        
        
    
    """
    Computes the log-likelihoods of a sample of points being generated by the current distribution (cluster centers and bandwidths). 
        These computations are done frequently enough to warrant separating them into their own method.
    """
    def _log_likelihoods(self, attempt_gpu:bool, samples:t.Tensor = None):
        if samples is None: samples = self.cluster_centers
        if attempt_gpu and t.cuda.is_available(): device = "cuda" # local reference
        else: device = "cpu"

        # if the inverse covariance matrix is consistent for all clusters, put it onto the device being used for computations now
        if self.consistent_variance and self.covariance_matrix_type == "full": inv = t.linalg.inv(self.bandwidths).to(device = device)
        
        # compute batch size
        if device == "cuda":
            pynvml.nvmlInit()
            floats_available = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 2 / float(str(self.dtype)[-2:]) / 8)
        else:
            floats_available = int(psutil.virtual_memory()[4] / 2 / float(str(self.dtype)[-2:]) / 8)
            
        batch_size = int((floats_available - 2*len(self.cluster_centers)**2 - 2*len(self.cluster_centers))/(2*len(self.cluster_centers)*self.cluster_centers.shape[1])) # need to hold responsibility values before assigning, log_probs_front, and as many triple sets of one cluster center minus all the other cluster centers as possible
        if batch_size < 1: 
            batch_size = 1 # shouldn't be enough space, try anyway

        # compute normalization piece, using batches of covariance matrices if necessary
        if self.consistent_variance: # do not need to use batches
            if self.covariance_matrix_type == 'full': log_likelihoods_front = (-1/2)*t.slogdet(self.bandwidths.to(device=device))[1].item() # scalar
            elif self.covariance_matrix_type == "diagonal": log_likelihoods_front = (-1/2) * t.sum(t.log(self.bandwidths), dim=1).item()
            else: log_likelihoods_front = (-self.cluster_centers.shape[1]/2) * t.log(self.bandwidths).item()
        elif self.covariance_matrix_type == "diagonal": log_likelihoods_front = (-1/2) * t.sum(t.log(self.bandwidths), dim=1)[:,None]
        elif self.covariance_matrix_type == "scalar": log_likelihoods_front = (-self.cluster_centers.shape[1]/2) * t.log(self.bandwidths)
        
        else: # do need to use batches
            log_likelihoods_front = t.zeros(size=(len(self.cluster_centers),1), device="cpu", dtype=self.dtype)
            if device == "cuda":
                pynvml.nvmlInit()
                floats_available = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 2 / float(str(self.dtype)[-2:]) / 8)
            else:
                floats_available = int(psutil.virtual_memory()[4] / 2 / float(str(self.dtype)[-2:]) / 8)
            batch_size = int(floats_available/(2*self.cluster_centers.shape[1]**2 + 1))
            if self.covariance_matrix_type == "full":
                for n in range(int(len(self.cluster_centers)/batch_size)):
                    log_likelihoods_front[int(n*batch_size):int((n+1)*batch_size)] = (-1/2) * t.slogdet(self.bandwidths[int(n*batch_size):int((n+1)*batch_size)].to(device = device))[1][:,None].cpu()
                log_likelihoods_front[int(len(self.cluster_centers)-((len(self.cluster_centers)%batch_size))):] = (-1/2) * t.slogdet(self.bandwidths[int(len(self.cluster_centers)-((len(self.cluster_centers)%batch_size))):].to(device = device))[1][:,None].cpu()

        # compute log-likelihood piece. Need to use batches for every case because of the size of "diffs"
        log_likelihoods = t.ones(size=(self.cluster_centers.shape[0],samples.shape[0]), device = "cpu", dtype=self.dtype) # instantiate holder for the log-likelihood computations on cpu
        # the same covariance matrix is used for all clusters. The syntax is slightly different this way. 
        if self.consistent_covariance:
            if self.covariance_matrix_type == "full": # using full covariance matrices
                inv = t.linalg.inv(self.bandwidths.to(device = device)) # we can keep the same inverse covariance on the device doing the computations the whole time
                # compute batch size 
                if device == "cuda":
                    pynvml.nvmlInit()
                    floats_available = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 2 / float(str(self.dtype)[-2:]) / 8)
                else: floats_available = int(psutil.virtual_memory()[4] / 2 / float(str(self.dtype)[-2:]) / 8)
                sample_batch_size = int(floats_available/(4*self.cluster_centers.shape[1]))
                if sample_batch_size > len(samples): cluster_batch_size = int(sample_batch_size/len(samples))
                else: cluster_batch_size = 1
                # perform batch computations using as many samples as possible, down to at least one sample.
                for n in range(int(len(self.cluster_centers)/cluster_batch_size)):
                    for m in range(int(len(samples)/sample_batch_size)):
                        diffs = samples[None,int(m*sample_batch_size):int((m+1)*sample_batch_size),:].to(device=device)-self.cluster_centers[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),None,:].to(device=device) # cluster_batch_size x samples_batch_size x num_features
                        log_likelihoods[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),int(m*sample_batch_size):int((m+1)*sample_batch_size)] = (-1/2) * t.sum(t.abs(t.matmul(diffs, inv) * diffs),dim=2).cpu() # cluster_batch_size x sample_batch_size
                    diffs = samples[None,int(len(samples)-((len(samples)%sample_batch_size))):,:].to(device=device)-self.cluster_centers[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),None,:].to(device=device) # batch_size x num_clusters x num_features
                    log_likelihoods[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),int(len(samples)-((len(samples)%sample_batch_size))):] = (-1/2) * t.sum(t.abs(t.matmul(diffs, inv) * diffs),dim=2).cpu()
                for m in range(int(len(samples)/sample_batch_size)):
                    diffs = samples[None,int(m*sample_batch_size):int((m+1)*sample_batch_size),:].to(device=device)-self.cluster_centers[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,None,:].to(device=device) # cluster_batch_size x samples_batch_size x num_features
                    log_likelihoods[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),int(m*sample_batch_size):int((m+1)*sample_batch_size)] = (-1/2) * t.sum(t.abs(t.matmul(diffs, inv) * diffs),dim=2).cpu() # cluster_batch_size x sample_batch_size
                diffs = samples[None,int(len(samples)-((len(samples)%sample_batch_size))):,:].to(device=device)-self.cluster_centers[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,None,:].to(device=device) # batch_size x num_clusters x num_features
                log_likelihoods[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,int(len(samples)-((len(samples)%sample_batch_size))):] = (-1/2) * t.sum(t.abs(t.matmul(diffs, inv) * diffs),dim=2).cpu()
            
            else: # using diagonal or scalar representations of the inver covariance matrix
                cov = self.bandwidths.to(device=device) # can still keep the inverse on the device doing computations the whole time
                # compute batch size 
                if device == "cuda":
                    pynvml.nvmlInit()
                    floats_available = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 2 / float(str(self.dtype)[-2:]) / 8)
                else: floats_available = int(psutil.virtual_memory()[4] / 2 / float(str(self.dtype)[-2:]) / 8)
                sample_batch_size = int(floats_available/(4*self.cluster_centers.shape[1]))
                if sample_batch_size > len(samples): cluster_batch_size = int(sample_batch_size/len(samples))
                else: cluster_batch_size = 1
                # perform batch computations using as many samples as possible, down to at least one sample.
                for n in range(int(len(self.cluster_centers)/cluster_batch_size)):
                    for m in range(int(len(samples)/sample_batch_size)):
                        diffs = samples[None,int(m*sample_batch_size):int((m+1)*sample_batch_size),:].to(device=device)-self.cluster_centers[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),None,:].to(device=device) # cluster_batch_size x samples_batch_size x num_features
                        log_likelihoods[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),int(m*sample_batch_size):int((m+1)*sample_batch_size)] = (-1/2) * t.einsum("ijk,ijk,ij", diffs/cov, diffs).cpu() # cluster_batch_size x sample_batch_size
                    diffs = samples[None,int(len(samples)-((len(samples)%sample_batch_size))):,:].to(device=device)-self.cluster_centers[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),None,:].to(device=device) # batch_size x num_clusters x num_features
                    log_likelihoods[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),int(len(samples)-((len(samples)%sample_batch_size))):] = (-1/2) * t.einsum("ijk,ijk,ij", diffs/cov, diffs).cpu()
                for m in range(int(len(samples)/sample_batch_size)):
                    diffs = samples[None,int(m*sample_batch_size):int((m+1)*sample_batch_size),:].to(device=device)-self.cluster_centers[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,None,:].to(device=device) # cluster_batch_size x samples_batch_size x num_features
                    log_likelihoods[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),int(m*sample_batch_size):int((m+1)*sample_batch_size)] = (-1/2) * t.einsum("ijk,ijk,ij", diffs/cov, diffs).cpu() # cluster_batch_size x sample_batch_size
                diffs = samples[None,int(len(samples)-((len(samples)%sample_batch_size))):,:].to(device=device)-self.cluster_centers[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,None,:].to(device=device) # batch_size x num_clusters x num_features
                log_likelihoods[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,int(len(samples)-((len(samples)%sample_batch_size))):] = (-1/2) * t.einsum("ijk,ijk,ij", diffs/cov, diffs).cpu()

        else: # using a different covariance matrix for each cluster center
            if self.covariance_matrix_type == "full":
                # compute batch size 
                if device == "cuda":
                    pynvml.nvmlInit()
                    floats_available = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 2 / float(str(self.dtype)[-2:]) / 8)
                else: floats_available = int(psutil.virtual_memory()[4] / 2 / float(str(self.dtype)[-2:]) / 8)
                sample_batch_size = int((floats_available-3*self.cluster_centers.shape[1]**2)/(4*self.cluster_centers.shape[1]))
                if sample_batch_size > len(samples): cluster_batch_size = int(sample_batch_size/len(samples))
                else: cluster_batch_size = 1
                # perform batch computations using as many samples as possible, down to at least one sample.
                for n in range(int(len(self.cluster_centers)/cluster_batch_size)):
                    for m in range(int(len(samples)/sample_batch_size)):
                        diffs = samples[None,int(m*sample_batch_size):int((m+1)*sample_batch_size),:].to(device=device)-self.cluster_centers[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),None,:].to(device=device) # cluster_batch_size x samples_batch_size x num_features
                        log_likelihoods[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),int(m*sample_batch_size):int((m+1)*sample_batch_size)] = (-1/2) * t.sum(t.abs(t.matmul(diffs, t.linalg.inv(self.bandwidths[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),:,:].to(device=device))) * diffs),dim=2).cpu() # cluster_batch_size x sample_batch_size
                    diffs = samples[None,int(len(samples)-((len(samples)%sample_batch_size))):,:].to(device=device)-self.cluster_centers[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),None,:].to(device=device) # batch_size x num_clusters x num_features
                    log_likelihoods[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),int(len(samples)-((len(samples)%sample_batch_size))):] = (-1/2) * t.sum(t.abs(t.matmul(diffs, t.linalg.inv(self.bandwidths[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),:,:].to(device=device))) * diffs),dim=2).cpu()
                for m in range(int(len(samples)/sample_batch_size)):
                    diffs = samples[None,int(m*sample_batch_size):int((m+1)*sample_batch_size),:].to(device=device)-self.cluster_centers[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,None,:].to(device=device) # cluster_batch_size x samples_batch_size x num_features
                    log_likelihoods[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),int(m*sample_batch_size):int((m+1)*sample_batch_size)] = (-1/2) * t.sum(t.abs(t.matmul(diffs, t.linalg.inv(self.bandwidths[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,:,:].to(device=device))) * diffs),dim=2).cpu() # cluster_batch_size x sample_batch_size
                diffs = samples[None,int(len(samples)-((len(samples)%sample_batch_size))):,:].to(device=device)-self.cluster_centers[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,None,:].to(device=device) # batch_size x num_clusters x num_features
                log_likelihoods[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,int(len(samples)-((len(samples)%sample_batch_size))):] = (-1/2) * t.sum(t.abs(t.matmul(diffs, t.linalg.inv(self.bandwidths[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,:,:].to(device=device))) * diffs),dim=2).cpu()
            else: # covariance matrices have diagonal or scalar representations
                # same as above but slighlty different handling of covariance matrices
                # compute batch size 
                if device == "cuda":
                    pynvml.nvmlInit()
                    floats_available = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 2 / float(str(self.dtype)[-2:]) / 8)
                else: floats_available = int(psutil.virtual_memory()[4] / 2 / float(str(self.dtype)[-2:]) / 8)
                sample_batch_size = int((floats_available-3*self.cluster_centers.shape[1])/(4*self.cluster_centers.shape[1]))
                if sample_batch_size > len(samples): cluster_batch_size = int(sample_batch_size/len(samples))
                else: cluster_batch_size = 1
                # perform batch computations using as many samples as possible, down to at least one sample.
                for n in range(int(len(self.cluster_centers)/cluster_batch_size)):
                    for m in range(int(len(samples)/sample_batch_size)):
                        diffs = samples[None,int(m*sample_batch_size):int((m+1)*sample_batch_size),:].to(device=device)-self.cluster_centers[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),None,:].to(device=device) # cluster_batch_size x samples_batch_size x num_features
                        log_likelihoods[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),int(m*sample_batch_size):int((m+1)*sample_batch_size)] = (-1/2) * t.einsum("ijk,ijk,ij", diffs/self.bandwidths[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),:,:].to(device=device), diffs).cpu() # cluster_batch_size x sample_batch_size
                    diffs = samples[None,int(len(samples)-((len(samples)%sample_batch_size))):,:].to(device=device)-self.cluster_centers[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),None,:].to(device=device) # batch_size x num_clusters x num_features 
                    log_likelihoods[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),int(len(samples)-((len(samples)%sample_batch_size))):] = (-1/2) * t.einsum("ijk,ijk,ij", diffs/self.bandwidths[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),:,:].to(device=device), diffs).cpu()
                for m in range(int(len(samples)/sample_batch_size)):
                    diffs = samples[None,int(m*sample_batch_size):int((m+1)*sample_batch_size),:].to(device=device)-self.cluster_centers[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,None,:].to(device=device) # cluster_batch_size x samples_batch_size x num_features
                    log_likelihoods[int(n*cluster_batch_size):int((n+1)*cluster_batch_size),int(m*sample_batch_size):int((m+1)*sample_batch_size)] = (-1/2) * t.einsum("ijk,ijk,ij", diffs/self.bandwidths[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,:,:].to(device=device), diffs).cpu() # cluster_batch_size x sample_batch_size
                diffs = samples[None,int(len(samples)-((len(samples)%sample_batch_size))):,:].to(device=device)-self.cluster_centers[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,None,:].to(device=device) # batch_size x num_clusters x num_features
                log_likelihoods[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,int(len(samples)-((len(samples)%sample_batch_size))):] = (-1/2) * t.einsum("ijk,ijk,ij", diffs/self.bandwidths[int(len(self.cluster_centers)-((len(self.cluster_centers)%cluster_batch_size))):,:,:].to(device=device), diffs).cpu()

        return log_likelihoods_front + log_likelihoods

    """"""
    def _cuda_initial_bandwidth(self):
        # instantiate mean holder on cuda
        mean = t.zeros(size = (self.cluster_centers.shape[1],1), device = "cuda", dtype = self.dtype)

        # compute batch_size for batch mean computation
        pynvml.nvmlInit()
        floats_available = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 2 / float(str(self.dtype)[-2:]) / 8) # using half of gpu memory
        batch_size = int((floats_available - self.cluster_centers.shape[1])/(2*self.cluster_centers.shape[1] + self.cluster_probabilities.shape[1]))
        
        # compute mean in batches
        for n in range(int(len(self.cluster_centers)/batch_size)):
            mean += t.sum(self.cluster_centers[int(n*batch_size):int((n+1)*batch_size),:].to(device = "cuda")*self.cluster_probabilities[n*batch_size:(n+1)*batch_size,:].to(device = "cuda"), dim=0)
        mean += t.sum(self.cluster_centers[int(len(self.cluster_centers)-((len(self.cluster_centers)%batch_size))):,:].to(device = "cuda")*self.cluster_probabilities[int(len(self.cluster_centers)-((len(self.cluster_centers)%batch_size))),:].to(device = "cuda"), dim=0)
        gc.collect()
        
        # instantiate covariance matrix holder on cuda
        bandwidth = t.zeros(size=(self.cluster_centers.shape[1],self.cluster_centers.shape[1]), device="cuda", dtype=self.dtype)

        # compute batch_size for batch covariance matrix computation
        pynvml.nvmlInit()
        floats_available = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 2 / float(str(self.dtype)[-2:]) / 8) # using half of gpu memory
        batch_size = int(floats_available/(2*self.cluster_centers.shape[1] + self.cluster_probabilities.shape[1]**2))
        
        # compute initial covariance matrix in batches
        for n in range(int(len(self.cluster_centers)/batch_size)):
            diffs = self.cluster_centers[int(n*batch_size):int((n+1)*batch_size),:].to(device = "cuda") - mean
            bandwidth += t.einsum("ij,kj->ik", diffs, diffs*self.cluster_probabilities[int(n*batch_size):int((n+1)*batch_size),:].to(device="cuda"))
        diffs = self.cluster_centers[int(n*batch_size):int((n+1)*batch_size),:].to(device = "cuda") - mean
        bandwidth += t.einsum("ij,kj->ik", diffs, diffs*self.cluster_probabilities[int(len(self.cluster_centers)-((len(self.cluster_centers)%batch_size))):,:].to(device="cuda"))
        
        return bandwidth # return initial covariance matrix (on cuda)


def join_distributions(distribution1:Gaussian_Mixture_Model, distribution2:Gaussian_Mixture_Model, limited_memory:bool = True, dtype = t.float64):
    # check that distribution1 and distribution2 can be joined
    if not distribution1.fitted or not distribution2.fitted: raise ArgumentError("Both Gaussian_Mixture_Model objects must have been fit before they can be joined.")
    if (distribution1.cluster_centers.shape[1] != distribution2.cluster_centers.shape[1]): raise ArgumentError("Both Gaussian_Mixture_Model objects must have been fit using the same feature space.")
    # put both distributions onto the same device:
    if distribution1.device != distribution2.device:
        distribution2.cluster_centers = distribution2.cluster_centers.to(device=distribution1.device, dtype=distribution1.dtype)
        distribution2.cluster_probabilities = distribution2.cluster_probabilities.to(device=distribution1.device, dtype=distribution1.dtype)
        distribution2.bandwidths = distribution2.bandwidths.to(device=distribution1.device, dtype=distribution1.dtype)
    
    # The current implementation only allows the use of full-sized independent covariance matrices for each cluster center. It is assumed that the user will cluster() their 
    # distributions to reduce the number of clusters before joining them.
    
    # define the new distribution
    new_distribution = Gaussian_Mixture_Model(consistent_variance = False, covariance_matrix_type = "full", dtype = dtype)
    new_distribution.device = distribution1.device
    
    if limited_memory: # will do one cluster at a time to avoid memory overflow
        # define placeholders
        new_distribution.cluster_centers = t.ones(size=(len(distribution1.cluster_centers)*len(distribution2.cluster_centers), distribution1.cluster_centers.shape[1]), device=distribution1.device, dtype=dtype)
        new_distribution.bandwidths = t.ones(size=(len(distribution1.cluster_centers)*len(distribution2.cluster_centers), distribution1.cluster_centers.shape[1], distribution1.cluster_centers.shape[1]), device=distribution1.device, dtype=dtype)
        for n in range(len(distribution1.cluster_centers)):
            for m in range(len(distribution2.cluster_centers)): # for each joined cluster, compute the new cluster center and covariance matrix
                distribution1_weight = distribution1.cluster_probabilities[n] / (distribution1.cluster_probabilities[n] + distribution2.cluster_probabilities[m])
                distribution2_weight = distribution2.cluster_probabilities[m] / (distribution1.cluster_probabilities[n] + distribution2.cluster_probabilities[m])
                inv = t.linalg.inv(distribution1_weight*distribution1.bandwidths[n] + distribution2_weight*distribution2.bandwidths[m])
                new_distribution.cluster_centers[n*len(distribution2.cluster_centers)+m] = t.matmul(distribution1.cluster_centers[n][None,:],t.matmul(inv, distribution1_weight*distribution2.bandwidths[m])) + t.matmul(distribution2.cluster_centers[m][None,:], t.matmul(inv, distribution2_weight*distribution1.bandwidths[n])) # (1,num_features)
                new_distribution.bandwidths[n*len(distribution2.cluster_centers)+m] = t.matmul(t.matmul(distribution1_weight*distribution1.bandwidths[n], inv), distribution2_weight*distribution2.bandwidths[m]) # num_features x num_features
    else: # the following section has not been tested
        distribution1_weights = distribution1.cluster_probabilities / (distribution1.cluster_probabilities + distribution2.cluster_probabilities.T) # num_clusters1 x num_clusters2
        distribution2_weights = distribution2.cluster_probabilities / (distribution1.cluster_probabilities + distribution2.cluster_probabilities.T)
        inverses = t.linalg.inv(distribution1_weights[:,:,None,None] * distribution1.bandwidths[:,None,:,:] + distribution2_weights[:,:,None,None] * distribution2.bandwidths[None,:,:,:]) # num_clusters1 x num_clusters2 x num_features x num_features
        new_distribution.cluster_centers = t.flatten(t.matmul(t.matmul(distribution1_weights[:,:,None,None] * distribution1.cluster_centers[:,None,None,:], inverses), distribution2.bandwidths[None,:,:,:]) + t.matmul(t.matmul(distribution2_weights[:,:,None,None] * distribution2.cluster_centers[None,:,None,:], inverses), distribution1.bandwidths[:,None,:,:]),start_dim=0,end_dim=1)
        new_distribution.bandwidths = t.flatten(t.matmul(t.matmul(distribution1_weights[:,:,None,None]*distribution1.bandwidths[:,None,:,:], inverses),distribution2_weights[:,:,None,None]*distribution2.bandwidths[None,:,:,:]), start_dim=0,end_dim=1)
    
    # compute all of the new cluster probabilities at once since it doesn't require as much memory
    new_distribution.cluster_probabilities = t.flatten(distribution1.cluster_probabilities * distribution2.cluster_probabilities.T)[:,None] # num_joined_clusters X 1
    new_distribution.cluster_probabilities = new_distribution.cluster_probabilities/t.sum(new_distribution.cluster_probabilities)
        
    new_distribution.fitted = True
    
    return new_distribution