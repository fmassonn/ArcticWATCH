#!/usr/bin/python
#
# Tools to perform Empirical Mode Decomposition
# 
# F. Massonnet, 22 March 2017
#
# Algorithm: https://www.clear.rice.edu/elec301/Projects02/empiricalMode/process.html
#            http://rcada.ncu.edu.tw/1.%20ON%20INTRINSIC%20MODE%20FUNCTION.pdf
#            https://en.wikipedia.org/wiki/Hilbert%E2%80%93Huang_transform
import matplotlib; matplotlib.use('Agg')
def emd(X, t = None):
  import numpy as np
  import matplotlib.pyplot as plt
  import scipy
  import sys
  from scipy import interpolate

  if type(X) is not np.ndarray:
    print("input is of type " + str(type(X)))
    sys.exit("(emd) input data is not numpy array")

  if t is None:
    t = np.arange(len(X))

  if len(X) != len(t):
    sys.exit("(emd) time and data lengths don't match")
  
  def get_imf(data, t, n_iter = 10000, criterion = 0.1, printfig = None):
    """
       Processes the EMD algorithm on "data" until the relative difference between
       successive proto-IMFs is less than criterion"
    """


    data_current = data 

    j_iter = 1
    err = criterion * 1e36

    while err > criterion and j_iter <= n_iter:
      plt.close("all")
      # 1. Locate local maxima and minima of input data
      imax = [i for i in np.arange(1, len(data_current) - 1) \
             if data_current[i] > data_current[i - 1] and data_current[i] >= data_current[i + 1]] 

      imin = [i for i in np.arange(1, len(data_current) - 1) \
             if data_current[i] < data_current[i - 1] and data_current[i] <= data_current[i + 1]] 


      # Case 1: there are less than three mins or three maxs --> finish
      if len(imin) <= 3 or len(imax) <= 3:
        print("(get_imf) not possible to get imf, less than two local max or min")
        success = False
        imf = data_current

        # print last figure
        if printfig is not None:
          figname = printfig + "-" + str(j_iter).zfill(3) + ".png"
          plt.figure(figsize = (10, 5))
          plt.title("Iteration " + str(j_iter) + "; No fit possible")
          plt.plot(t, data_current, 'k-', lw = 1)
          plt.plot(t[imax], data_current[imax], 'g.', ms = 10)
          plt.plot(t[imin], data_current[imin], 'r.', ms = 10)
       
          plt.tight_layout()
          plt.savefig(figname)
          plt.cla()
        
        break

      # Case 2: still possible to fit
      else:
        # Make mirror of data to take care of edges 
        # First, determine which of the min or max is the furthest
        index_extend_min = np.max((imin[0], imax[0])) + 1

        # Find the indices for which a mirrorring will have to be applied
        index_to_left_mirror = np.arange(index_extend_min, 0, -1)
        # What are the times to left-mirror?
        t_left = 2 * t[0] - t[index_to_left_mirror] 
        # Apply mirrorring through central symmetry
        data_left = data_current[0] + 1.0 * (t_left - t[0]) * ((data_current[index_to_left_mirror] - data_current[0]) / (t[index_to_left_mirror] - t[0]))
   
        # Repeat for right mirror
        index_extend_max = np.min((imin[-1], imax[-1])) - 1
        index_to_right_mirror = np.arange(len(data_current) - 2, index_extend_max - 1, - 1)
        t_right = 2 * t[-1] - t[index_to_right_mirror]
        data_right = data_current[-1] + 1.0 * (t_right - t[-1]) * ((data_current[-1] - data_current[index_to_right_mirror]) / (t[-1] - t[index_to_right_mirror]))

        data_current_ext = np.append(data_left, np.append(data_current, data_right))
        t_ext = np.append(t_left, np.append(t, t_right))

        # Re-compute mins and maxs
        imax_ext = [i for i in np.arange(1, len(data_current_ext) - 1) \
                if data_current_ext[i] > data_current_ext[i - 1] and data_current_ext[i] >= data_current_ext[i + 1]]
        imin_ext = [i for i in np.arange(1, len(data_current_ext) - 1) \
               if data_current_ext[i] < data_current_ext[i - 1] and data_current_ext[i] <= data_current_ext[i + 1]]



        # 2. Spline-fitting and mean envelope
        # Python proposes essentially two ways to spline curves.
        # - scipy.interpolate.spline connects the points exactly
        #up = scipy.interpolate.spline(t_ext[imax_ext], data_current_ext[imax_ext], t_ext)
        #lo = scipy.interpolate.spline(t_ext[imin_ext], data_current_ext[imin_ext], t_ext)
        # - scipy.interpolate.UnivariateSpline does a piecewise least-square fitting
        # up = scipy.interpolate.UnivariateSpline(t_ext[imax_ext], data_current_ext[imax_ext])(t_ext)
        # lo = scipy.interpolate.UnivariateSpline(t_ext[imin_ext], data_current_ext[imin_ext])(t_ext)
        #
        # - scipy.interpolate.CubicSpline
        #up = scipy.interpolate.CubicSpline(t_ext[imax_ext], data_current_ext[imax_ext], bc_type='periodic')
        #lo = scipy.interpolate.CubicSpline(t_ext[imin_ext], data_current_ext[imin_ext], bc_type='periodic')
        # - scipy.interpolate.Akima1DInterpolator
        up = scipy.interpolate.Akima1DInterpolator(t_ext[imax_ext], data_current_ext[imax_ext])(t_ext)
        lo = scipy.interpolate.Akima1DInterpolator(t_ext[imin_ext], data_current_ext[imin_ext])(t_ext)
        # I found the second approach less stable, especially when minima or maxima are all alike near
        # the end of the series; this gives rise to instabilities. 
        # The disadvantage of the first method is that it artificially adds one extra point at zero 
        # at the ends

        me = (up + lo) / 2.0
        print(len(me))
        print(len(data_current))
        # 3. proto-IMF as difference btw original data and mean envelope
        h = data_current - me[index_extend_min:index_extend_min + len(data_current)]

        # 4. Check if h can be promoted as an IMF    
        eps = 1e-20 # to avoid division by zero

        #
        err = np.nanmean(me[index_extend_min:index_extend_min + len(data_current)]) / np.std(data_current)

#        err = np.mean( ((h - data_current) / (data_current + eps)) ** 2 )
        #print("HELLO")
        #print(np.std(me[index_extend_min:index_extend_min + len(data_current)]))
        #print(me[index_extend_min:index_extend_min + len(data_current)])
        print("  get_imf: err = " + str(np.round(err, 2)) + "; j_iter = " + str(j_iter).zfill(3))

        if printfig is not None:
          figname = printfig + "-" + str(j_iter).zfill(3) + ".png"
          plt.figure(figsize = (10, 5))
#          plt.xlim((30, 70)); plt.ylim(-10000, 10000)
          plt.plot(t_ext, data_current_ext, color = (0.5, 0.5, 0.5), lw = 1, linestyle = ":")
          plt.plot(t_ext[imax_ext], data_current_ext[imax_ext], '.', color = (0.6, 1.0, 0.6), ms = 8)
          plt.plot(t_ext[imin_ext], data_current_ext[imin_ext], '.', color = (1.0, 0.6, 0.6), ms = 8)
          plt.plot(t, data_current, 'k-', lw = 2)
          plt.plot(t[imax], data_current[imax], 'g.', ms = 10)
          plt.plot(t[imin], data_current[imin], 'r.', ms = 10)

          plt.plot(t_ext, up, 'g--')
          plt.plot(t_ext, lo, 'r--')
          plt.plot(t_ext, me, color = (0.5, 0.5, 0.5), lw = 2)
          plt.plot(t, h, 'b')
          plt.title("Iteration " + str(j_iter) + "; Err = " + str(err))
          plt.tight_layout()
          plt.savefig(figname)
          plt.cla()

        if err <= criterion:
          imf = h
          success = True
          break
        else:
          # Repeat with now h as the input data
          data_current = h
          j_iter += 1
   
       
    return imf, success

  IMF = list()

  j_IMF = 0

  while j_IMF < 100:
    print("--> IMF #" + str(j_IMF + 1))
    if j_IMF == 0:
      datain = X
    else:
      datain = np.asarray([X[j] - np.sum([i[j] for i in IMF]) for j in range(len(X))])

    tmp, success = get_imf(datain, t, n_iter = 100, printfig = "./figs/IMF-" + str(j_IMF + 1).zfill(3))
    IMF.append(tmp)
    

    print("Attempt to IMF led to success: " + str(success))
    print("IMF: ")
    print(tmp)
    if not success:
      break

    j_IMF += 1

  # Print all Figures
  print("FRRRRR")
  fig = plt.figure(figsize = (8, 5))
  f, a = plt.subplots(1 + len(IMF), sharex = True, figsize = (7, 14))
  a[0].plot(t, X, 'k')
  a[0].plot(t, np.asarray([np.sum([i[j] for i in IMF]) for j in range(len(X))]), 'r')
  a[0].set_ylim(np.min((np.min(X), np.min(IMF))), np.max((np.max(X), np.max(IMF))))

  for i in range(len(IMF)):
    a[i + 1].plot(t, np.zeros((len(t))), 'k--')
    a[i + 1].plot(t, IMF[i], 'b')
    a[i + 1].set_ylim(np.min((np.min(X), np.min(IMF))), np.max((np.max(X), np.max(IMF))))
  plt.savefig("./figs/all.png", dpi = 500)
  plt.cla()

  return IMF

def example():
  import numpy as np
  import matplotlib.pyplot as plt 
  import random
  
  np.random.seed(1)
  nt =  12 * 37     # nb time steps
  t = 1979 + np.arange(nt) / 12.0 
  #X = 0.5 * np.random.randn(nt) + 2.0 / nt * t + np.sin(2 * np.pi * t / (80.0 + 10 * np.sin(2 * np.pi * t / 50)))
  n = np.empty(nt)
  n[0] =  0.2 * np.random.randn(1)
  a = 0.8
  for jt in np.arange(1, nt):
    n[jt] = a * n[jt - 1] + 0.2 * np.random.randn(1)
  
  # Mimicking the Arctic sea ice extent
  X =  11.0 + (- (t - 1979) * 0.2)  + 4.0 * np.sin(2 * np.pi * t )   + n                 
       # Trend -------------------
       #                              Annual cycle ---------------
       #                                                              Persistent noise
  emd(X, t)

def help():
  print("Help not coded yet")

if __name__ == '__main__':
  help()
  example()
