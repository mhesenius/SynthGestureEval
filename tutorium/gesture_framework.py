#!pip install numpy
#!pip install copy
#!pip install scipy
#!pip install tensorflow

import numpy as np
import copy
import math
from scipy.optimize import linear_sum_assignment
import tensorflow as tf

class gesture_framework:
    # expectation of gesture: [[[x, y, t], [x, y, t], ...], [[x, y, t], ...], ...]
    def __init__(self):
        return
    
    def compare_physical_properties(self, originals, synthetic, bin_no):
        kl_d = {}
        pp_o = get_physical_properties(originals)
        pp_s = get_physical_properties(synthetic)
        for key in pp_o:
            dist_o = pp_o[key]
            dist_s = pp_s[key]
            minimum = np.min([np.min(dist_o), np.min(dist_s)])
            maximum = np.max([np.max(dist_o), np.max(dist_s)])
            if minimum == maximum:
                print(key)
            bins_o = get_binned_population(dist_o, bin_no, maximum, minimum)
            bins_s = get_binned_population(dist_s, bin_no, maximum, minimum)
            kl_d[key] = get_kl_distance(bins_o, bins_s)
        return kl_d
    
    def get_wasserstein_distance(self, originals, synthetic):
        return calc_WD(originals, synthetic)
    
    def tstr(self, originals, synthetic, labels, file, classifier_o, epochs=20):
        '''trains an untrained classifier on the synthetic data with early stopping and returns synthetic test accuracy and original test accuracy on this synthetic classifier, as well as an evaluation of a given classifier trained on real data'''
        classifier = tf.keras.models.load_model(file)
        classifier.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss = 'categorical_crossentropy', metrics=['accuracy'])
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        train = synthetic[:int(0.8*synthetic.shape[0])]
        valid = synthetic[int(0.8*synthetic.shape[0]):int(0.9*synthetic.shape[0])]
        test = synthetic[int(0.9*synthetic.shape[0]):]
        classifier.fit(train, labels[:int(0.8*synthetic.shape[0])], validation_data=(valid, labels[int(0.8*synthetic.shape[0]):int(0.9*synthetic.shape[0])]), epochs=epochs, callbacks=[callback], verbose=1)
        _, test_acc = classifier.evaluate(test, labels[int(0.9*synthetic.shape[0]):], verbose = 0)
        _, real_acc = classifier.evaluate(originals, labels, verbose = 0)
        _, real_acc_r = classifier_o.evaluate(originals, labels, verbose = 0)
        return {'synth classifier synth acc:': test_acc, 'synth classifier real acc:': real_acc, 'real classifier real acc:': real_acc_r}
    
    def nnad_novelty(self, gesturesP, gesturesQ):
        nn = 0
        for g in gesturesP:
            g = g[0]
            if len(g) < 1:
                continue
            nn += nnad(g, gesturesQ)
        return nn/len(gesturesP)
    
    def nnad_diversity(self, gestures):
        nn = 0
        for g in gestures:
            gc = copy.copy(gestures)
            if len(g[0]) < 1:
                continue
            gc.remove(g)
            g = g[0]
            nn += nnad(g, gc)
        return nn/len(gestures)
    
    def trts(self, synthetic, labels, classifier):
        _, test_acc = classifier.evaluate(synthetic, labels, verbose = 0)
        return test_acc

    def dau(self, originals, synthetic, labels, file, portions=[0.1, 0.2, 0.3, 0.4, 0.5], epochs=20):
        train = originals[:int(0.8*originals.shape[0])]
        valid = originals[int(0.8*originals.shape[0]):int(0.9*originals.shape[0])]
        test = originals[int(0.9*originals.shape[0]):]
        results = []
        for p in portions:
            if p > 1:
                return 'too big portion'
            classifier = tf.keras.models.load_model(file)
            classifier.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss = 'categorical_crossentropy', metrics=['accuracy'])
            trainset = np.append(train,synthetic[:int(p*originals.shape[0])], axis=0)
            labelset = np.append(labels[:int(0.8*originals.shape[0])],labels[:int(p*originals.shape[0])], axis=0)
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
            classifier.fit(trainset, labelset, validation_data=(valid, labels[int(0.8*len(originals)):int(0.9*len(originals))]), epochs=epochs, callbacks=[callback], verbose=1)
            results.append(classifier.evaluate(test, labels[int(0.9*len(originals)):], verbose=0))
        return results
    
    def temporal_depencies(self, synthetics, classifiers, shapes, min_l):
        mses = []
        for i, c in enumerate(classifiers):
            s = shapes[i]
            inputs, outputs = [], []
            for g in synthetics:
                max_idx = len(g) - min_l
                if max_idx < 1:
                    continue
                idx = np.random.randint(max_idx)
                inputs.append(g[idx:idx+s[0]])
                outputs.append(g[idx+s[0]:idx+s[0]+s[1]])
            inputs = np.asarray(inputs)
            outputs = np.asarray(outputs)
            _, mse = c.evaluate(inputs, outputs, verbose=0)
            mses.append(mse)
        return mses

def get_physical_properties(gestures):
    '''takes gestures in form of [[[]]] and computes the physical properties'''
    pp = {'v_mean': [], 'a_mean': [], 'v_median_3f': [], 'v_median_3l': [], 'a_median_3f': [], 'a_median_3l': [], 'v_20': [], 'v_80': [], 'a_20': [], 'a_80': [], 'mrl': [], 'area': [], 'length': [], 'time': [], 'bending': []}
    for x in gestures:
        v_profile_all = []
        a_profile_all = []
        all_points = []
        z = []
        bending_all = []
        v_median_3f = []
        v_median_3l = []
        a_median_3f = []
        a_median_3l = []
        eucl_sum = 0
        time = 0
        for g in x:
            all_points = all_points + g
            g = np.asarray(g)
            if len(g) < 4:
                continue
            # v Profil und a Profil
            v_profile = []
            for i in range(g.shape[0]-1):
                eucl_sum += eucl_dist(g[i], g[i+1])
                diff = g[i+1]-g[i]
                if diff[0] == 0 and diff[1] == 0:
                    v_profile.append(0)
                    continue
                v_profile.append(eucl_dist(g[i], g[i+1]))
                if i < g.shape[0]-2:
                    diff2 = g[i+2] - g[i+1]
                    arccos = np.dot(diff, diff2)/(np.sqrt(diff[0]**2 + diff[1]**2)*np.sqrt(diff2[0]**2 + diff2[1]**2))
                    arccos = max(-1, min(arccos, 1))
                    theta = np.arccos(arccos)
                    bending_all.append(theta)
                    z.append(np.exp(1j*theta))
            v_profile_all.extend(v_profile)
            v_profile = np.asarray(v_profile)
            a_profile = []
            for i in range(v_profile.shape[0]-1):
                a_profile.append(np.abs(v_profile[i] - v_profile[i+1]))
            a_profile_all.extend(a_profile)
            a_profile = np.asarray(a_profile)
            # median
            v_median_3f.append(float(get_percentile(copy.copy(v_profile[:3]), 0.5)))
            v_median_3l.append(float(get_percentile(copy.copy(v_profile[-3:]), 0.5)))
            a_median_3f.append(float(get_percentile(copy.copy(a_profile[:3]), 0.5)))
            a_median_3l.append(float(get_percentile(copy.copy(a_profile[-3:]), 0.5)))
            time += len(g)
        # durchschnitt mehrerer strokes berechnen
        if len(v_median_3f) < 1:
            continue
        pp['mrl'].append(np.abs(np.mean(z).real))
        pp['v_mean'].append(np.mean(v_profile_all))
        pp['a_mean'].append(np.mean(a_profile_all))
        pp['v_median_3f'].append(np.mean(v_median_3f))
        pp['v_median_3l'].append(np.mean(v_median_3l))
        pp['a_median_3f'].append(np.mean(a_median_3f))
        pp['a_median_3l'].append(np.mean(a_median_3l))
        # 20 & 80% Perzentil v
        pp['v_20'].append(float(get_percentile(copy.copy(v_profile_all), 0.2)))
        pp['v_80'].append(float(get_percentile(copy.copy(v_profile_all), 0.8)))
        # 20 & 80% Perzentil a
        pp['a_20'].append(float(get_percentile(copy.copy(a_profile_all), 0.2)))
        pp['a_80'].append(float(get_percentile(copy.copy(a_profile_all), 0.8)))
        all_points = np.asarray(all_points)
        rect1 = (np.min(all_points[:,0]), np.min(all_points[:,1]), np.max(all_points[:,0]), np.max(all_points[:,1]))
        pp['area'].append(get_area_rect(rect1))
        pp['length'].append(eucl_sum)
        pp['time'].append(time)
        pp['bending'].append(np.mean(bending_all))
    return pp

def get_percentile(arr, p):
    arr = np.sort(arr)
    l = arr.shape[0]
    if l%2==0:
        return (arr[int(l*p)-1] + arr[int(l*p)])/2
    return arr[int(l*p)]

def eucl_dist(x, y):
    x_ = y[0]-x[0]
    y_ = y[1]-x[1]
    return np.sqrt(x_*x_ + y_*y_)

def get_area_rect(rect):
  'rect = (x_min y_min x_max y_max)'
  return (rect[2] - rect[0])*(rect[3] - rect[1])

def get_binned_population(distribution, bins_no, maximum, minimum):
    bins = np.zeros(bins_no)
    if minimum == maximum:
        return bins
    for d in distribution:
        index = int((bins_no-1)*(d-minimum)/(maximum-minimum))
        bins[index] += 1
    bins = np.asarray(bins)/np.sum(bins)
    return bins

def get_kl_distance(bins1, bins2):
    kl_sum = 0
    bins1 += 0.000000001
    bins2 += 0.000000001
    for i in range(len(bins1)):
        kl_sum += bins1[i]*math.log(bins1[i]/bins2[i])
    return kl_sum

def dtw(x, x_prime, R, dim=2, q=1):
  inf = 2**32
  for i in range(len(x)):
    for j in range(len(x_prime)):
      if dim==2:
        R[i, j] = eucl_dist(x[i], x_prime[j])
      else:
        R[i, j]= np.abs(x[i]-x_prime[j])
      if i > 0 or j > 0:
        R[i, j] += min(
          R[i-1, j  ] if i > 0             else inf,
          R[i  , j-1] if j > 0             else inf,
          R[i-1, j-1] if (i > 0 and j > 0) else inf
          # Note that these 3 terms cannot all be
          # inf if we have (i > 0 or j > 0)
        )

  return R[-1, -1] ** (1. / q)

#Wasserstein Distance
def calc_WD(real, fake):
    real_wd, fake_wd = [], []
    for i in range(len(real)):
        if not len(real[i][0]) < 1:
            real_wd.append(real[i][0])
    for i in range(len(fake)):
        if not len(fake[i][0]) < 1:
            fake_wd.append(fake[i][0])
    a = calc_cdist(real_wd, fake_wd)
    rows, cols = linear_sum_assignment(a)
    w_d = np.sum(a[rows, cols])
    return w_d

def calc_cdist(a, b):
  result = np.zeros(shape=[len(a), len(b)])
  for i, _ in enumerate(a):
      for j, _ in enumerate(b):
        R = np.zeros(shape=[len(a[i]), len(b[j])])
        result[i, j] = dtw(a[i], b[j], R)
  return result

def nnad(gesture, compare):
  minimum = 2**32
  gesture = np.asarray(gesture)
  for c in compare:
    c = c[0]
    if len(c) < 1:
        continue
    c = np.asarray(c)
    R = np.zeros(shape=[gesture.shape[0], c.shape[0]])
    d = dtw(gesture, c, R)
    if d <= minimum:
      minimum = d
  return minimum