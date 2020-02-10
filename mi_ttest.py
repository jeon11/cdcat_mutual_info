from scipy import stats
from mi_accre import get_avg_mi

nScans = 187
nRuns = 8

cc1, mi1 = get_avg_mi('cdcatmr066', 'rtf', verbose=True)  # register to first scan
cc2, mi2 = get_avg_mi('cdcatmr066', 'rtm', verbose=True)  # register to mean

assert len(cc1) == len(cc2) == nScans * nRuns
assert len(mi1) == len(mi2) == nScans * nRuns

cc_t, cc_p = stats.ttest_ind(cc1, cc2)
mi_t, mi_p = stats.ttest_ind(mi1, mi2)


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# sns.set(color_codes=True)
sns.set_color_codes('pastel')
# sns.set_palette("tab10")

plt.figure(0)
sns.distplot(cc1, label='rtf');
sns.distplot(cc2, label='rtm');
plt.legend()
plt.title('correlation coefficient n=' + str(len(cc1)))
plt.savefig('/home/jeonj1/proj/mi/cc_distplot.png')

plt.figure(1)
sns.distplot(mi1, label='rtf');
sns.distplot(mi2, label='rtm');
plt.legend()
plt.title('mutual info | n=' + str(len(mi1)))
plt.savefig('/home/jeonj1/proj/mi/mi_distplot.png')


# let's plot box-dist plot combined
f, (ax_box1, ax_box2, ax_dist) = plt.subplots(3, sharex=True, gridspec_kw= {"height_ratios": (0.3, 0.3, 1)})

cc1_mean = np.mean(cc1)
cc2_mean = np.mean(cc2)

sns.boxplot(cc1, ax=ax_box1, color='b')
sns.boxplot(cc2, ax=ax_box2, color='r')
ax_box1.axvline(cc1_mean, color='g', linestyle='--')
ax_box2.axvline(cc2_mean, color='m', linestyle='--')
plt.subplots_adjust(top=0.87)
plt.suptitle('correlation coefficient n=' + str(len(cc1)), fontsize = 16)

sns.distplot(cc1, ax=ax_dist, label='rtf', color='b', norm_hist=True)
sns.distplot(cc2, ax=ax_dist, label='rtm', color='r', norm_hist=True)
ax_dist.axvline(cc1_mean, color='g', linestyle='--')
ax_dist.axvline(cc2_mean, color='m', linestyle='--')
plt.legend()
ax_box1.set(xlabel='')
ax_box2.set(xlabel='')
# plt.title('correlation coefficient n=' + str(len(cc1)))
plt.savefig('/home/jeonj1/proj/mi/cc_box_distplot.png')


# let's plot box-dist plot combined
f, (ax_box1, ax_box2, ax_dist) = plt.subplots(3, sharex=True, gridspec_kw= {"height_ratios": (0.3, 0.3, 1)})

mi1_mean = np.mean(mi1)
mi2_mean = np.mean(mi2)

sns.boxplot(mi1, ax=ax_box1, color='b')
sns.boxplot(mi2, ax=ax_box2, color='r')
ax_box1.axvline(mi1_mean, color='g', linestyle='--')
ax_box2.axvline(mi2_mean, color='m', linestyle='--')
plt.subplots_adjust(top=0.87)
plt.suptitle('mutual information n=' + str(len(cc1)), fontsize = 16)

sns.distplot(mi1, ax=ax_dist, label='rtf', color='b', norm_hist=True)
sns.distplot(mi2, ax=ax_dist, label='rtm', color='r', norm_hist=True)
ax_dist.axvline(mi1_mean, color='g', linestyle='--')
ax_dist.axvline(mi2_mean, color='m', linestyle='--')
plt.legend()
ax_box1.set(xlabel='')
ax_box2.set(xlabel='')
# plt.title('correlation coefficient n=' + str(len(cc1)))
plt.savefig('/home/jeonj1/proj/mi/mi_box_distplot.png')
