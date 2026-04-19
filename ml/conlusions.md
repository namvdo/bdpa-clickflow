### Conclusions

Additional feature engineering experiments such as applying sin/cos transformations to the day number, deriving weekday, adding a month variable, and including or excluding price 2, led to degraded clustering performance, suggesting that the **month variable introduces noise** and does not capture meaningful event-level patterns. The slight improvement observed in configuration without "day" also indicates that a similar issue apply to the **day number feature**.

Overall, findings suggest that simpler feature representations are more effective.

Although the day variable may contain some temporal information, its inclusion does not improve clustering quality and slightly degrades performance (see experiment 0. and 1. conclusions). Therefore, it is excluded from the final model to ensure more compact and well-separated clusters. Peak performance: Without day K = 8 [Experiment 1](full1.ipynb), with day K = 12 [Experiment 0](full.ipynb) and 15 components. 

Less clusters indicate: Cleaner structure, fewer and stronger clusters and better separation (higher silhouette). With "day" more clusters needed, slightly worse metrics, most likely capturing noise or minor variations, although periodical transform for day with sin + cos should give in general better results than naive minmax, in this case it was tested and it doesn't, basic MinMax scaling gives better results.